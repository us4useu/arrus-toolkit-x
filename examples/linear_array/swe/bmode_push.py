"""
This script acquires and reconstructs RF img for plane wave imaging
(synthetic aperture).

GPU usage is recommended.
"""

import arrus
import arrus.session
import arrus.utils.imaging
import arrus.utils.us4r
import queue
import numpy as np
import arrus.ops.tgc
import arrus.medium
from collections import deque
import pickle
import time
import sys
from arrus.ops.us4r import Scheme, Pulse, DataBufferSpec
from arrus.ops.imaging import PwiSequence
from arrus.utils.gui import Display2D
from arrus.utils.imaging import get_bmode_imaging, get_extent

from arrus.ops.us4r import (
    Scheme,
    Pulse,
    Tx,
    Rx,
    TxRx,
    TxRxSequence
)
from arrus.utils.imaging import (
    Pipeline,
    SelectFrames,
    Squeeze,
    Lambda,
    RemapToLogicalOrder
)
from arrus.utils.gui import (
    Display2D
)

session_cfg = "us4r.prototxt"

arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("swe_test.log", arrus.logging.TRACE)


def run_bmode():
    # Here starts communication with the device.
    medium = arrus.medium.Medium(name="cirs049a", speed_of_sound=1540)
    with arrus.Session(session_cfg, medium=medium) as sess:
        us4r = sess.get_device("/Us4R:0")
        us4r.set_hv_voltage(40)

        sequence = PwiSequence(
            angles=np.linspace(-10, 10, 32) * np.pi / 180,
            pulse=Pulse(center_frequency=6e6, n_periods=2, inverse=False),
            rx_sample_range=(256, 1024 * 4),
            downsampling_factor=1,
            speed_of_sound=1450,
            pri=200e-6,
            tgc_start=14,
            tgc_slope=2e2)

        # Imaging output grid.
        x_grid = np.arange(-15, 15, 0.1) * 1e-3
        z_grid = np.arange(5, 35, 0.1) * 1e-3
        scheme = Scheme(
            tx_rx_sequence=sequence,
            processing=get_bmode_imaging(sequence=sequence, grid=(x_grid, z_grid)))
        # Upload sequence on the us4r-lite device.
        buffer, metadata = sess.upload(scheme)
        display = Display2D(metadata=metadata, value_range=(20, 80), cmap="gray",
                            title="B-mode", xlabel="OX (mm)", ylabel="OZ (mm)",
                            extent=get_extent(x_grid, z_grid) * 1e3,
                            show_colorbar=True)
        sess.start_scheme()
        display.start(buffer)
    # When we exit the above scope, the session and scheme is properly closed.
    print("Stopping the example.")


def run_push_sequence(dataset_id: str):
    # Parameters
    id = 0
    push_hv = 50
    push_freq = np.array([130.0 / 26 * 1e6, 130.0 / 26 * 1e6, 130.0 / 26 * 1e6])
    push_length = np.array([200e-6, 200e-6, 200e-6])
    push_txelements = np.array([32, 64, 128])
    push_focus = np.array([15e-3, 25e-3, 38e-3])

    pwi_hv = push_hv + 20
    pwi_freq = 130.0 / 26 * 1e6
    pwi_txncycles = 2
    pwi_pri = 100e-6
    pwi_angles = [0.0]
    n_samples = 4 * 1024 - 256
    imaging_pw_n_repeats = 60  # ile faktycznie ramek 128 kanałowych złapać (liczba strzałów jest 2x wieksza)

    # Process parameters
    datafile = "data_id_" + str(id)
    push_txncycles = push_length * push_freq
    push_txncycles.astype(int)
    if (pwi_hv < push_hv):
        pwi_hv = push_hv + 2

    hv_voltage_0 = pwi_hv
    hv_voltage_1 = push_hv
    push_pri = push_length + 50e-6

    ##### Here starts communication with the device. ######
    medium = arrus.medium.Medium(name="cirs049a", speed_of_sound=1540)
    with arrus.Session(session_cfg, medium=medium) as sess:

        us4r = sess.get_device("/Us4R:0")
        n_elements = us4r.get_probe_model().n_elements

        # Set the HVPS HV voltages
        us4r.set_hv_voltage((hv_voltage_0, hv_voltage_0), (hv_voltage_1, hv_voltage_1))

        # Configure push transmits
        push_sequence = [0] * len(push_freq)
        push_rx_aperture = arrus.ops.us4r.Aperture(center=0, size=0)  # empty rx aperture for push

        for i in range(len(push_freq)):
            push_tx_aperture = arrus.ops.us4r.Aperture(center=0, size=push_txelements[i])

            push_sequence[i] = TxRx(
                # NOTE: full transmit aperture.
                Tx(aperture=push_tx_aperture,  # to jest maska binarna (długośc n_elements)
                   excitation=Pulse(center_frequency=push_freq[i], n_periods=int(push_txncycles[i]), inverse=False,
                                    amplitude_level=1),
                   focus=push_focus[i],  # [m]
                   angle=0,  # [rad]
                   speed_of_sound=medium.speed_of_sound
                   ),
                Rx(
                    aperture=push_rx_aperture,
                    sample_range=(0, n_samples),
                    downsampling_factor=1
                ),
                pri=push_pri[i]
            )
        imaging_sequence = [
                               TxRx(
                                   Tx(
                                       aperture=arrus.ops.us4r.Aperture(center=0),
                                       excitation=Pulse(center_frequency=pwi_freq, n_periods=pwi_txncycles,
                                                        inverse=False, amplitude_level=0),
                                       focus=np.inf,  # [m]
                                       angle=angle / 180 * np.pi,  # [rad]
                                       speed_of_sound=medium.speed_of_sound
                                   ),
                                   Rx(
                                       aperture=arrus.ops.us4r.Aperture(center=0),
                                       sample_range=(0, n_samples),
                                       downsampling_factor=1
                                   ),
                                   pri=pwi_pri
                               )
                               for angle in pwi_angles
                           ] * imaging_pw_n_repeats

        seq = TxRxSequence(ops=push_sequence + imaging_sequence, sri=1)  # SSI pushes + imaging

        # Declare the complete scheme to execute on the devices.
        scheme = Scheme(
            # Run the provided sequence.
            tx_rx_sequence=seq,
            work_mode="MANUAL",
            # Processing pipeline to perform on the GPU device.
            processing=Pipeline(
                steps=(
                    RemapToLogicalOrder(),
                ),
                placement="/GPU:0"
            )
        )

        # Upload the scheme on the us4r-lite device.
        buffer, metadata = sess.upload(scheme)
        us4r.set_tgc(arrus.ops.tgc.LinearTgc(start=52, slope=2e2))
        sess.run()
        data = buffer.get()[0]
        print("Waiting for another trigger")

        sess.stop_scheme()
        # Save last data to the file at the end
        pickle.dump({"data": data, "metadata": metadata}, open(datafile + ".pkl", "wb"))

        print("Display closed, stopping the script.")

    # When we exit the above scope, the session and scheme is properly closed.
    print("Stopping the example.")


if __name__ == "__main__":
    run_bmode()
    dataset_id = sys.argv[1]
    run_push_sequence(dataset_id)

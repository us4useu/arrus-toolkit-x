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

arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("swe_test.log", arrus.logging.TRACE)

datafile          = "datasets/data_id0e"

def main():
    # Here starts communication with the device.
    medium = arrus.medium.Medium(name="cirs049a", speed_of_sound=1540)
    with arrus.Session("us4r.prototxt", medium=medium) as sess:
        us4r = sess.get_device("/Us4R:0")
        us4r.set_hv_voltage(21)  # ew opakowac to w petle i inkrementować ze sleepami napiecie
        us4r.set_hv_voltage(41)
        # us4r.set_hv_voltage(51)
        us4r.set_hv_voltage((52, 52), (51, 51))

        push_pulse_length = 200e-6  # [s]
        push_pri = push_pulse_length + 100e-6

        n_elements = us4r.get_probe_model().n_elements
        n_samples = 4*1024-256  #4096
        tx_frequency = 4.4e6
        push_pulse_n_periods = push_pulse_length*tx_frequency
        
        # Calc settings
        tx_pri = 100e-6
        
        # Make sure a single TX/RX for the push sequence will be applied.
        push_tx_aperture = arrus.ops.us4r.Aperture(center=0)
        push_rx_aperture = arrus.ops.us4r.Aperture(center=0, size=0)
        push_sequence = [
            TxRx(
                # NOTE: full transmit aperture.
                Tx(aperture=push_tx_aperture,  # to jest maska binarna (długośc n_elemenmts)
                    excitation=Pulse(center_frequency=tx_frequency, n_periods=push_pulse_n_periods, inverse=False, amplitude_level=1),
                    focus=25e-3,  # [m]
                    angle=0,  # [rad]
                    speed_of_sound=medium.speed_of_sound
                ),
                Rx(
                    aperture=push_rx_aperture,
                    sample_range=(0, n_samples),
                    downsampling_factor=1
                ),
                pri=push_pri
            ),
            # TxRx(
            #     # NOTE: full transmit aperture.
            #     Tx(aperture=[True] * n_elements,
            #        excitation=Pulse(center_frequency=tx_frequency, n_periods=32, inverse=False),
            #        focus=10e-3,  # [m]
            #        angle=0,  # [rad]
            #        speed_of_sound=medium.speed_of_sound
            #        ),
            #     Rx(
            #         aperture=push_rx_aperture,
            #         sample_range=(0, n_samples),
            #         downsampling_factor=1
            #     ),
            #     pri=500e-6
            # )
        ]
        # angles = [-10, 0, 10]  # deg
        angles = [-4.0, 0.0, 4.0]
        imaging_pw_n_repeats = 60    # ile faktycznie ramek 128 kanałowych złapać (liczba strzałów jest 2x wieksza)
        imaging_sequence = [
            TxRx(
                Tx(
                    aperture=arrus.ops.us4r.Aperture(center=0),
                    excitation=Pulse(center_frequency=tx_frequency, n_periods=1, inverse=False, amplitude_level=0),
                    focus=np.inf,  # [m]
                    angle=angle/180*np.pi,  # [rad]
                    speed_of_sound=medium.speed_of_sound
                ),
                Rx(
                    aperture=arrus.ops.us4r.Aperture(center=0),
                    sample_range=(0, n_samples),
                    downsampling_factor=1
                ),
                pri=tx_pri
            )
            for angle in angles
            ]*imaging_pw_n_repeats

        #seq = TxRxSequence(ops=push_sequence+imaging_sequence, sri=1)   # push + imaging
        seq = TxRxSequence(ops=imaging_sequence, sri=1)   # imaging ONLY (for SNR evaluation)
        
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
        #time.sleep(3)
        #sess.run()
        #data = buffer.get()[0]
        sess.stop_scheme()
        # Save last data to the file at the end
        pickle.dump({"data":data, "metadata": metadata}, open(datafile + ".pkl", "wb"))

        
        print("Display closed, stopping the script.")

    # When we exit the above scope, the session and scheme is properly closed.
    print("Stopping the example.")


if __name__ == "__main__":
    main()

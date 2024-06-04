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


def main():
    # Here starts communication with the device.
    medium = arrus.medium.Medium(name="water", speed_of_sound=1490)
    with arrus.Session("us4r.prototxt", medium=medium) as sess:
        us4r = sess.get_device("/Us4R:0")
        us4r.set_hv_voltage(30)  # ew opakowac to w petle i inkrementować ze sleepami napiecie

        n_elements = us4r.get_probe_model().n_elements
        #n_samples = 3*1024  #4096
        
        tx_frequency = 5e6
        push_pulse_length = 200e-6  # [s]
        push_pulse_n_periods = push_pulse_length*tx_frequency
        
        # Calc settings
        #tx_pri = np.max([100e-6, push_pulse_length+200e-6])
        tx_pri = 800e-6
        fs = 65e6
        s = (push_pulse_length * fs)
        s = int(s/64)
        s = s * 64
        n_samples = np.max([2*1024, s]) 
        
        print(n_samples)
        print(tx_pri/1e-6)
        
        print(push_pulse_n_periods)
        # Make sure a single TX/RX for the push sequence will be applied.
        push_rx_aperture = [False]*n_elements
        push_sequence = [
            TxRx(
                # NOTE: full transmit aperture.
                Tx(aperture=[True]*n_elements,  # to jest maska binarna (długośc n_elemenmts)
                    excitation=Pulse(center_frequency=tx_frequency, n_periods=push_pulse_n_periods, inverse=False, amplitude_level=1),
                    focus=20e-3,  # [m]
                    angle=0,  # [rad]
                    speed_of_sound=medium.speed_of_sound
                ),
                Rx(
                    aperture=push_rx_aperture,
                    sample_range=(0, n_samples),
                    downsampling_factor=1
                ),
                pri=0.9
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
        imaging_pw_n_repeats = 2    # ile faktycznie ramek 128 kanałowych złapać (liczba strzałów jest 2x wieksza)
        imaging_sequence = [
            TxRx(
                Tx(
                    aperture=[True]*n_elements,
                    excitation=Pulse(center_frequency=tx_frequency, n_periods=2, inverse=False, amplitude_level=0),
                    focus=np.inf,  # [m]
                    angle=0,  # [rad]
                    speed_of_sound=medium.speed_of_sound
                ),
                Rx(
                    aperture=[True]*n_elements,
                    sample_range=(0, n_samples),
                    downsampling_factor=1
                ),
                pri=tx_pri
            )
            ]*imaging_pw_n_repeats

        seq = TxRxSequence(ops=push_sequence+imaging_sequence, sri=1)   # push + imaging
        # Declare the complete scheme to execute on the devices.
        q = deque(maxlen=1)
        scheme = Scheme(
            # Run the provided sequence.
            tx_rx_sequence=seq,
            # Processing pipeline to perform on the GPU device.
            processing=Pipeline(
                steps=(
                    RemapToLogicalOrder(),
                    Lambda(lambda data: (q.append(data.get()), data)[1]),
                    Squeeze(),
                    SelectFrames([1]),
                    Squeeze(),
                ),
                placement="/GPU:0"
            )
        )
        # Upload the scheme on the us4r-lite device.
        buffer, metadata = sess.upload(scheme)
        us4r.set_tgc(arrus.ops.tgc.LinearTgc(start=34, slope=2e2))
        # Created 2D image display.
        display = Display2D(metadata=metadata, value_range=(-100, 100))
        # Start the scheme.
        sess.start_scheme()
        # Start the 2D display.
        # The 2D display will consume data put the the input queue.
        # The below function blocks current thread until the window is closed.
        display.start(buffer)
        np.save("data.npy", np.stack(q))
        print("Display closed, stopping the script.")

    # When we exit the above scope, the session and scheme is properly closed.
    print("Stopping the example.")


if __name__ == "__main__":
    main()

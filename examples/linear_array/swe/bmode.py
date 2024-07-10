######################################################################################
# Realtime B-mode imaging using coherent plane wave compounding (CPWC) method
# Example script
#
# File: bmode.py
# Author: Damian Cacko, Piotr Jarosik, us4us Ltd.
# Revision: 1.0.0
#
# Aim of the example is to demonstrate how to:
#    - configure basic CPWC TX/RX sequence
#    - perform B-mode imaging processing using arrus packages
#
# Description:
#    This script configures the acquisition as coherent plane-wave compounding (CPWC). 
#    After configuration of the TX/RX sequence, the script runs the acqusution and 
#    performs the processing to obtain greyscale B-mode images and displays them 
#    in real-time. This script can be used for SWE purposes to set the position of
#    the probe over the desired target or phantom area.
#
######################################################################################


# Import all the required packages [TBD]
import arrus
import arrus.session
import arrus.utils.imaging
import arrus.utils.us4r
import arrus.ops.tgc
import arrus.medium

from arrus.ops.us4r import (
    Scheme,
    Pulse,
    Tx,
    Rx,
    TxRx,
    TxRxSequence,
    DataBufferSpec
)

from arrus.ops.imaging import PwiSequence

from arrus.utils.imaging import (
    Pipeline,
    SelectFrames,
    Squeeze,
    Lambda,
    RemapToLogicalOrder,
    get_bmode_imaging,
    get_extent
)
from arrus.utils.gui import (
    Display2D
)

import queue
import numpy as np
from collections import deque
import time

# Logging settings
arrus.set_clog_level(arrus.logging.DEBUG)
arrus.add_log_file("bmode.log", arrus.logging.TRACE)


def main():

    session_cfg = "us4r_esaote.prototxt"
    medium = arrus.medium.Medium(name="cirs049a", speed_of_sound=1540.0)
    
    # Here starts communication with the device.
    with arrus.Session(session_cfg, medium=medium) as sess:
        us4r = sess.get_device("/Us4R:0")
        
        # Set the HV voltage
        us4r.set_hv_voltage(10)

        # Create the PWI TX/RX sequence
        sequence = PwiSequence(
            angles=np.linspace(-10, 10, 32) * np.pi / 180,   # 32 angles between -10 and 10 degrees
            pulse=Pulse(center_frequency=9.2e6, n_periods=2, inverse=False),
            rx_sample_range=(256, 1024 * 5),
            downsampling_factor=1,
            speed_of_sound=medium.speed_of_sound,
            pri=200e-6,
            tgc_start=14,
            tgc_slope=2e2)

        # Create the imaging output grid.
        x_grid = np.arange(-28, 28, 0.1) * 1e-3
        z_grid = np.arange(5, 50, 0.1) * 1e-3

        # Declare the complete scheme to execute on the devices
        scheme = Scheme(
            tx_rx_sequence=sequence,
            processing=get_bmode_imaging(sequence=sequence, grid=(x_grid, z_grid))
        )
        
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


if __name__ == "__main__":
    main()

# Script ends here
######################################################################################

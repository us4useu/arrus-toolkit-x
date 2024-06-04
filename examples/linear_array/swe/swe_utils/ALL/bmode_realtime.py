import arrus
import arrus.session
import arrus.utils.imaging
import arrus.utils.us4r
import arrus.kernels
import arrus.kernels.kernel
import queue
import pickle
import numpy as np
import cupy as cp
import scipy.io
from datetime import datetime
import argparse
import TXPB_Configuration.device
from SWE_utils_cupy_pipelined import *

from arrus.ops.us4r import (
    Scheme,
    Pulse,
    Tx,
    Rx,
    TxRx,
    TxRxSequence,
    DataBufferSpec
)

from arrus.ops.imaging import (
    PwiSequence
)

from arrus.utils.imaging import (
    Pipeline,
    SelectFrames,
    Squeeze,
    Lambda,
    RemapToLogicalOrder,
    Operation,
    Transpose
)
from arrus.utils.gui import (
    Display2D
)
from arrus.metadata import (
    FrameAcquisitionContext,
    EchoDataDescription
)

from arrus.utils.imaging import (
    BandpassFilter,
    QuadratureDemodulation,
    Decimation,
    ReconstructLri,
    Mean,
    EnvelopeDetection,
    Transpose,
    LogCompression,
    SelectFrames,
    Squeeze,
    Sum,
    get_extent
)

from arrus.devices.us4r import Us4RDTO
from arrus.devices.probe import ProbeDTO, ProbeModel, ProbeModelId


arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("test.log", arrus.logging.TRACE)

   
## PARAMETERS #############################################
tx_angles = [-12, -8, -4, -2, 0, 2, 4, 8, 12]
tx_freq   = 6.0e6
n_samples = 6400#5056
acq_depth = 82    # [mm] - only for T/R switch open to configure TX
pwi_pri   = 250
c         = 1540.0
sri       = 50e-3
n_periods = 3

tgc_start = 44
tgc_slope = 250

grid_step = 0.2

fs = 65e6

###########################################################
angles = np.array(tx_angles) * np.pi/180

    
# Create RX sequence configuration
rx_conf  = []
rx_conf += [(n_samples, pwi_pri), (n_samples, pwi_pri)]*len(tx_angles)
    
# Here starts communication with the device.
with arrus.Session("./us4r_bmode.prototxt") as sess:
    us4r = sess.get_device("/Us4R:0")
    us4r.disable_hv()
    n_elements = us4r.get_probe_model().n_elements
    print(f"Probe has {n_elements}")

    # Custom TX/RX seq configuration
    txrx_entries = []
        
    aperture = [False]*(192) + [True]*64
    delays = [0]*64

    for i, (n_samples, pri) in enumerate(rx_conf):
        # Tx_def has no effect - it is configured by other script
        tx_def = Tx(aperture=aperture,
                    excitation=Pulse(center_frequency=tx_freq, n_periods=n_periods, inverse=False),
                    delays=delays)
        rx_def = Rx(aperture=aperture,
                    sample_range=(0, n_samples),
                    downsampling_factor=1)

        txrx_entries.append(TxRx(tx_def, rx_def, pri*1e-6))
        
    tgc_curve = compute_linear_tgc(tgc_start, tgc_slope, sample_range=(0, n_samples), c=c)
    tgc_curve = np.clip(tgc_curve, 14, 54)

    seq = TxRxSequence(
        ops=txrx_entries,
        tgc_curve=tgc_curve, 
        sri=sri
    )

    x_grid = np.arange(-30, 30, grid_step) * 1e-3
    z_grid = np.arange(0, n_samples*c*0.5/fs, grid_step*1e-3)
        
    scheme = Scheme(
        # Run the provided sequence.
        tx_rx_sequence=seq,
        rx_buffer_size=2,
        output_buffer=DataBufferSpec(type="FIFO", n_elements=2),
        # Processing pipeline to perform on the GPU device.
        processing=Pipeline(
            steps=(
                TxpbAdapter(
                    n_push=0, n_pwi=len(tx_angles), n_samples=n_samples,
                    angles=angles, center_frequency=tx_freq,
                    n_periods=n_periods, speed_of_sound=c),
                Transpose(axes=(0, 1, 3, 2)),
                Lambda(lambda data: (np.save("rf.npy", data), data)[1]),
                BandpassFilter(),
                QuadratureDemodulation(),
                Decimation(decimation_factor=4, cic_order=2),
                # Data beamforming.
                ReconstructLri(x_grid=x_grid, z_grid=z_grid, rx_tang_limits = [-0.7, 0.7]),
                # Lambda(lambda data: (np.save("lri.npy", data), data)[1]),
                Squeeze(),
                Mean(axis=0),
                # Post-processing to B-mode image.
                EnvelopeDetection(),
                Transpose(),
                LogCompression(),
            ),
            placement="/GPU:0"
        ),
        work_mode="HOST"
    )
        
 
    # Configure TX
        
    txpb = TXPB_Configuration.device.TXPB()
    txpb_issue = 0
    
    if txpb.open():
        print('Setup TxB ...')
        txpb.setup()
        txpb.set_qspi_aux_cycles(0)
        txpb.set_qspi_clk_freq(1)
        print('Configuring the sequence')

        txpb.configure_CPWI_procedure(pwi_tx_freq = tx_freq/1e6, \
                                        pwi_tx_cycles = n_periods, \
                                        pwi_tx_aperture = 128, \
                                        pwi_tx_angles = tx_angles, \
                                        acq_depth = acq_depth, \
                                        c = c, \
                                        fs = fs/1e6)

        txpb.close()

    else:
        print('Could not open TXPB communication link ...')
        txpb_issue = 1
        
        
    # Upload the scheme on the us4r-lite device.
    if(txpb_issue == 0):
        buffer, metadata = sess.upload(scheme)
        # Start the scheme.
        cmd = input("\n\n PROGRAMMING IS DONE. Press ENTER to START FIRST TRIGGER, press q to exit")

        #display = Display2D(metadata=metadata, value_range=(-1000, 1000))
        # B-mode display.
        display = Display2D(metadata=metadata, value_range=(20, 90),
                            cmap="gray",
                            title="B-mode", xlabel="OX (mm)", ylabel="OZ (mm)",
                            extent=get_extent(x_grid, z_grid) * 1e3,
                            show_colorbar=True)
        sess.start_scheme()
        display.start(buffer)
        print("Close window to stop example")
     
    else:
        print("Stopping the example due to TXPB communication issue.")
        
print("Stopping the example.")



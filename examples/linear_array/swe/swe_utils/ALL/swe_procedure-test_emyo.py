import arrus
import arrus.session
import arrus.utils.imaging
import arrus.utils.us4r
import arrus.kernels
import arrus.kernels.kernel
import queue
import pickle
import argparse
import time
import numpy as np
import cupy as cp
import scipy as sp
import scipy.io
from scipy.signal import firwin, butter, buttord, freqz
from datetime import datetime

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


def main(idx):
    
    ############## PARAMETERS #############################################
    path          = './eMyoSound/'
    acq_type      = 'ssi'           # Acquisition type: {'ssi', 'cuse', 'cpwi'}

    ## ACQUISITION PARAMETERS #####
    # Push parameters
    pb_tx_freq    = [3.0] #[4.4, 4.4, 4.4]  # Push beam frequency [MHz]
    pb_tx_length  = [100] #[150, 150, 150]  # Push beam duration [us]
    pb_focus_z    = [600000]  #[20, 25, 30]     # Push beam focus [mm] - each beam is focused at the same depth. Specify very big number for plane wave
    pb_aperture   = [128]  #[32, 42, 51]     # Push beam aperture [# of elements of each beam] - must be even
    pb_cuse_beams = [1]   #[2, 2, 2]        # Number of comb beams. Always spaced to for symmetric comb. Relevant only for 'cuse' acq type.
    pb_cuse_sep   = [64]     # Comb beams separation [# of elements between beam centers].  Relevant only for 'cuse' acq type.
    pb_shift      = 0              # Push beam shift (# probe elements +/-, relative to center). Relevant only for 'ssi' acq type.

    if(acq_type == 'cpwi'):
        n_push = 0
    else:    
        n_push = len(pb_tx_length)

    # Tracking sequence parameters
    pwi_frames    = 45            # PWI frames to capture
    pwi_pri       = 150           # PWI PRI [us] - it is only for half RX aperture! 
    pwi_tx_angles = [0, 0, 0]       # Angles for CPWC
    pwi_tx_freq   = 3.0 * 1e6            # PWI pulses frequency [Hz]
    pwi_ncycles   = 4                # PWI pulses number of cycles
    n_samples     = 7040             # RX number of samples to capture
    acq_depth     = 92               # Acquisition depth [mm] - only for T/R switch open to configure TX

    # System parameters and constants
    fs            = 65e6             # RX sampling frequency [Hz]
    c             = 1540.0           # Medium speed of sound [m/s]

    tgc_start = 44                   # TGC starting gain [dB]
    tgc_slope = 250                  # TGC gain slope [dB/s]


    ###########################################################

    #angles = np.array(pwi_tx_angles) * np.pi/180
    angles = np.tile(pwi_tx_angles, int(np.ceil(pwi_frames/len(pwi_tx_angles))))*np.pi/180
    angles = angles[0:pwi_frames]

    # Create RX sequence configuration
    rx_conf = []
    for i in pb_tx_length:
        rx_conf +=  [(n_samples, i+40)] # PUSH TX/RX

    rx_conf += [(n_samples, pwi_pri), (n_samples, pwi_pri)]*pwi_frames

    # Here starts communication with the device.
    with arrus.Session("./us4r.prototxt") as sess:
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
                        excitation=Pulse(center_frequency=pwi_tx_freq, n_periods=pwi_ncycles, inverse=False),
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
            #sri=sri
        )

        scheme = Scheme(
            # Run the provided sequence.
            tx_rx_sequence=seq,
            rx_buffer_size=2,
            output_buffer=DataBufferSpec(type="FIFO", n_elements=2),
            # Processing pipeline to perform on the GPU device.
            processing=Pipeline(
                steps=(
                    TxpbAdapter(
                        n_push=n_push, n_pwi=pwi_frames, n_samples=n_samples,
                        angles=angles, center_frequency=pwi_tx_freq,
                        n_periods=pwi_ncycles, speed_of_sound=c),
                    Squeeze(),
                ),
                placement="/GPU:0"
            ),
            work_mode="MANUAL"
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

            if(acq_type == 'cpwi'):
                txpb.configure_CPWI_procedure(pwi_tx_freq = pwi_tx_freq/1e6, \
                                              pwi_tx_cycles = pwi_ncycles, \
                                              pwi_tx_aperture = 128, \
                                              pwi_tx_angles = pwi_tx_angles, \
                                              acq_depth = acq_depth, \
                                              c = c, \
                                              fs = fs/1e6)            
            
            elif(acq_type == 'cuse'):
                txpb.configure_SWE_CUSE_procedure(pb_tx_freq = pb_tx_freq,
                                                  pb_tx_length = pb_tx_length, 
                                                  pb_aperture = pb_aperture, 
                                                  pb_focus_z = pb_focus_z, 
                                                  pb_cuse_beams = pb_cuse_beams, 
                                                  pb_cuse_sep = pb_cuse_sep,
                                                  pwi_tx_freq = pwi_tx_freq/1e6, 
                                                  pwi_tx_cycles = pwi_ncycles, 
                                                  pwi_tx_angles = pwi_tx_angles, 
                                                  pwi_tx_aperture = 128,
                                                  pwi_frames = pwi_frames, 
                                                  acq_depth = acq_depth, 
                                                  c =c, 
                                                  fs = fs/1e6)

            elif(acq_type == 'ssi'):
                txpb.configure_SWE_SSI_procedure(pb_tx_freq = pb_tx_freq,
                                                 pb_tx_length = pb_tx_length,
                                                 pb_aperture = pb_aperture,  
                                                 pb_focus_z = pb_focus_z,  
                                                 pb_shift = pb_shift,
                                                 pwi_tx_freq = pwi_tx_freq/1e6,
                                                 pwi_tx_cycles = pwi_ncycles,
                                                 pwi_tx_angles = pwi_tx_angles, 
                                                 pwi_tx_aperture = 128, 
                                                 pwi_frames = pwi_frames, 
                                                 acq_depth = acq_depth, 
                                                 c = c, 
                                                 fs = fs/1e6)

            else:
                print('Unknown acqusition type. Aborting....')
                txpb_issue = 1

            txpb.close()

        else:
            print('Could not open TXPB communication link ...')
            txpb_issue = 1



        # Upload the scheme on the us4r-lite device.
        if(txpb_issue == 0):
            buffer, metadata = sess.upload(scheme)
            # Start the scheme.
            #cmd = input("\n\n PROGRAMMING IS DONE. Press ENTER to START ACQUISITION, press q to exit")
            #start = time.time()
            sess.run()

            # Get and save data
            cpu_data = buffer.get()
            #end = time.time()
            #print(end - start)

            rf = cpu_data[0] #np.array [N_samples*N_tx(liczba TX/RX - liczba triggerów ktore wypuszcza us4lite)*,32]
                
            # Save data
            scipy.io.savemat(path + 'rf_data_id_' + str(idx) + '.mat', dict(data=rf)) 

        else:
            print("Stopping the example due to TXPB communication issue.")

    print("Stopping the example.")


# Parser    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single TXPB parametric script.")
    parser.add_argument("--idx", dest="idx", type=int)
    args = parser.parse_args()
    args = main(args.idx)  

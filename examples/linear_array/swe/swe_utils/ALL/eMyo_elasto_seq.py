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

import TXPB_Configuration.device_eMyo
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

from arrus.utils.gui import (
    Display2D
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

def main(Focus, pbLen, idx):
    
    ############## PARAMETERS #############################################
    path          = './eMyoSound/'
    acq_type      = 'push'           # Acquisition type: {'elasto', 'pulse', 'push'}

    ## ACQUISITION PARAMETERS #####
    if(acq_type == 'elasto'):
        # Description:
            # Generates a push (or a series of push pulses) and then switches into tracking sequence.
            # Tracking sequence is a series of transmits by full aperture, and 64 elements receive.
            # EEach elasto sequence needs to be started by software.
        
        # Push parameters
        pb_tx_freq    = [3.0] #[4.4, 4.4, 4.4]   # Push beam frequency [MHz]
        pb_tx_length  = [pbLen] #[150, 150, 150]   # Push beam duration [us]
        pb_focus_pt   = np.array([[0, 0, Focus]])   # Push beam focus point ndarray [[x,y,z], [x,y,z], ...] in [mm]
        pb_aperture   = [128]  #[32, 42, 51]     # Push beam aperture [# of elements of each beam] - must be even

        # Tracking sequence parameters
        pwi_frames    = 45                     # PWI frames to capture (x2 if only 64 channels captured)
        pwi_pri       = 150                     # PWI PRI [us] - it is only for half RX aperture! 
        pwi_focus_pt  = np.array([[0, 0, Focus], [0, 0, Focus], [0, 0, Focus]])    # Focus points for tracking sequence (3 focus points to be sequenced)
        #pwi_focus_pt  = np.array([[0, 7, 60], [0, 7, 60], [0, 7, 60]])    # Focus points for tracking sequence (3 focus points to be sequenced)
        pwi_tx_freq   = 3.0 * 1e6               # PWI pulses frequency [Hz]
        pwi_ncycles   = 4                       # PWI pulses number of cycles
        n_samples     = 7040                    # RX number of samples to capture
        acq_depth     = 92                      # Acquisition depth [mm] - only for T/R switch open to configure TX
        
        # Do not modify below
        n_push = len(pb_tx_length)
        mode = "MANUAL"

    
    elif(acq_type == 'push'):
        # Description:
            # Generates a push (or a series of push pulses).
            # To generate another push, software trigger is required.
        
        # Push parameters
        push_period   = 5    # [s]
        pb_tx_freq    = [3.0] #[4.4, 4.4, 4.4]  # Push beam frequency [MHz]
        pb_tx_length  = [pbLen] #[150, 150, 150]   # Push beam duration [us]
        pb_focus_pt   = np.array([[0, 0, 5000000]])       # Push beam focus point ndarray [[x,y,z], [x,y,z], ...] in [mm]
        #pb_focus_pt   = np.array([50])
        pb_aperture   = [128]  #[32, 42, 51]      # Push beam aperture [# of elements of each beam] - must be even

        # Do not modify below
        n_push = len(pb_tx_length)
        n_samples     = 4224                    # RX number of samples to capture
        acq_depth     = 160                      # Acquisition depth [mm] - only for T/R switch open to configure TX
        pwi_frames    = 1
        pwi_pri       = 100
        pwi_tx_freq   = 3.0 * 1e6
        pwi_ncycles   = 1
        mode = "MANUAL"
      
    elif(acq_type == 'pulse'):
        # Description:
            # Generates a repetitive, single, short TX. PRI is controlled by hardware. (For watertank test).
            # It triggers endless (until stopped by the user).
            # Actually it is TX0 - pri - TX1 -- sri.... and so on. Make pri = sri for stable period.
            # It captures 0:63 for TX0, and 64:127 for TX1. 
            
        # Pulse parameters
        pwi_pri       = 20e3                     # PWI PRI [us] - it is only for half RX aperture! 
        pwi_focus_pt  = np.array([0, 0, 40000000])    # Focus point for tracking sequence (can be a list of points)
        #pwi_focus_pt   = Focus
        pwi_tx_aperture = 128                   # TX aperture (number of channels, counting from from 1)
        pwi_tx_freq   = 25 * 1e6               # PWI pulses frequency [Hz]
        pwi_ncycles   = 5                       # PWI pulses number of cycles
        sri           = 60e-3                   # Sequence repetition interval
        sri_period    = 5
        
        # Do not modify below
        n_samples     = 7040                    # RX number of samples to capture
        acq_depth     = 90                      # Acquisition depth [mm] - only for T/R switch open to configure TX 
        pwi_frames    = 1
        n_push        = 0
        mode          = "HOST"


    # System parameters and constants
    fs            = 65e6             # RX sampling frequency [Hz]
    c             = 1490.0           # Medium speed of sound [m/s]

    tgc_start = 44                   # TGC starting gain [dB]
    tgc_slope = 250                  # TGC gain slope [dB/m]

    ###########################################################

    #angles = np.array(pwi_tx_angles) * np.pi/180
    #angles = np.tile(pwi_tx_angles, int(np.ceil(pwi_frames/len(pwi_tx_angles))))*np.pi/180
    #angles = angles[0:pwi_frames]
    angles = np.zeros(pwi_frames)
    
    # Create RX sequence configuration
    rx_conf = []
    
    if (n_push != 0):
        for i in pb_tx_length:
            if(acq_type == 'push'):
                rx_conf +=  [(n_samples, i+200)] # PUSH TX/RX
            else:
                rx_conf +=  [(n_samples, i+40)] # PUSH TX/RX

    rx_conf += [(n_samples, pwi_pri), (n_samples, pwi_pri)]*pwi_frames

    # Here starts communication with the device.
    with arrus.Session("./us4r.prototxt") as sess:
        us4r = sess.get_device("/Us4R:0")
        us4r.disable_hv()
        n_elements = us4r.get_probe_model().n_elements
        print(f"Probe has {n_elements}")

        print(rx_conf)
        
        # Custom TX/RX seq configuration
        txrx_entries = []

        aperture = [False]*(192) + [True]*64
        delays = [0]*64

        for i, (n_samples, pri) in enumerate(rx_conf):
            # Tx_def has no effect - it is configured by other script
            tx_def = Tx(aperture=aperture,
                        excitation=Pulse(center_frequency=pwi_tx_freq, n_periods=30, inverse=False),
                        delays=delays)
            rx_def = Rx(aperture=aperture,
                        sample_range=(0, n_samples),
                        downsampling_factor=1)

            txrx_entries.append(TxRx(tx_def, rx_def, pri*1e-6))

        tgc_curve = compute_linear_tgc(tgc_start, tgc_slope, sample_range=(0, n_samples), c=c)
        tgc_curve = np.clip(tgc_curve, 14, 54)

        if(acq_type == 'pulse'):    
            seq = TxRxSequence(
                ops=txrx_entries,
                tgc_curve=tgc_curve, 
                sri=sri
            )
        else:
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
                    #Transpose(axes=(0, 1, 3, 2)),
                    Squeeze(),
                ),
                placement="/GPU:0"
            ),
            work_mode=mode
        )
 
        # Configure TX
        txpb = TXPB_Configuration.device_eMyo.TXPB()
        txpb_issue = 0

        if txpb.open():
            print('Setup TxB ...')
            txpb.setup()
            txpb.set_qspi_aux_cycles(0)
            txpb.set_qspi_clk_freq(1)
            print('Configuring the sequence')

            if(acq_type == 'pulse'):
                txpb.configure_pulse_procedure(pwi_tx_freq = pwi_tx_freq/1e6, 
                                               pwi_tx_cycles = pwi_ncycles, 
                                               pwi_tx_aperture = pwi_tx_aperture, 
                                               pwi_tx_focus_pt = pwi_focus_pt, 
                                               acq_depth = acq_depth, 
                                               c = c, 
                                               fs = fs/1e6)

            elif(acq_type == 'elasto'):
                txpb.configure_elasto_sequence(pb_tx_freq = pb_tx_freq,
                                               pb_tx_length = pb_tx_length,
                                               pb_aperture = pb_aperture, 
                                               pb_focus_pt = pb_focus_pt, 
                                               pwi_tx_freq = pwi_tx_freq/1e6,
                                               pwi_tx_cycles = pwi_ncycles,
                                               pwi_tx_focus_pt = pwi_focus_pt, 
                                               pwi_tx_aperture = 128,
                                               pwi_frames = pwi_frames,
                                               acq_depth = acq_depth,
                                               c = c,
                                               fs= fs/1e6)
            
            elif(acq_type == 'push'):
                txpb.configure_push_sequence(pb_tx_freq = pb_tx_freq, 
                                             pb_tx_length = pb_tx_length, 
                                             pb_aperture = pb_aperture, 
                                             pb_focus_pt = pb_focus_pt, 
                                             acq_depth = acq_depth, 
                                             c = c, 
                                             fs = fs)
                
            else:
                print('Unknown acqusition type. Aborting....')
                txpb_issue = 1

            #txpb.close()

        else:
            print('Could not open TXPB communication link ...')
            txpb_issue = 1

            
        # Upload the scheme on the us4r-lite device.
        if(txpb_issue == 0):
            buffer, metadata = sess.upload(scheme)
            
            # 'pulse' sequence
            if(mode == "HOST"):
                display = Display2D(metadata=metadata, value_range=(-1000, 1000))
                sess.start_scheme()
                display.start(buffer)
            
            else:   
                # 'push' sequence
                if(acq_type == 'push'):
                    print("Press CTRL+C to stop the script.")
                    while(1):
                        
                        # Start the scheme.
                        sess.run()
                        # Getdata
                        cpu_data = buffer.get()
                        
                        # Reload TXPB
                        txpb.reset_scan_table_index()
                        time.sleep(0.05)
                        txpb.preload_req()
                        
                        time.sleep(push_period)
                        
                elif(acq_type == 'pulse'):
                    print("Press CTRL+C to stop the script.")
                    while(1):
                        
                        # Start the scheme.
                        sess.run()
                        # Getdata
                        cpu_data = buffer.get()
                        
                        # Reload TXPB
                        txpb.reset_scan_table_index()
                        time.sleep(0.05)
                        txpb.preload_req()
                        
                        time.sleep(sri_period)                        
                
                # 'elasto' sequence
                else: 
                    # Start the scheme.
                    sess.run()
                    # Get and save data
                    cpu_data = buffer.get()
                    rf = cpu_data[0] 
    
                    # Save data
                    scipy.io.savemat(path + 'rf_data_id_' + str(idx) + '.mat', dict(data=rf)) 

        else:
            print("Stopping the example due to TXPB communication issue.")

            
    txpb.set_hw_trigger_enable(enable=0) 
    time.sleep(0.1)
    txpb.close()        
    print("Script finished.")


# Parser    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single TXPB eMyoSound parametric script.")
    parser.add_argument("--Focus", dest="Focus", type=float)
    parser.add_argument("--pbLen", dest="pbLen", type=float)
    parser.add_argument("--idx", dest="idx", type=int)
    args = parser.parse_args()
    args = main( args.Focus, args.pbLen, args.idx)  

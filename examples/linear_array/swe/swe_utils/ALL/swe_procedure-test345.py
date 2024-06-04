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


def main(pbFreq, pbFoci, pbLen, pbAp, idx):
    
    ############## PARAMETERS #############################################
    path          = './Parametric_tests/Test_6/'
    acq_type      = 'ssi'           # Acquisition type: {'ssi', 'cuse', 'cpwi'}

    ## ACQUISITION PARAMETERS #####
    # Push parameters
    pb_tx_freq    = [pbFreq] #[4.4, 4.4, 4.4]  # Push beam frequency [MHz]
    pb_tx_length  = [pbLen] #[150, 150, 150]  # Push beam duration [us]
    pb_focus_z    = [pbFoci]  #[20, 25, 30]     # Push beam focus [mm] - each beam is focused at the same depth. Specify very big number for plane wave
    pb_aperture   = [pbAp]  #[32, 42, 51]     # Push beam aperture [# of elements of each beam] - must be even
    pb_cuse_beams = [1]   #[2, 2, 2]        # Number of comb beams. Always spaced to for symmetric comb. Relevant only for 'cuse' acq type.
    pb_cuse_sep   = [64, 64, 64]     # Comb beams separation [# of elements between beam centers].  Relevant only for 'cuse' acq type.
    pb_shift      = 0              # Push beam shift (# probe elements +/-, relative to center). Relevant only for 'ssi' acq type.

    if(acq_type == 'cpwi'):
        n_push = 0
    else:    
        n_push = len(pb_tx_length)

    # Tracking sequence parameters
    pwi_frames    = 150            # PWI frames to capture
    pwi_pri       = 100           # PWI PRI [us] - it is only for half RX aperture! 
    pwi_tx_angles = [-4, 0, 4]       # Angles for CPWC
    pwi_tx_freq   = 4.4 * 1e6            # PWI pulses frequency [Hz]
    pwi_ncycles   = 2                # PWI pulses number of cycles
    n_samples     = 4224             # RX number of samples to capture
    acq_depth     = 52               # Acquisition depth [mm] - only for T/R switch open to configure TX

    # System parameters and constants
    fs            = 65e6             # RX sampling frequency [Hz]
    c             = 1540.0           # Medium speed of sound [m/s]

    tgc_start = 44                   # TGC starting gain [dB]
    tgc_slope = 250                  # TGC gain slope [dB/s]

    ## RECONSTRUCTION PARAMETERS #####
    # RF Filtering
    rf_filter_band        = [4.0e6, 7e6]  # Desired pass band, [Hz]
    rf_filter_trans_width = 1e6           # Width of transition from pass band to stop band, [Hz]
    rf_filter_numtaps     = 236           # Size of the FIR filter (# of taps)

    # Post down conversion IQ filtering
    demod_filter_cutoff      = 0.5 * pwi_tx_freq  # Desired cutoff frequency, [Hz]
    demod_filter_trans_width = 0.5 * pwi_tx_freq  # Width of transition from pass band to stop band, [Hz]
    demod_filter_numtaps     = 128          # Size of the FIR filter (# of taps)

    # Beamforming
    grid_step = 0.2                          # Reconstruction grid step [mm]
    x_grid    = np.arange(-20, 20, grid_step) * 1e-3
    z_grid    = np.arange(0, n_samples*c*0.5/fs, grid_step*1e-3)
    rx_tang_limits = [-0.7, 0.7]

    # Shear wave detection
    swd_mode              = 'kasai'        # 'kasai' or 'loupas'
    swd_zGate_length      = 4              # Averaging z axis kernel size
    swd_ensemble_length   = 4              # Averaging slow-time axis kernel size

    # Shear wave motion data filtering
    df_sws_range = [0.5, 4.0]              # SWS filtering pass band [m/s]
    df_f_range   = [40.0, 800.0]           # Shear wave motion frequency filtering passband [Hz]
    df_k_range   = 0.9                     # Shear wave motion spatial high corner frequency.

    # SWS estimation
    swse_interp_factor = 5                      # Interpolation factor
    swse_interp_order  = 2                      # (Spline) interpolation order. Must be in 0-5 range.
    swse_d             = 16                     # SWS estimation kernel size (in pixels)
    swse_frames        = [0, 99]                # Range of frames to be used for SWS estimation
    swse_SWV_range     = [0.5, 4.0]             # Range of SWS of interest. All higher or lower will be clipped.
    swse_x_range       = [[50, 300], [0, 200]]  # Range within x-dimension to perform SWS estimation (LR and RL, repsectively)
    swse_z_clip        = [5, 10]                # How many pixels to clip from top and bottom (SWS not estimated, and set to 0).

    # Post-processing
    median_filter_size = 5    # Median filter kernel size. Kernel is a square.


    ###########################################################

    ## Design a band-pass FIR filter for RF filtering of raw channel data
    band = rf_filter_band                # Desired pass band, Hz
    trans_width = rf_filter_trans_width  # Width of transition from pass band to stop band, Hz
    numtaps = rf_filter_numtaps          # Size of the FIR filter.

    edges = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5*fs]
    rf_fir_taps = signal.remez(numtaps, edges, [0, 1, 0], Hz=fs)

    ## Design a low-pass FIR filter for filtering of down-conversion products  
    cutoff = demod_filter_cutoff            # Desired cutoff frequency, Hz
    trans_width = demod_filter_trans_width  # Width of transition from pass band to stop band, Hz
    numtaps = demod_filter_numtaps          # Size of the FIR filter.
    iq_fir_taps = signal.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*fs], [1, 0], Hz=fs)

    #angles = np.array(pwi_tx_angles) * np.pi/180
    angles = np.tile(pwi_tx_angles, int(np.ceil(pwi_frames/len(pwi_tx_angles))))*np.pi/180
    angles = angles[0:pwi_frames]

    # Create RX sequence configuration
    rx_conf = []
    for i in pb_tx_length:
        rx_conf +=  [(n_samples, i+20)] # PUSH TX/RX

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
                    Transpose(axes=(0, 1, 3, 2)),
                    FirFilter(taps=rf_fir_taps, num_pkg=None, filter_pkg=None),
                    QuadratureDemodulation(),
                    Decimation(filter_type="fir", filter_coeffs=iq_fir_taps, decimation_factor=1),
                    ReconstructLri(x_grid=x_grid, z_grid=z_grid, rx_tang_limits=rx_tang_limits),
                    Squeeze(),
                    AngleCompounding(nAngles=len(pwi_tx_angles)),
                    Pipeline(
                        steps=(
                            ShearwaveDetection(mode=swd_mode, packet_size=swd_ensemble_length, z_gate=swd_zGate_length, fc=pwi_tx_freq),
                            Output(),
                            ShearwaveMotionDataFiltering(sws_range=df_sws_range, f_range=df_f_range, k_range=df_k_range),
                            Output(),
                            SWS_Estimation(x_range=swse_x_range, z_clip = swse_z_clip, frames_range = swse_frames, d=swse_d, fri=pwi_pri*2*1e-6,
                                           interp_factor=swse_interp_factor, interp_order=swse_interp_order, px_pitch=grid_step*1e-3, sws_range=swse_SWV_range),
                            #SWS_Compounding(),
                            #MedianFiltering(kernel_size=median_filter_size)
                        ),
                        placement="/GPU:0"
                    ),
                    Lambda(lambda data: data[5], lambda metadata: metadata.copy(input_shape=metadata.input_shape[1:])), 
                ),
                placement="/GPU:0"
            ),
            work_mode="MANUAL"
        )

        # Prepare GPU memory control utilities
        fft_plan_cache = cp.fft.config.get_plan_cache()
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()    

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

            Bmode = cpu_data[0] #np.array [N_samples*N_tx(liczba TX/RX - liczba triggerów ktore wypuszcza us4lite)*,32]
            sws   = cpu_data[1]
            swdf  = cpu_data[2]
            swd   = cpu_data[3]
            
            # Processing #
            # Get energy map from swd data
            ddataE = np.squeeze(swd[10:-20, ...])
            m = np.mean(ddataE, axis=2)
            ddataE = np.transpose(ddataE, [2,0,1])
            ddataE = np.subtract(ddataE, m)
            ddataE = np.transpose(ddataE, [1,2,0])
            E1 = np.power(ddataE, 2)
            E1 = np.squeeze(np.sum(E1, 2))
            
            # Get energy map from swdf data
            ddataE = np.squeeze(swdf[0, 10:-20, ...])
            m = np.mean(ddataE, axis=2)
            ddataE = np.transpose(ddataE, [2,0,1])
            ddataE = np.subtract(ddataE, m)
            ddataE = np.transpose(ddataE, [1,2,0])
            E2 = np.power(ddataE, 2)
            E2 = np.squeeze(np.sum(E2, 2))            
                
            # Save data
            scipy.io.savemat(path + 'sws'   + '_pbfreq' + str(pbFreq) + '_foc' + str(pbFoci) + '_pblen' + str(pbLen) + '_pbAp' + str(pbAp) + '_id_' + str(idx) +'.mat', dict(data=sws)) 
            scipy.io.savemat(path + 'energy_swd'   + '_pbfreq' + str(pbFreq) + '_foc' + str(pbFoci) + '_pblen' + str(pbLen) + '_pbAp' + str(pbAp) + '_id_' + str(idx) +'.mat', dict(data=E1)) 
            scipy.io.savemat(path + 'energy_swdf'  + '_pbfreq' + str(pbFreq) + '_foc' + str(pbFoci) + '_pblen' + str(pbLen) + '_pbAp' + str(pbAp) + '_id_' + str(idx) +'.mat', dict(data=E2)) 

            scipy.io.savemat(path + 'swd'    + '_pbfreq' + str(pbFreq) + '_foc' + str(pbFoci) + '_pblen' + str(pbLen) + '_pbAp' + str(pbAp) + '_id_' + str(idx) +'.mat', dict(data=swd)) 
            scipy.io.savemat(path + 'swdf'   + '_pbfreq' + str(pbFreq) + '_foc' + str(pbFoci) + '_pblen' + str(pbLen) + '_pbAp' + str(pbAp) + '_id_' + str(idx) +'.mat', dict(data=swdf)) 
            scipy.io.savemat(path + 'bmode'  + '_pbfreq' + str(pbFreq) + '_foc' + str(pbFoci) + '_pblen' + str(pbLen) + '_pbAp' + str(pbAp) + '_id_' + str(idx) +'.mat', dict(data=Bmode)) 

            # Clear GPU memory
            fft_plan_cache.clear()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()

        else:
            print("Stopping the example due to TXPB communication issue.")

    print("Stopping the example.")


# Parser    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single TXPB parametric script.")
    parser.add_argument("--pbFreq", dest="pbFreq", type=float)
    parser.add_argument("--pbFoci", dest="pbFoci", type=float)
    parser.add_argument("--pbLen", dest="pbLen", type=float)
    parser.add_argument("--pbAp", dest="pbAp", type=int)
    parser.add_argument("--idx", dest="idx", type=int)
    args = parser.parse_args()
    args = main(args.pbFreq, args.pbFoci, args.pbLen, args.pbAp, args.idx)  

import scipy.io
import numpy as np
import scipy as sp
import cupy as cp
import matplotlib.pyplot as plt
import os
import time
import argparse
from SWE_utils_cupy_standalone import *
from scipy.signal import firwin, butter, buttord, freqz

from arrus.ops.us4r import *
from arrus.ops.imaging import *
from arrus.metadata import *
from arrus.devices.probe import *
from arrus.devices.us4r import Us4RDTO
from arrus.utils.imaging import *

def main(iOrder, iFactor, gridStep, shift, idx):

    
    ### General settings ################
    dataset_id  = idx
    #directory   = '/media/damian/8A4F-A24E/Test11/'
    directory = "./Parametric_tests/Test_13/"
    directory2  = "./Parametric_tests/Test_13/rec"

    # Constants
    c              = 1540.0
    probe_elements = 128
    probe_pitch    = 0.298e-3
    fs             = 65e6
    t0             = 43

    # Sequence parameters
    pwi_txFreq   = 4.4e6
    pwi_nCycles  = 3
    pwi_txAngles = [-4.0, 0.0, 4.0]
    pwi_txPri    = 80e-6
    pwi_fri      = 2* pwi_txPri

    # RF Filter
    rf_filter_band        = [4e6, 7e6]    # Desired pass band, Hz
    rf_filter_trans_width = 1e6           # Width of transition from pass band to stop band, Hz
    rf_filter_numtaps     = 256 #236           # Size of the FIR filter.

    # Post down conversion IQ filtering
    demod_filter_cutoff = 0.5 * 4.4e6       # Desired cutoff frequency, Hz
    demod_filter_trans_width = 0.5 * 4.4e6  # Width of transition from pass band to stop band, Hz
    demod_filter_numtaps = 64

    # Beamforming
    px_size = gridStep  # in [mm]
    x_grid = np.arange(-20, 20, px_size) * 1e-3
    z_grid = np.arange(0, 50, px_size)   * 1e-3
    rx_tang_limits = [-0.7, 0.7]

    # Shear wave detection
    swd_mode              = 'kasai'
    swd_zGate_length      = int(4.0*0.2/gridStep)
    swd_ensemble_length   = 4

    # Input parameters
    df_sws_range = [0.5, 4.5];
    df_f_range   = [40.0, 700.0];
    df_k_range   = 0.9;

    # SWS estimation
    swse_interp_factor = iFactor
    swse_interp_order  = iOrder
    swse_d             = int(20*0.2/gridStep)
    swse_frames        = [0, 90];
    swse_SWV_range     = [0.5, 4.5];
    swse_x_range       = [[0, 420], [0, 420]]
    #swse_x_range       = [[90, 300], [0, 110]]
    swse_z_clip        = [5, 10]

    # Post-processing
    median_filter_size = int(5*0.2/gridStep)

    ### LOAD the dataset and crop data ###################
    # Load a dataset
    data = sp.io.loadmat(directory + 'rf_id_' + str(dataset_id) + '_shift_' + str(shift) + '.mat')
    rf = data["data"]
    print("Input RF data shape:")
    print(rf.shape)
    rf = np.transpose(rf, [0,1,3,2])
    Nframes = rf.shape[1]

    # Crop the data (time zero, # of frames)
    rf = rf[:, :, t0:, ...]
    Nframes = rf.shape[1]

    ### Processing ###################
    ## Design a band-pass FIR filter for RF filtering of raw channel data
    band = rf_filter_band                # Desired pass band, Hz
    trans_width = rf_filter_trans_width  # Width of transition from pass band to stop band, Hz
    numtaps = rf_filter_numtaps          # Size of the FIR filter.

    edges = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5*fs]
    rf_fir_taps = signal.remez(numtaps, edges, [0, 1, 0], Hz=fs)

    ## Design a low-pass FIR filter for filtering of down-conversion products
    # Specify the filter parameters    
    cutoff = demod_filter_cutoff            # Desired cutoff frequency, Hz
    trans_width = demod_filter_trans_width  # Width of transition from pass band to stop band, Hz
    numtaps = demod_filter_numtaps          # Size of the FIR filter.
    iq_fir_taps = signal.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*fs], [1, 0], Hz=fs)
    
    ## Perform FIR filtering
    rf = sp.signal.lfilter(rf_fir_taps, 1, rf, axis=2)

    ## Create meatadata
    tx_angles = np.tile(pwi_txAngles, int(np.ceil(rf.shape[1]/len(pwi_txAngles))))*np.pi/180

    sequence = PwiSequence(
        pulse=Pulse(center_frequency=pwi_txFreq, n_periods=pwi_nCycles, inverse=False),
        rx_sample_range=(0, rf.shape[1]),
        speed_of_sound=c, # [m/s],
        angles=tx_angles,
        pri=pwi_txPri
    )

    model=ProbeModel(
        model_id=ProbeModelId("atl", "l7-4"),
        n_elements=probe_elements,
        pitch=probe_pitch,
        curvature_radius=0,
    )

    probe = ProbeDTO(
        model=model
    )

    device=Us4RDTO(
        sampling_frequency=fs,
        probe=probe
    )

    context = arrus.metadata.FrameAcquisitionContext(
        device=device, 
        sequence=sequence,
        raw_sequence=None,
        medium=None,
        custom_data={})

    data_desc=EchoDataDescription(
        sampling_frequency=fs,
    )

    metadata = ConstMetadata(context=context, data_desc=data_desc, input_shape=rf.shape, is_iq_data=False, dtype=np.int16, version=None)

    ## Define the processing pipeline
    processing = Pipeline(
        steps=(
            Transpose(axes=(0, 1, 3, 2)),
            #FirFilter(taps=rf_fir_taps, num_pkg=None, filter_pkg=None),
            QuadratureDemodulation(),
            Decimation(filter_type="fir", filter_coeffs=iq_fir_taps, decimation_factor=1),
            ReconstructLri(x_grid=x_grid, z_grid=z_grid, rx_tang_limits=rx_tang_limits),
            Squeeze(),
        ),
        placement="/GPU:0"
    )

    # Prepare pipeline
    processing.prepare(metadata)

    # Transfer data to GPU
    din = cp.asarray(rf)

    ## Run the pipeline
    output = processing.process(din)
    lri_data_gpu = output[0]

    ## Compounding ################
    AngleCompounder = AngleCompounding(nAngles=len(pwi_txAngles))
    AngleCompounder.prepare()
    hri_data_gpu = AngleCompounder.process(data=lri_data_gpu)

    ## Display B-mode ################
    data = cp.squeeze(hri_data_gpu[5, ...])
    # Envelope detection
    data = cp.abs(data)
    # Log compression
    data[data==0] = 10**-10
    data = 20 * cp.log10(data)
    data_dim = data.shape
    if(data_dim[1] > data_dim[0] ):
        data = np.transpose(data, [1,0])

    # Save B-mode
    data_cpu = data.get()
    scipy.io.savemat(directory2 + 'bmode'   + '_grid' + str(gridStep) + '_iFactor' + str(iFactor) + '_iOrder' + str(iOrder) + '_id_' + str(idx) + '_shift_' + str(shift) + '.mat', dict(data=data_cpu)) 


    ## Shear wave detection ################
    ShearDetector = ShearwaveDetection(mode=swd_mode, packet_size=swd_ensemble_length, z_gate=swd_zGate_length, frame_pri=pwi_fri, c=c, fc=pwi_txFreq, fs=fs)
    ShearDetector.prepare(c=c, frame_pri=200e-6, fs=65e6)
    ddata = ShearDetector.process(data=hri_data_gpu)
    
    ## Find SW SNR ################
    # Get noise energy map from swd data
    ddataE_N = ddata[:, :, 100:114]
    m = cp.mean(ddataE_N, axis=2)
    ddataE_N = cp.transpose(ddataE_N, [2,0,1])
    ddataE_N = cp.subtract(ddataE_N, m)
    ddataE_N = cp.transpose(ddataE_N, [1,2,0])
    E_N = cp.power(ddataE_N, 2)
    E_N = cp.squeeze(np.sum(E_N, 2))
    E_N = E_N / ddataE_N.shape[2]

    # Get signal energy map from swd data
    ddataE_S = ddata[:, :, 1:80]
    m = np.mean(ddataE_S, axis=2)
    ddataE_S = np.transpose(ddataE_S, [2,0,1])
    ddataE_S = np.subtract(ddataE_S, m)
    ddataE_S = np.transpose(ddataE_S, [1,2,0])
    E_S = np.power(ddataE_S, 2)
    E_S = np.squeeze(np.sum(E_S, 2))
    E_S = E_S / ddataE_S.shape[2]

    SNR = E_S/E_N
    SNR[SNR==0] = 10**-10
    SNRi = 20 * cp.log10(SNR)

    # Save
    data_cpu = SNRi.get()
    scipy.io.savemat(directory2 + 'swSNR'   + '_grid' + str(gridStep) + '_iFactor' + str(iFactor) + '_iOrder' + str(iOrder) + '_id_' + str(idx) + '_shift_' + str(shift) +'.mat', dict(data=data_cpu)) 

    ## Dir filtering ################
    # Shear wave motion data filtering in Fourier domain
    DirFilter = ShearwaveMotionDataFiltering(sws_range=df_sws_range, f_range=df_f_range, k_range=df_k_range)
    DirFilter.prepare(input_shape = ddata.shape, fs=1.0/pwi_fri)
    ddata_f = DirFilter.process(data=ddata)

    ## SWS estimation ################
    dim = ddata_f.shape
    SWS_Estimator = SWS_Estimation(x_range=swse_x_range, z_clip = swse_z_clip, frames_range = swse_frames,
                                   d=swse_d, fri = pwi_fri, interp_factor=swse_interp_factor, interp_order=swse_interp_order, 
                                    px_pitch=px_size*1e-3, sws_range=swse_SWV_range)
    SWS_Estimator.prepare(input_shape = ddata_f.shape)
    SWV = SWS_Estimator.process(data=ddata_f)
    data_cpu = SWV.get()
    scipy.io.savemat(directory2 + 'sws' + '_grid' + str(gridStep) + '_iFactor' + str(iFactor) + '_iOrder' + str(iOrder) + '_id_' + str(idx) + '_shift_' + str(shift) + '.mat', dict(data=data_cpu))     

    
# Parser    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single TXPB parametric script.")
    parser.add_argument("--iOrder", dest="iOrder", type=int)
    parser.add_argument("--iFactor", dest="iFactor", type=int)
    parser.add_argument("--shift", dest="shift", type=int)
    parser.add_argument("--gridStep", dest="gridStep", type=float)
    parser.add_argument("--idx", dest="idx", type=int)
    args = parser.parse_args()
    args = main(args.iOrder, args.iFactor, args.gridStep, args.shift, args.idx)      
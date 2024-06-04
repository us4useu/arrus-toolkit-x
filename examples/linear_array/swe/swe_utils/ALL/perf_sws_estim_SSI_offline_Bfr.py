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

def main(iOrder, iFactor, gridStep, idx, p):

    
    ### General settings ################
    dataset_id  = idx
    #directory   = '/media/damian/8A4F-A24E/Test11/'
    directory = "./Parametric_tests/Test_13/"
    directory2  = "./Parametric_tests/Test_13/rec/"

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
    swd_zGate_length      = 4 #int(4.0*0.2/gridStep)
    swd_ensemble_length   = 4

    # Input parameters
    df_sws_range = [0.5, 4.5];
    df_f_range   = [40.0, 700.0];
    df_k_range   = 0.9;

    # SWS estimation
    swse_interp_factor = iFactor
    swse_interp_order  = iOrder
    swse_d             = int(p*0.2/gridStep)  # default p=20
    swse_frames        = [0, 90];
    swse_SWV_range     = [0.5, 4.5];
    swse_x_range       = [[0, 420], [0, 420]]
    #swse_x_range       = [[90, 300], [0, 110]]
    swse_z_clip        = [int(5*0.2/gridStep), int(10*0.2/gridStep)]
    
    # Compounding
    # Regions to mask to 0
    a    = 0.7
    A_LR = [0, int(99*0.2/gridStep)]
    A_RL = [0, int(200*0.2/gridStep)]

    B_LR = [0, int(150*0.2/gridStep)]
    B_RL = [int(50*0.2/gridStep), int(200*0.2/gridStep)]

    C_LR = [0, int(200*0.2/gridStep)]
    C_RL = [int(101*0.2/gridStep), int(200*0.2/gridStep)]    

    
    #A_LR = [0, 99]
    #A_RL = [0, 200]

    #B_LR = [0, 150]
    #B_RL = [50, 200]

    #C_LR = [0, 200]
    #C_RL = [101, 200]  
    
    # Post-processing
    median_filter_size = int(5*0.2/gridStep)

    ### LOAD the dataset and crop data ###################
    
    # Load a dataset
    data1 = sp.io.loadmat(directory + 'rf_id_' + str(dataset_id) + '_shift_-30.mat')
    rf_0 = data1["data"]
    print("Input data shape:")
    print(rf_0.shape)
    
    data2 = sp.io.loadmat(directory + 'rf_id_' + str(dataset_id) + '_shift_0.mat')
    rf_1 = data2["data"]
    
    data3 = sp.io.loadmat(directory + 'rf_id_' + str(dataset_id) + '_shift_30.mat')
    rf_2 = data3["data"]   
    
    #rf_0 = cp.asarray(rf_0)
    #rf_1 = cp.asarray(rf_1)
    #rf_2 = cp.asarray(rf_2)  
      
    
    # Crop the data (time zero, # of frames)
    rf_0 = rf_0[:, :, t0:, ...]
    rf_1 = rf_1[:, :, t0:, ...]
    rf_2 = rf_2[:, :, t0:, ...]
    Nframes = rf_0.shape[1]

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
    #rf_0 = sp.signal.lfilter(rf_fir_taps, 1, rf_0, axis=2)
    #rf_1 = sp.signal.lfilter(rf_fir_taps, 1, rf_1, axis=2)
    #rf_2 = sp.signal.lfilter(rf_fir_taps, 1, rf_2, axis=2)
    
    print("Data shape after RF filtering:")
    print(rf_0.shape)
    
    rf_0 = cp.asarray(rf_0)
    rf_1 = cp.asarray(rf_1)
    rf_2 = cp.asarray(rf_2)        

    ## Create meatadata
    tx_angles = np.tile(pwi_txAngles, int(np.ceil(rf_0.shape[1]/len(pwi_txAngles))))*np.pi/180

    sequence = PwiSequence(
        pulse=Pulse(center_frequency=pwi_txFreq, n_periods=pwi_nCycles, inverse=False),
        rx_sample_range=(0, rf_0.shape[1]),
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

    metadata = ConstMetadata(context=context, data_desc=data_desc, input_shape=rf_0.shape, is_iq_data=False, dtype=np.int16, version=None)

    ## Define the processing pipeline
    processing_A = Pipeline(
        steps=(
            Transpose(axes=(0, 1, 3, 2)),
            FirFilter(taps=rf_fir_taps, num_pkg=None, filter_pkg=None),
            QuadratureDemodulation(),
            Decimation(filter_type="fir", filter_coeffs=iq_fir_taps, decimation_factor=1),
            ReconstructLri(x_grid=x_grid, z_grid=z_grid, rx_tang_limits=rx_tang_limits),
            Squeeze(),
        ),
        placement="/GPU:0"
    )
    
    processing_B = Pipeline(
        steps=(
            Transpose(axes=(0, 1, 3, 2)),
            FirFilter(taps=rf_fir_taps, num_pkg=None, filter_pkg=None),
            QuadratureDemodulation(),
            Decimation(filter_type="fir", filter_coeffs=iq_fir_taps, decimation_factor=1),
            ReconstructLri(x_grid=x_grid, z_grid=z_grid, rx_tang_limits=rx_tang_limits),
            Squeeze(),
        ),
        placement="/GPU:0"
    )
    
    processing_C = Pipeline(
        steps=(
            Transpose(axes=(0, 1, 3, 2)),
            FirFilter(taps=rf_fir_taps, num_pkg=None, filter_pkg=None),
            QuadratureDemodulation(),
            Decimation(filter_type="fir", filter_coeffs=iq_fir_taps, decimation_factor=1),
            ReconstructLri(x_grid=x_grid, z_grid=z_grid, rx_tang_limits=rx_tang_limits),
            Squeeze(),
        ),
        placement="/GPU:0"
    )    

    # Prepare pipeline
    processing_A.prepare(metadata)
    processing_B.prepare(metadata)
    processing_C.prepare(metadata)

    # Transfer data to GPU and run the pipeline
    din = cp.asarray(rf_0)
    output_A = processing_A.process(din)
    lri_data_gpu_0 = output_A[0]
    
    din = cp.asarray(rf_1)
    output_B = processing_B.process(din)
    lri_data_gpu_1 = output_B[0]
    
    din = cp.asarray(rf_2)
    output_C = processing_C.process(din)
    lri_data_gpu_2 = output_C[0]    

    ## Compounding ################
    AngleCompounder = AngleCompounding(nAngles=len(pwi_txAngles))
    AngleCompounder.prepare()
    hri_data_gpu_0 = AngleCompounder.process(data=lri_data_gpu_0)
    hri_data_gpu_1 = AngleCompounder.process(data=lri_data_gpu_1)
    hri_data_gpu_2 = AngleCompounder.process(data=lri_data_gpu_2)
    
    print(hri_data_gpu_0.shape)

    ## Display B-mode ################
    data = cp.squeeze(hri_data_gpu_0[5, ...])
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
    scipy.io.savemat(directory2 + 'bmode'   + '_grid' + str(gridStep) + '_iFactor' + str(iFactor) + '_iOrder' + str(iOrder) + '_id_' + str(idx+40) + '.mat', dict(data=data_cpu)) 


    ## Shear wave detection ################
    ShearDetector = ShearwaveDetection(mode=swd_mode, packet_size=swd_ensemble_length, z_gate=swd_zGate_length, frame_pri=pwi_fri, c=c, fc=pwi_txFreq, fs=fs)
    ShearDetector.prepare(c=c, frame_pri=200e-6, fs=65e6)
    ddata_0 = ShearDetector.process(data=hri_data_gpu_0)
    ddata_1 = ShearDetector.process(data=hri_data_gpu_1)
    ddata_2 = ShearDetector.process(data=hri_data_gpu_2)
    
    # Export swdf for debug
    if(0):
        swd_A = ddata_0.get()
        swd_B = ddata_1.get()
        swd_C = ddata_2.get()

        scipy.io.savemat(directory2 + 'swd_A' + '_grid' + str(gridStep) + '_iFactor' + str(iFactor) + '_iOrder' + str(iOrder) + '_id_' + str(idx+40) + '.mat', dict(data=swd_A)) 
        scipy.io.savemat(directory2 + 'swd_B' + '_grid' + str(gridStep) + '_iFactor' + str(iFactor) + '_iOrder' + str(iOrder) + '_id_' + str(idx+40) + '.mat', dict(data=swd_B)) 
        scipy.io.savemat(directory2 + 'swd_C' + '_grid' + str(gridStep) + '_iFactor' + str(iFactor) + '_iOrder' + str(iOrder) + '_id_' + str(idx+40) + '.mat', dict(data=swd_C))    

    ## Dir filtering ################
    # Shear wave motion data filtering in Fourier domain
    DirFilter = ShearwaveMotionDataFiltering(sws_range=df_sws_range, f_range=df_f_range, k_range=df_k_range)
    DirFilter.prepare(input_shape = ddata_0.shape, fs=1.0/pwi_fri)
    ddata_f_0 = DirFilter.process(data=ddata_0)
    ddata_f_1 = DirFilter.process(data=ddata_1)
    ddata_f_2 = DirFilter.process(data=ddata_2)
    
    
    # Export swdf for debug
    if(0):
        swdf_A = ddata_f_0.get()
        swdf_B = ddata_f_1.get()
        swdf_C = ddata_f_2.get()

        scipy.io.savemat(directory2 + 'swdf_A' + '_grid' + str(gridStep) + '_iFactor' + str(iFactor) + '_iOrder' + str(iOrder) + '_id_' + str(idx+40) + '_p' + str(p) +'.mat', dict(data=swdf_A)) 
        scipy.io.savemat(directory2 + 'swdf_B' + '_grid' + str(gridStep) + '_iFactor' + str(iFactor) + '_iOrder' + str(iOrder) + '_id_' + str(idx+40) + '_p' + str(p) +'.mat', dict(data=swdf_B)) 
        scipy.io.savemat(directory2 + 'swdf_C' + '_grid' + str(gridStep) + '_iFactor' + str(iFactor) + '_iOrder' + str(iOrder) + '_id_' + str(idx+40) + '_p' + str(p) +'.mat', dict(data=swdf_C))

    ## SWS estimation ################
    dim = ddata_f_0.shape
    SWS_Estimator = SWS_Estimation(x_range=swse_x_range, z_clip = swse_z_clip, frames_range = swse_frames,
                                   d=swse_d, fri = pwi_fri, interp_factor=swse_interp_factor, interp_order=swse_interp_order, 
                                    px_pitch=px_size*1e-3, sws_range=swse_SWV_range)
    SWS_Estimator.prepare(input_shape = ddata_f_0.shape)
    SWV_0 = SWS_Estimator.process(data=ddata_f_0)
    SWV_A = SWV_0.get()
    SWS_Estimator.prepare(input_shape = ddata_f_1.shape)
    SWV_1 = SWS_Estimator.process(data=ddata_f_1)
    SWV_B = SWV_1.get()
    SWV_2 = SWS_Estimator.process(data=ddata_f_2)
    SWV_C = SWV_2.get()
    
    sws_dim = SWV_0.shape
    
    ## Save SWS maps
    scipy.io.savemat(directory2 + 'sws_A_grid' + str(gridStep) + '_iFactor' + str(iFactor) + '_iOrder' + str(iOrder) + '_id_' + str(idx+40) + '_p' + str(p) +'.mat', dict(data=SWV_A)) 
    scipy.io.savemat(directory2 + 'sws_B_grid' + str(gridStep) + '_iFactor' + str(iFactor) + '_iOrder' + str(iOrder) + '_id_' + str(idx+40) + '_p' + str(p) +'.mat', dict(data=SWV_B)) 
    scipy.io.savemat(directory2 + 'sws_C_grid' + str(gridStep) + '_iFactor' + str(iFactor) + '_iOrder' + str(iOrder) + '_id_' + str(idx+40) + '_p' + str(p) +'.mat', dict(data=SWV_C))      
    
    ## Compounding
    # Mask the r maps
    xn = SWV_0.shape[3]
    x = cp.linspace(0, xn, xn)
    
    #A_LR = [0, 99]
    #A_RL = [0, 200]

    #B_LR = [0, 150]
    #B_RL = [50, 200]

    #C_LR = [0, 200]
    #C_RL = [101, 200]    
    
    yA_0 = 1 / ( 1 + cp.exp(-a*(x-A_LR[1])))
    yA_1 = 0
    
    yB_0 = 1 / (1 + cp.exp(-a*(x-B_LR[1])))
    yB_1 = 1 - (1 / (1 + cp.exp(-a*(x-B_RL[0]))))
    
    yC_0 = x * 0
    yC_1 = 1 - (1 / (1 + cp.exp(-a*(x-C_RL[0]))))
    
    
    #SWV_0[1, 0, :, A_LR[0]:A_LR[1]] = 0
    #SWV_0[1, 1, :, A_RL[0]:A_RL[1]] = 0

    #SWV_1[1, 0, :, B_LR[0]:B_LR[1]] = 0
    #SWV_1[1, 1, :, B_RL[0]:B_RL[1]] = 0

    #SWV_2[1, 0, :, C_LR[0]:C_LR[1]] = 0
    #SWV_2[1, 1, :, C_RL[0]:C_RL[1]] = 0 
    
    SWV_0[1, 0, ...] = SWV_0[1, 0, ...] * yA_0
    SWV_0[1, 1, ...] = SWV_0[1, 1, ...] * yA_1
    SWV_1[1, 0, ...] = SWV_1[1, 0, ...] * yB_0
    SWV_1[1, 1, ...] = SWV_1[1, 1, ...] * yB_1
    SWV_2[1, 0, ...] = SWV_2[1, 0, ...] * yC_0
    SWV_2[1, 1, ...] = SWV_2[1, 1, ...] * yC_1
    

    ## Compound image
    r_im = cp.zeros([6, sws_dim[2], sws_dim[3]])
    sws_im = cp.zeros([6, sws_dim[2], sws_dim[3]])
    
    r_im[0, ...] = SWV_0[1, 0, ...]
    r_im[1, ...] = SWV_0[1, 1, ...]
    r_im[2, ...] = SWV_1[1, 0, ...]
    r_im[3, ...] = SWV_1[1, 1, ...]
    r_im[4, ...] = SWV_2[1, 0, ...]
    r_im[5, ...] = SWV_2[1, 1, ...]
    
    sws_im[0, ...] = SWV_0[0, 0, ...]
    sws_im[1, ...] = SWV_0[0, 1, ...]
    sws_im[2, ...] = SWV_1[0, 0, ...]
    sws_im[3, ...] = SWV_1[0, 1, ...]
    sws_im[4, ...] = SWV_2[0, 0, ...]
    sws_im[5, ...] = SWV_2[0, 1, ...]    
    
    r_sum = cp.squeeze(cp.sum(r_im, axis=0))
    r_sum[r_sum==0] = 10e-6
    sws_map = cp.sum((sws_im * r_im), axis=0) / r_sum  
    
    # Save the final image
    data_cpu = sws_map.get()    
    scipy.io.savemat(directory2+'sws'+'_grid'+str(gridStep)+'_iFactor'+str(iFactor)+'_iOrder'+str(iOrder)+'_id_'+str(idx+40)+'_p'+str(p)+'.mat', dict(data=data_cpu))     

    
# Parser    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single TXPB parametric script.")
    parser.add_argument("--iOrder", dest="iOrder", type=int)
    parser.add_argument("--iFactor", dest="iFactor", type=int)
    parser.add_argument("--gridStep", dest="gridStep", type=float)
    parser.add_argument("--idx", dest="idx", type=int)
    parser.add_argument("--p", dest="p", type=int, default=20)
    args = parser.parse_args()
    args = main(args.iOrder, args.iFactor, args.gridStep, args.idx, args.p)      
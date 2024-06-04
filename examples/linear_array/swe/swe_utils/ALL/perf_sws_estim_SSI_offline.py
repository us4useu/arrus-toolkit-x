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

def main(iOrder, iFactor, idx):

    ### General settings ################
    dataset_id  = idx
    #directory   = '/media/damian/8A4F-A24E/Test11/'
    directory = "./Parametric_tests/Test_13/"
    directory2  = "./Parametric_tests/Test_13/rec_sws_estim/"

    # Constants
    c              = 1540.0
    probe_elements = 128
    probe_pitch    = 0.298e-3
    fs             = 65e6
    t0             = 43

    # Sequence parameters
    pwi_txFreq   = 4.4e6
    pwi_nCycles  = 2
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
    px_size = 0.2  # in [mm]
    x_grid = np.arange(-20, 20, px_size) * 1e-3
    z_grid = np.arange(0, 50, px_size)   * 1e-3
    rx_tang_limits = [-0.7, 0.7]

    # Shear wave detection
    swd_mode              = 'kasai'
    swd_zGate_length      = 4
    swd_ensemble_length   = 4

    # Input parameters
    df_sws_range = [0.5, 4.5];
    df_f_range   = [40.0, 700.0];
    df_k_range   = 0.9;

    # SWS estimation
    swse_interp_factor = iFactor
    swse_interp_order  = iOrder
    swse_d             = 20
    swse_frames        = [0, 90];
    swse_SWV_range     = [0.5, 5.0];
    swse_x_range       = [[0, 420], [0, 420]]
    #swse_x_range       = [[90, 300], [0, 110]]
    swse_z_clip        = [5, 10]
    
    # Compounding
    # Regions to mask to 0
    a    = 0.7
    A_LR = [0, 99]
    A_RL = [0, 200]

    B_LR = [0, 150]
    B_RL = [50, 200]

    C_LR = [0, 200]
    C_RL = [101, 200]

    # Post-processing
    #median_filter_size = int(5*0.2/gridStep)

    ### LOAD the dataset and crop data ###################
    # Load swdf SSI datasets
    data = sp.io.loadmat(directory + 'rf_id_' + str(dataset_id) + '_shift_-30.mat')
    swdf_0 = data["data"]
    print("Input data shape:")
    print(swdf_0.shape)
    
    data = sp.io.loadmat(directory + 'rf_id_' + str(dataset_id) + '_shift_0.mat')
    swdf_1 = data["data"]
    
    data = sp.io.loadmat(directory + 'rf_id_' + str(dataset_id) + '_shift_30.mat')
    swdf_2 = data["data"]   
    
    swdf_0 = cp.asarray(swdf_0)
    swdf_1 = cp.asarray(swdf_1)
    swdf_2 = cp.asarray(swdf_2)

    ### Processing #######################################
    
    ## SWS estimation 
    dim = swdf_0.shape
    SWS_Estimator = SWS_Estimation(x_range=swse_x_range, z_clip = swse_z_clip, frames_range = swse_frames,
                                   d=swse_d, fri = pwi_fri, interp_factor=swse_interp_factor, interp_order=swse_interp_order, 
                                    px_pitch=px_size*1e-3, sws_range=swse_SWV_range)
    SWS_Estimator.prepare(input_shape = dim)
    SWV_0 = SWS_Estimator.process(data=swdf_0)
    SWV_1 = SWS_Estimator.process(data=swdf_1)
    SWV_2 = SWS_Estimator.process(data=swdf_2)
    
    sws_dim = SWV_0.shape
    
    # Save the pre-compunding results
    data_A_cpu = SWV_0.get()
    data_B_cpu = SWV_1.get()
    data_C_cpu = SWV_2.get()
    scipy.io.savemat(directory2 + 'sws_A_' + '_iFactor' + str(iFactor) + '_iOrder' + str(iOrder) + '_id_' + str(idx) + '.mat', dict(data=data_A_cpu))   
    scipy.io.savemat(directory2 + 'sws_B_' + '_iFactor' + str(iFactor) + '_iOrder' + str(iOrder) + '_id_' + str(idx) + '.mat', dict(data=data_B_cpu)) 
    scipy.io.savemat(directory2 + 'sws_C_' + '_iFactor' + str(iFactor) + '_iOrder' + str(iOrder) + '_id_' + str(idx) + '.mat', dict(data=data_C_cpu)) 
    
    ## Compounding
    # Mask the r maps
    
    x = cp.linspace(0, 200, 200)
    
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
    

    # Compound image
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
    scipy.io.savemat(directory2 + 'sws' + '_iFactor' + str(iFactor) + '_iOrder' + str(iOrder) + '_id_' + str(idx) + '.mat', dict(data=data_cpu))     

    
# Parser    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single TXPB parametric script.")
    parser.add_argument("--iOrder", dest="iOrder", type=int)
    parser.add_argument("--iFactor", dest="iFactor", type=int)
    parser.add_argument("--idx", dest="idx", type=int)
    args = parser.parse_args()
    args = main(args.iOrder, args.iFactor, args.idx)      
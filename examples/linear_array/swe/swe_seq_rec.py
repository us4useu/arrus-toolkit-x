######################################################################################
# Basic, single push, 2-D SWE data acqusition configuration with us4R-lite
# Example script
#
# File: seq_seq.py
# Author: Damian Cacko, us4us Ltd.
# Revision: 1.0.0
#
# Aim of the example is to demonstrate how to:
#    - configure push pulse generation
#    - configure shear wave tracking sequence
#    - run the data acqusition
#    - perform data processing with GPU using Arrus utils or custom functions
#    - dump collected data for off-line processing
#
# Description:
#    This script configures the acquisition constisting of a single push pulse 
#    followed by a fast plane-wave tracking sequence (CPWC). Parameters of both
#    parts (frequencies, apertures and so on) can be controlled by the user
#    using top-level variables. After configuration of the TX/RX sequence, the
#    script runs the acqusution, reconstruct data and dumps collected data to the file for
#    further off-line processing. The processing included in the script involves pre-processing
#    (RF filtering, down-conversion), beamforming, and shear wave detection.
#
######################################################################################


# Import all the required packages
import arrus
import arrus.session
import arrus.utils.imaging
import arrus.utils.us4r
import arrus.ops.tgc
import arrus.medium

from arrus.ops.us4r import *
from arrus.utils.imaging import *

import queue
import numpy as np
import scipy as sp
import scipy.io
from scipy.signal import remez
from collections import deque
import pickle
import time

from SWE_utils_cupy_pipelined2 import *

# Logging settings
arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("swe_seq.log", arrus.logging.TRACE)


def main():

    ##### Parameters of the TX/RX sequence (edit below this line) #####
    protofile        = "us4r_esaote.prototxt"  # name of the prototxt file to be used
    
    id               = 4                 # id of the file to store collected RF data
    
    push_hv          = 50                # Push pulse HV voltage [V] 
    push_freq        = 130.0/20*1e6      # Push pulse center frequency [Hz]. Divisor of 130.0 must be even!
    push_length      = 400e-6            # Push pulse duration [s]
    push_txelements  = 96                # Push pulse aperture size [# of probe elements]
    push_center      = 0                 # Push pulse aperture lateral position [m]. 0 is the center of the probe.
    push_focus       = 30e-3             # Push pulse focus depth [m]
    
    pwi_hv        = push_hv + 10         # PWI tracking pulse HV voltage [V]. Range: [push_hv+2; push_hv+20]    
    pwi_freq      = 130.0/20*1e6         # PWI tracking pulse center frequency [Hz]. Divisor of 130.0 must be even!
    pwi_txncycles = 2                    # PWI tracking pulse number of cycles 
    pwi_pri       = 120e-6               # PWI tracking sequence pulse repetition interval [s]. Minimum is roundtrip time (defined by n_samples at 65 MHz sampling clock) +7us.
    pwi_angles    = [0.0]                # PWI tracking sequnce CPWC angles. List of angles in [degrees], for example: [0.0], or [-4.0, 0.0, 4.0]
    n_samples     = 5120                 # Number of samples to be collected in each RX operation of tracking sequence. Must be multiple of 64.
    imaging_pw_n_repeats = 80            # Number of frames to be captured in tracking sequence.

    sos           = 1540.0               # Speed of sound of the medium [m/s]

    ##### Parameters of the TX/RX sequence (edit above this line) #####


    ##### Reconstruction parameters (edit below this line) #####
    # Constants            
    fs             = 65e6                 # RF samplings frequency [Hz]
    
    # RF Filter
    rf_filter_band        = [6e6, 12e6]   # Desired pass band corenr frequencies, Hz
    rf_filter_trans_width = 1e6           # Width of transition from pass band to stop band, Hz
    rf_filter_numtaps     = 256           # Size of the FIR filter (numer of taps).
    
    # Post down conversion IQ filtering
    demod_f = pwi_freq                        # Demodulation frequency [Hz]
    demod_filter_cutoff = 0.5 * demod_f       # Desired cutoff frequency [Hz] for post-demodulation I and Q products
    demod_filter_trans_width = 0.5 * demod_f  # Width of transition from pass band to stop band, [Hz] for post-demodulation I and Q products
    demod_filter_numtaps = 64                 # Size of the FIR filter (numer of taps).
    
    # Beamforming
    px_size = 0.2                                  # grid step (pixel size) [mm]                        
    x_grid = np.arange(-25, 25, px_size) * 1e-3    # x grid for reconstruction
    z_grid = np.arange(0, 50, px_size)   * 1e-3    # z grid for reconstruction
    rx_tang_limits = [-0.7, 0.7]                   # tangent values limits to include when apodizing directivity of elements during beamforming
    
    # Shear wave detection
    swd_mode              = 'kasai'                # Shear wave detection method: 'kasai' (recommended) or 'loupas'
    swd_zGate_length      = int(4.0*0.2/px_size)   # Kernel size for Kasai autocorreleator in z axis
    swd_ensemble_length   = 4                      # Ensemble for Kasai autocorreleator (in slow-time axis)
    
    ##### Reconstruction parameters (edit above this line) #####
    
    
    ##### Process parameters #####
    datafile = "datasets/data_id_"  + str(id)
    push_txncycles = int(push_length * push_freq)
    if(pwi_hv<push_hv):
        pwi_hv = push_hv + 2

    if(pwi_hv>(push_hv+10)):
        pwi_hv = push_hv + 10    

    hv_voltage_0 = pwi_hv
    hv_voltage_1 = push_hv
    push_pri = push_length + 50e-6 # 50us is the time between end of push pulse and strat of first tracking pulse. Can be reduced down to minimum 7us if required.

    ### Design filters and perform RF data filtering ###################
    ## Design a band-pass FIR filter for RF filtering of raw channel data
    band = rf_filter_band                # Desired pass band, Hz
    trans_width = rf_filter_trans_width  # Width of transition from pass band to stop band, Hz
    numtaps = rf_filter_numtaps          # Size of the FIR filter.
    
    edges = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5*fs]
    rf_fir_taps = scipy.signal.remez(numtaps, edges, [0, 1, 0], fs=fs)
    
    ## Design a low-pass FIR filter for filtering of down-conversion products
    # Specify the filter parameters    
    cutoff = demod_filter_cutoff            # Desired cutoff frequency, Hz
    trans_width = demod_filter_trans_width  # Width of transition from pass band to stop band, Hz
    numtaps = demod_filter_numtaps          # Size of the FIR filter.
    iq_fir_taps = scipy.signal.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*fs], [1, 0], fs=fs)

    ## Create medium
    medium = arrus.medium.Medium(name="cirs049a", speed_of_sound=sos)
    
    ##### Here starts communication with the device. ######
    with arrus.Session(protofile, medium=medium) as sess:
        us4r = sess.get_device("/Us4R:0")

        us4r.set_hv_voltage((hv_voltage_1, hv_voltage_1), (hv_voltage_0, hv_voltage_0))
        
        n_elements = us4r.get_probe_model().n_elements
 
        ## Create SWE TX/RX sequence ##
        
        # Build the pushing sequence
        push_sequence = [
            TxRx(
                Tx(aperture=arrus.ops.us4r.Aperture(center=push_center, size=push_txelements),
                    excitation=Pulse(center_frequency=push_freq, n_periods=push_txncycles, inverse=False, amplitude_level=1),
                    focus=push_focus,
                    angle=0,
                    speed_of_sound=medium.speed_of_sound
                ),
                Rx(
                    aperture=arrus.ops.us4r.Aperture(center=0, size=0),
                    sample_range=(0, n_samples),
                    downsampling_factor=1
                ),
                pri=push_pri
            )
        ]

        # Build the imaging sequence    
        imaging_sequence = [
            TxRx(
                Tx(
                    aperture=arrus.ops.us4r.Aperture(center=0), 
                    excitation=Pulse(center_frequency=pwi_freq, n_periods=pwi_txncycles, inverse=False, amplitude_level=2),
                    focus=np.Inf,  # [m]
                    angle=angle/180*np.pi,  # [rad]
                    speed_of_sound=medium.speed_of_sound
                ),
                Rx(
                    aperture=arrus.ops.us4r.Aperture(center=0),
                    sample_range=(0, n_samples),
                    downsampling_factor=1
                ),
                pri=pwi_pri
            )
            for angle in pwi_angles
            ]*int(imaging_pw_n_repeats/len(pwi_angles))

        # Combine (push + imaging) sequences 
        seq = TxRxSequence(ops=push_sequence+imaging_sequence)   

        ## Declare the complete scheme to execute on the devices ##
        scheme = Scheme(
            # Run the provided sequence.
            tx_rx_sequence=seq,
            work_mode="MANUAL",
            # Processing pipeline to perform on the GPU device. Here it is very limited
            processing=Pipeline(
            steps=(
                RemapToLogicalOrder(),
                Output(), # Output node: RF data
                Transpose(axes=(0, 1, 3, 2)),
                SelectFrames(np.arange(1, imaging_pw_n_repeats)),  # First frame is removed (reverberation effect)
                FirFilter(taps=rf_fir_taps),
                QuadratureDemodulation(),
                Decimation(filter_type="fir", filter_coeffs=iq_fir_taps, decimation_factor=1),
                ReconstructLri(x_grid=x_grid, z_grid=z_grid, rx_tang_limits=rx_tang_limits),  # Beamforming
                Squeeze(),
                AngleCompounding(nAngles=len(pwi_angles)), # CPWC if multiple plane waves are used 
                Output(), # Output node: beamformed I/Q data
                ShearwaveDetection(mode=swd_mode, packet_size=swd_ensemble_length, z_gate=swd_zGate_length, fc=pwi_freq), # Kasai's method
                # Automatic output node: shear wave motion data
                ),
            placement="/GPU:0"
            )
        )
        
        # Upload the scheme on the us4r-lite device.
        buffer, metadata = sess.upload(scheme)

        # Configure the TGC profile
        us4r.set_tgc(arrus.ops.tgc.LinearTgc(start=30, slope=400))   # 30 dB + 800 dB/m (30 dB + 4 dB/cm - distance assumes one side propagation - here TGC saturates at max 54 dB for echos from depth of 4cm)
        
        # Run the TX/RX sequence and wait for data
        sess.run()
        data = buffer.get()
        sess.stop_scheme()
        
        # Save collected data to the file
        rf_data   = data[2]  # raw RF data
        bf_data   = data[1]  # beamformed I/Q data
        swd_data  = data[0]  # shear wave motion data 
        
        
        # Comment the lines that are not required
        if(1): # for Python
            pickle.dump({"data":rf_data, "metadata": metadata}, open(datafile + "rf.pkl", "wb"))
            pickle.dump({"data":bf_data}, open(datafile + "bf.pkl", "wb"))
            pickle.dump({"data":swd_data}, open(datafile + "swd.pkl", "wb"))


        if(0): # for Matlab
            scipy.io.savemat(datafile + 'rf.mat', dict(data=rf_data)) 
            scipy.io.savemat(datafile + 'bf.mat', dict(data=bf_data)) 
            scipy.io.savemat(datafile + 'swd.mat', dict(data=dswd_ata)) 
    
    # When we exit the above scope, the session and scheme is properly closed.
    print("Stopping the example.")


if __name__ == "__main__":
    main()

# Script ends here
######################################################################################

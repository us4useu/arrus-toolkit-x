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
#    - dump collected data for off-line processing
#
# Description:
#    This script configures the acquisition constisting of a single push pulse 
#    followed by a fast plane-wave tracking sequence (CPWC). Parameters of both
#    parts (frequencies, apertures and so on) can be controlled by the user
#    using top-level variables. After configuration of the TX/RX sequence, the
#    script runs the acqusution and dumps collected data to the file for an
#    off-line processing.
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
    TxRxSequence
)
from arrus.utils.imaging import (
    Pipeline,
    SelectFrames,
    Squeeze,
    Lambda,
    RemapToLogicalOrder
)

import queue
import numpy as np
from collections import deque
import pickle
import time

# Logging settings
arrus.set_clog_level(arrus.logging.DEBUG)
arrus.add_log_file("swe_seq.log", arrus.logging.TRACE)


def main():

    ##### Parameters of the TX/RX sequence (edit below this line) #####
    protofile        = "us4r_atl.prototxt"  # name of the prototxt file to be used
    
    id               = 0                 # id of the file to store collected RF data
    
    push_hv          = 50                # Push pulse HV voltage [V] 
    push_freq        = 130.0/26*1e6      # Push pulse center frequency [Hz] [TBD]
    push_length      = 500e-6            # Push pulse duration [s]
    push_txelements  = 128               # Push pulse aperture size [# of probe elements]
    push_center      = 0                 # Push pulse aperture lateral position [m]. 0 is the center of the probe.
    push_focus       = 25e-3             # Push pulse focus depth [m]
    
    pwi_hv        = push_hv + 20         # PWI tracking pulse HV voltage [V]. Range: [push_hv+2; push_hv+20]    
    pwi_freq      = 130.0/26*1e6         # PWI tracking pulse center frequency [Hz]
    pwi_txncycles = 2                    # PWI tracking pulse number of cycles 
    pwi_pri       = 120e-6               # PWI tracking sequence pulse repetition interval [s]
    pwi_angles    = [0.0]                # PWI tracking sequnce CPWC angles. List of angles in [degrees], for example: [0.0], or [-4.0, 0.0, 4.0]
    n_samples     = 5*1024-256           # Number of samples to be collected in each RX operation of tracking sequence.
    imaging_pw_n_repeats = 80            # Number of frames to be captured in tracking sequence.

    sos           = 1540.0               # Speed of sound of the medium [m/s]
    
    ##### Parameters of the TX/RX sequence (edit above this line) #####
    
    ##### Process parameters #####
    datafile = "datasets/data_id_"  + str(id)
    push_txncycles = int(push_length * push_freq)
    if(pwi_hv<push_hv):
        pwi_hv = push_hv + 2

    if(pwi_hv>(push_hv+20)):
        pwi_hv = push_hv + 20    

    hv_voltage_0 = pwi_hv
    hv_voltage_1 = push_hv
    push_pri = push_length + 50e-6

    medium = arrus.medium.Medium(name="cirs049a", speed_of_sound=sos)
    
    ##### Here starts communication with the device. ######
    with arrus.Session(protofile, medium=medium) as sess:
        us4r = sess.get_device("/Us4R:0")

        us4r.set_maximum_pulse_length(510e-6)
        us4r.set_hv_voltage((hv_voltage_0, hv_voltage_0), (hv_voltage_1, hv_voltage_1))
        
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
                    excitation=Pulse(center_frequency=pwi_freq, n_periods=pwi_txncycles, inverse=False, amplitude_level=0),
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
                ),
                placement="/GPU:0"
            )
        )
        
        # Upload the scheme on the us4r-lite device.
        buffer, metadata = sess.upload(scheme)

        # Configure the TGC profile
        us4r.set_tgc(arrus.ops.tgc.LinearTgc(start=52, slope=2e2))
        
        # Run the TX/RX sequence and wait for data
        sess.run()
        data = buffer.get()[0]
        sess.stop_scheme()
        
        # Save collected data to the file
        pickle.dump({"data":data, "metadata": metadata}, open(datafile + ".pkl", "wb"))

    
    # When we exit the above scope, the session and scheme is properly closed.
    print("Stopping the example.")


if __name__ == "__main__":
    main()

# Script ends here
######################################################################################

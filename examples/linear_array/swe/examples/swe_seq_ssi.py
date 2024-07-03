######################################################################################
# Basic, SSI scheme, 2-D SWE data acqusition configuration with us4R-lite
# Example script
#
# File: swe_seq_ssi.py
# Author: Damian Cacko, us4us Ltd.
# Revision: 1.0.0
#
# Aim of the example is to demonstrate how to:
#    - configure push pulse generation consisting of multiple push beams
#    - configure shear wave tracking sequence for each push beam
#    - combine multiple sequences for a complex scheme like in SSI method
#    - run the data acqusition
#    - dump collected data for off-line processing
#
# Description:
#    This script configures the acquisition constisting of a push pulse sequence 
#    as in SSI method: 3 beams focused at subsequent depths, followed by a fast 
#    plane-wave tracking sequence (CPWC). Then pushing sequence is repeated at different
#    lateral position and CPWC is repeated. Then the third lateral position for the same 
#    scheme is used. Such sequence allows to cover full FOV with shear waves to
#    reconstruct elasticity map in the whole FOV.
#    Parameters of both parts (frequencies, apertures and so on) can be controlled by the user
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
    RemapToLogicalOrder,
    Graph
)

import queue
import numpy as np
import arrus.ops.tgc
import arrus.medium
from collections import deque
import pickle
import time

# Logging settings
arrus.set_clog_level(arrus.logging.DEBUG)
arrus.add_log_file("swe_seq_ssi.log", arrus.logging.TRACE)


def main():

    ##### Parameters of the TX/RX sequence (edit below this line) #####
    protofile        = "us4r_atl.prototxt"  # name of the prototxt file to be used
    
    id               = 0    # id of the file to store collected RF data
    
    # Pushing sequence parameters
    push_hv          = 10                                                    # Push pulse HV voltage [V] 
    push_freq        = np.array([130.0/26*1e6, 130.0/28*1e6, 130.0/30*1e6])  # Push pulse center frequencies [Hz] [TBD]
    push_length      = np.array([50e-6, 50e-6, 50e-6])                       # Push pulse durations [s]
    push_txelements  = np.array([64, 64, 64])                                # Push pulse aperture sizes [# of probe elements]
    push_focus       = np.array([15e-3, 25e-3, 35e-3])                       # Push pulse focus depth [m]
    push_center      = np.array([63.5, 63.5, 63.5])                          # Push pulse aperture center elements indices (for each lateral position of SSI beam) 
    
    # Tracking sequence parameters
    pwi_hv        = push_hv + 20      # PWI tracking pulse HV voltage [V]. Range: [push_hv+2; push_hv+20]   
    pwi_freq      = 130.0/32*1e6      # PWI tracking pulse center frequency [Hz]
    pwi_txncycles = 2                 # PWI tracking pulse number of cycles 
    pwi_pri       = 120e-6            # PWI tracking sequence pulse repetition interval [s]
    pwi_angles    = [0.0]             # PWI tracking sequnce CPWC angles. List of angles in [degrees], for example: [0.0], or [-4.0, 0.0, 4.0]
    n_samples     = 5*1024-256        # Number of samples to be collected in each RX operation of tracking sequence.
    imaging_pw_n_repeats = 2          # Number of frames to be captured in tracking sequence

    # SSI sequence parameters
    ssi_sri       = 200e-3            # Sequence repetition interval [s]. Means interval between {push + imaging} sequences at each lateral position.

    # Other parametres
    sos           = 1540.0            # Speed of sound of the medium [m/s]

    ##### Parameters of the TX/RX sequence (edit above this line) #####

    # Process parameters
    datafile = "datasets/data_id_"  + str(id)
    push_txncycles = push_length * push_freq
    push_txncycles.astype(int)
    if(pwi_hv<push_hv):
        pwi_hv = push_hv + 2

    if(pwi_hv>(push_hv+20)):
        pwi_hv = push_hv + 20        

    hv_voltage_0 = pwi_hv
    hv_voltage_1 = push_hv
    push_pri = push_length + 70e-6
    
    ##### Here starts communication with the device. ######
    medium = arrus.medium.Medium(name="cirs049a", speed_of_sound=sos)
    with arrus.Session(protofile, medium=medium) as sess:
        us4r = sess.get_device("/Us4R:0")
        n_elements = us4r.get_probe_model().n_elements
        us4r.set_maximum_pulse_length(510e-6)
        us4r.set_hv_voltage((hv_voltage_0, hv_voltage_0), (hv_voltage_1, hv_voltage_1))
        
        ## Create SWE TX/RX sequence ##

        # Create an empty list to be filled with sequence definitions
        push_sequence = [0] * len(push_center) 
        for i in range(len(push_sequence)):
            push_sequence[i] = [0] * len(push_focus)

        # Build the pushing sequences
        for k in range(len(push_center)):     # sweep through lateral positions of the beams
            for i in range(len(push_focus)):  # sweep through axial focus positions
              
                push_sequence[k][i] = TxRx(
                        Tx(aperture=arrus.ops.us4r.Aperture(center_element=push_center[i], size=push_txelements[i]), 
                            excitation=Pulse(center_frequency=push_freq[i], n_periods=int(push_txncycles[i]), inverse=False, amplitude_level=1),
                            focus=push_focus[i], 
                            angle=0,
                            speed_of_sound=medium.speed_of_sound
                        ),
                        Rx(
                            aperture=arrus.ops.us4r.Aperture(center=0, size=0), # empty rx aperture
                            sample_range=(0, n_samples),
                            downsampling_factor=1
                        ),
                        pri=push_pri[i]
                    )     

        # Build the imaging sequence
        imaging_sequence = [
            TxRx(
                Tx(
                    aperture=arrus.ops.us4r.Aperture(center=0),
                    excitation=Pulse(center_frequency=pwi_freq, n_periods=pwi_txncycles, inverse=False, amplitude_level=0),
                    focus=np.inf,  # [m]
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

        # Build sub-sequences of a complete scheme
        seq0 = TxRxSequence(ops=push_sequence[0]+imaging_sequence, sri=5e-3, name="Seq0") # each lateral SSI push has its own sequence
        seq1 = TxRxSequence(ops=push_sequence[1]+imaging_sequence, sri=5e-3, name="Seq1") 
        seq2 = TxRxSequence(ops=push_sequence[2]+imaging_sequence, sri=5e-3, name="Seq2") 
        
        # Declare the complete scheme to execute on the devices.
        scheme = Scheme(
            # Run the provided sequence
            tx_rx_sequence=(seq0, seq1, seq2),  # TX/RX sequence consists of 3 sub-sequences
            work_mode="MANUAL",
            # Processing pipeline to perform on the GPU device. Here it is in a form of graph. Each sub-sequence has its own pipeline to process data.
            processing=Graph(
                operations={
                    Pipeline(
                        steps=(RemapToLogicalOrder(),),
                        placement="/GPU:0",
                        name="Pipeline0"
                    ),
                    Pipeline(
                        steps=(RemapToLogicalOrder(),),
                        placement="/GPU:0",
                        name="Pipeline1"
                    ),
                    Pipeline(
                        steps=(RemapToLogicalOrder(),),
                        placement="/GPU:0",
                        name="Pipeline2"
                    ),
                },
                dependencies={
                    "Pipeline0": "Seq0",
                    "Pipeline1": "Seq1",
                    "Pipeline2": "Seq2",
                    "Output:0": "Pipeline0",
                    "Output:1": "Pipeline1",
                    "Output:2": "Pipeline2",
                }
            )
        )
  
        # Upload the scheme on the us4r-lite device
        buffer, metadata = sess.upload(scheme)

        # Configure the TGC profile
        us4r.set_tgc(arrus.ops.tgc.LinearTgc(start=52, slope=2e2))

        # Run the TX/RX sequence and wait for data
        sess.run()
        data = buffer.get()
        sess.stop_scheme()
        
        # Save collected data to the file (pickle) 
        pickle.dump({"data":data, "metadata": metadata}, open(datafile + ".pkl", "wb"))

    # When we exit the above scope, the session and scheme is properly closed.
    print("Stopping the example.")


if __name__ == "__main__":
    main()

# Script ends here
######################################################################################

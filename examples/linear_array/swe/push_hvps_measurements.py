# Pojedynczy strzał push z zapisem danych pomiarów HVPS
# Anliza pomiarów w hvps_analysis.ipynb
# Powtarzany w pętli push jest w _loop.py
# Rekonstrukcja SWE: reconstruct_SWE.ipynb
import numpy as np

import time

import arrus
import arrus.medium
import arrus.ops.tgc
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
from arrus.utils.imaging import *

arrus.set_clog_level(arrus.logging.TRACE)
arrus.add_log_file("hvps_measurement.log", arrus.logging.TRACE)


def main():

    # Parameters
    id            = 100
    tx_elements   = 128
    hv_voltage_1  = 30
    tx_length     = 100e-6
    tx_freq       = 130.0/26*1e6
    
    
    hv_voltage_0  = hv_voltage_1 + 20

    #hv_voltage_0 = 30
    #hv_voltage_1 = 10
    
    amplitude_level = 1
    tx_focus = 25e-3
    
    tx_ncycles = int(tx_length * tx_freq)
    #tx_ncycles = 30
    tx_pri = tx_length + 100e-6 
    
    
    # Here starts communication with the device.
    measurements = []
    medium = arrus.medium.Medium(name="cirs049a", speed_of_sound=1540)
    #with arrus.Session("us4r_1oem_atl.prototxt") as sess:
    with arrus.Session("us4r.prototxt") as sess:    
        us4r = sess.get_device("/Us4R:0")
        #s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        #for v in s:
        #    us4r.set_hv_voltage((int(v*hv_voltage_0/10), int(v*hv_voltage_0/10)), (int(v*hv_voltage_1/10), int(v*hv_voltage_1/10)))
        us4r.set_hv_voltage((int(hv_voltage_0), int(hv_voltage_0)), (int(hv_voltage_1), int(hv_voltage_1)))
        
        n_elements = us4r.get_probe_model().n_elements
        n_us4oems = us4r.n_us4oems

        for oem in range(n_us4oems):
            us4r.get_us4oem(oem).set_hvps_sync_measurement(n_samples=509, frequency=1e6)

        # Apertures
        #tx_aperture = arrus.ops.us4r.Aperture(center=0, size=tx_elements)
        
        tx_aperture = np.zeros(n_elements).astype(bool)
        tx_aperture[0:tx_elements] = True
        #tx_aperture[24:32] = True  # pulser[0]
        #tx_aperture[0:8] = True   # pulser[3]
        #tx_aperture[8:16] = True  # pulser[2]
        #tx_aperture[16:24] = True  # pulser[1]?
        #tx_aperture[64:72] = True
        #tx_aperture[72:80] = True
        #tx_aperture[80:88] = True
        #tx_aperture[88:96] = True
        #tx_aperture[56:64] = 0
           
        rx_aperture = np.zeros(n_elements).astype(bool)
        #rx_aperture[:n_us4oems*32] = True
        rx_aperture[0] = True
        delays = np.arange(0, np.count_nonzero(tx_aperture), 1) 
        #delays = np.zeros(np.count_nonzero(tx_aperture))
        delays = delays * 8 * 1e-9
        #delays[0:8] = delays[0:8]
        #delays[8:16] =  delays[8:16] + 1e-6
        #delays[16:24] = delays[16:24] + 3e-6
        #delays[24:32] = delays[24:32] + 5e-6
        #delays[64:72] = delays[64:72] + 7e-6
        #delays[72:80] = delays[72:80] + 8e-6
        #delays[80:88] = delays[80:88] + 9e-6
        #delays[88:96] = delays[88:96] + 10e-6
        #delays = delays * 0
        #delays = delays + 400e-9
            
        seq = TxRxSequence(
            ops=[
                TxRx(
                    Tx(aperture=tx_aperture,
                        excitation=Pulse(center_frequency=tx_freq, n_periods=tx_ncycles, inverse=False, amplitude_level=amplitude_level),
                        #delays = delays),
                        focus=tx_focus,
                        angle=0,  # [rad]
                        speed_of_sound=medium.speed_of_sound),
                    Rx(aperture=rx_aperture, sample_range=(0, 4096), downsampling_factor=1),
                    pri=tx_pri
                ),
            ],
            tgc_curve=[],  # [dB]
            sri=500e-3
        )
        scheme = Scheme(
            tx_rx_sequence=seq,
            work_mode="MANUAL",
            processing=Pipeline(steps=(Squeeze(), ), placement="/GPU:0")
        )
        
        # Upload the scheme on the us4r-lite device.
        buffer, metadata = sess.upload(scheme)
        sess.run()  # Trigger execution asynchronously
        # Wait for the data to make sure that TX/RX finished
        buffer.get()
        sess.stop_scheme()
        oem_measurements = []
        for us4oem in range(n_us4oems):
            oem = us4r.get_us4oem(us4oem)
            measurement = oem.get_hvps_measurement()
            oem_measurements.append(measurement.get_array())
        measurements.append(oem_measurements)
    
    output_file = "hvps_measurements/hvps_push_id_" + str(id) + ".npy"
    np.save(output_file, measurements)
    print(f"Saved data to {output_file}")


if __name__ == "__main__":
    main()

import numpy as np

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
    hv_voltage_0  = 10
    hv_voltage_1  = 7
    amplitude_level = 0
    tx_freq = 130.0/26*1e6  # 5 MHz
    tx_ncycles = 150 * 5    
    
    # Here starts communication with the device.
    measurements = []
    with arrus.Session("us4r.prototxt") as sess:  #"us4r_1oem_atl.prototxt"
        us4r = sess.get_device("/Us4R:0")
        us4r.set_hv_voltage((hv_voltage_0, hv_voltage_0), (hv_voltage_1, hv_voltage_1))
        n_elements = us4r.get_probe_model().n_elements
        n_us4oems = us4r.n_us4oems

        #for oem in range(n_us4oems):
        #    us4r.get_us4oem(oem).set_hvps_sync_measurement(n_samples=509, frequency=1e6)

        for channel in range(n_elements):  # n_elements
            
            for oem in range(n_us4oems):
                us4r.get_us4oem(oem).set_hvps_sync_measurement(n_samples=509, frequency=1e6)
            
            print(f"channel {channel}")
            # Apertures
            tx_aperture = np.zeros(n_elements).astype(bool)
            tx_aperture[channel] = True
            rx_aperture = np.zeros(n_elements).astype(bool)
            #rx_aperture[:n_us4oems*32] = True
            rx_aperture[0] = True
            seq = TxRxSequence(
                ops=[
                    TxRx(
                        Tx(aperture=tx_aperture,
                           excitation=Pulse(center_frequency=tx_freq, n_periods=tx_ncycles, inverse=False, amplitude_level=amplitude_level),
                           delays=[0]),
                        Rx(aperture=rx_aperture, sample_range=(0, 4096), downsampling_factor=1),
                        pri=200e-6
                    ),
                ],
                tgc_curve=[],  # [dB]
                sri=200e-3
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
    measurements = np.stack(measurements)
    output_file = "hvps_measurements/hvps_measurements_666.npy"
    np.save(output_file, measurements)
    print(f"Saved data to {output_file}")



if __name__ == "__main__":
    main()

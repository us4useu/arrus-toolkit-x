import numpy as np

# Input parameters
tx_frequency = 5e6
push_pulse_length = 100e-6  # [s]
fs = 65e6

# Calc settings: PRI
tx_pri = np.max([100e-6, push_pulse_length+20e-6])
print("TX PRI=")
print(tx_pri)

# Calc settings: n_samples
s = (push_pulse_length * fs / 2)
s = int(s/64)
s = s * 64
n_samples = np.max([1024, s])

print("n_samples=")
print(n_samples)




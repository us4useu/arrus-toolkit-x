import numpy as np
import scipy as sp
import cupy as cp
import cupyx.scipy as cx
import matplotlib.pyplot as plt
import pickle

import arrus
import arrus.session
import arrus.utils.imaging
import arrus.utils.us4r
import arrus.kernels
import arrus.kernels.kernel

from arrus.metadata import *
from arrus.ops.imaging import (
    PwiSequence
)


from arrus.devices.probe import ProbeDTO, ProbeModel, ProbeModelId
from arrus.ops.us4r import *
from arrus.devices.probe import *
from arrus.devices.us4r import Us4RDTO
from arrus.utils.imaging import *

from scipy.signal import firwin, butter, buttord, freqz


## Probe definition ############################################
PROBE_MODEL = ProbeModel(
    model_id=ProbeModelId(manufacturer="atl/philips", name="l7-4"),
    n_elements=128,
    pitch=0.298e-3,
    curvature_radius=0
)

## Utilities ###################################################
def DrawShearWaveFrames(data, Vrange, frames):
    data = np.clip(data, Vrange[0], Vrange[1])
    norm = plt.Normalize(Vrange[0], Vrange[1], True)
    ax = [] * len(frames)
    fig, ax = plt.subplots(1, 6, figsize=(14, 2))
    #frames = [1, 4, 7, 10, 13, 16]
    for i in range(len(frames)):
        ax[i].imshow(data[:, :, frames[i]], cmap='jet', norm=norm)
        ax[i].set_title('Frame: ' + str(frames[i]))
    return  

def getHistogram(data, Vrange):
    #%matplotlib widget
    data= data.flatten();
    plt.hist(data, bins=128, range=Vrange)
    

def plotSWSmap(data, px_size=0.1, tick_grid_size=[5, 5], sws_disp_range=[0, 5]):
    
    # Normalize values
    norm_sws = plt.Normalize(sws_disp_range[0], sws_disp_range[1], True)
    plt.figure()
    plt.clf()
    plt.imshow(np.squeeze(data), cmap='jet', norm=norm_sws)
    plt.title('SWS [m/s]')
    plt.colorbar(label='SWS [m/s]')

    # Assign ticks
    sws_dim = data.shape
    y_grid = tick_grid_size[1]  #[mm]
    x_grid = tick_grid_size[0]  #[mm]

    yticks  = np.arange(0, sws_dim[0]-1, int(np.ceil(y_grid / px_size))) 
    yticks_labels = yticks * px_size
    yticks_labels = [str(int(x)) for x in yticks_labels]
    plt.yticks(yticks, yticks_labels)

    m = int(sws_dim[1]//2)
    a = np.arange(m, 0, -int(np.ceil(x_grid / px_size)))
    b = np.arange(m, sws_dim[1]-1, int(np.ceil(x_grid / px_size)))
    xticks = np.concatenate((a[1:], b))
    xticks_labels = (xticks - m) * px_size
    xticks_labels = [str(int(x)) for x in xticks_labels]
    plt.xticks(xticks, xticks_labels)

    plt.xlabel("x [mm]")
    plt.ylabel("z [mm]")

    plt.show()    
    
    
# TXPB Adapter Class ##############################################
class TxpbAdapter(arrus.utils.imaging.Operation):

    def __init__(self, n_push, n_pwi, n_samples, angles, center_frequency, n_periods, speed_of_sound):
        self.n_push = n_push
        self.n_pwi = n_pwi
        self.n_samples = n_samples
        self.angles = angles
        self.pulse = Pulse(center_frequency=center_frequency, n_periods=n_periods,
                           inverse=False)
        self.speed_of_sound = speed_of_sound
        self.output = None

    def prepare(self, const_metadata):
        N_CHANNELS = 128
        FS = 65e6
        self.output = cp.zeros((1, self.n_pwi, self.n_samples, N_CHANNELS), dtype=np.int16)
        # Modify metadata structures so they are corresponding to the
        # the specific of the data produced by txpb.
        seq = PwiSequence(
            angles=self.angles,
            pulse=self.pulse,
            downsampling_factor=1,
            speed_of_sound=self.speed_of_sound,
            rx_sample_range=(0, self.n_samples),
            pri=100e-6 # This value is not used by the imaging code.
        )
        #print(seq)
        probe = ProbeDTO(PROBE_MODEL)
        device = Us4RDTO(probe=probe, sampling_frequency=FS)
        kernel_context = arrus.kernels.kernel.KernelExecutionContext(
            device=device, medium=None, op=seq, custom={})
        raw_seq = arrus.kernels.get_kernel(type(seq))(kernel_context)
        #print(device)
        context = FrameAcquisitionContext(
            device,
            sequence=seq,
            raw_sequence=raw_seq,
            medium=None,
            custom_data={}
        )
        data_description=EchoDataDescription(sampling_frequency=FS,
                                             custom={})
        out_metadata = const_metadata.copy(input_shape=self.output.shape,
                                   context=context,
                                   data_desc=data_description,
                                   is_iq_data=False,
                                   dtype=np.int16)
        pickle.dump(out_metadata, open("out_metadata.pkl", "wb"))
        return out_metadata

    def process(self, data):
        module_n_samples = self.n_samples*self.n_push+self.n_samples*self.n_pwi*2
        us4oem0_data = data[:module_n_samples, :]
        us4oem1_data = data[module_n_samples:, :]

        # Strip out push data
        us4oem0_data = us4oem0_data[self.n_push*self.n_samples:, :]
        us4oem0_data = us4oem0_data.reshape((self.n_pwi, 2, self.n_samples, 32))
        us4oem1_data = us4oem1_data[self.n_push*self.n_samples:, :]
        us4oem1_data = us4oem1_data.reshape((self.n_pwi, 2, self.n_samples, 32))

        for i in range(self.n_pwi):
            self.output[0, i, :,   0:32] = us4oem0_data[i, 0, :, :]
            self.output[0, i, :,  32:64] = us4oem1_data[i, 0, :, ::-1]
            self.output[0, i, :,  64:96] = us4oem0_data[i, 1, :, :]
            self.output[0, i, :, 96:128] = us4oem1_data[i, 1, :, ::-1]
            # swap data centers of the second module
            self.output[0, i, :, [15, 16]] = self.output[0, i, :, [16, 15]]
            self.output[0, i, :, [47, 48]] = self.output[0, i, :, [48, 47]]
            self.output[0, i, :, [79, 80]] = self.output[0, i, :, [80, 79]]
            self.output[0, i, :, [111, 112]] = self.output[0, i, :, [112, 111]]
        return self.output

def compute_linear_tgc(tgc_start, tgc_slope, sample_range, c):
    start_sample, end_sample = sample_range
    fs = 65e6

    distance = np.arange(start=400,
                         stop=end_sample,
                         step=150)/fs*c
    # czas propagacji fali
    # tgc_slope [db/(m*Hz)]
    return tgc_start + distance*tgc_slope
    
    
# Operation classes ###############################################  
    
class AngleCompounding(Operation):
    # Expected input data format: 3D array [frame, x, z]
    # Output data format: 3D array [frame, x, z]
    
    def __init__(self, nAngles, num_pkg=None, filter_pkg=None, **kwargs):
        self.nAngles = nAngles
        
        self.xp = num_pkg
        self.filter_pkg = filter_pkg
        self.kwargs = kwargs
        
    def set_pkgs(self, num_pkg, filter_pkg, **kwargs):
        self.xp = num_pkg
        self.filter_pkg = filter_pkg        
    
    def prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        # Create a kernel for the convolution
        self.kernel  = cp.ones(self.nAngles) / self.nAngles
        
        # Metadata
        input_shape = np.asarray(const_metadata.input_shape)
        output_shape = input_shape - np.array([2*self.nAngles-2, 0, 0])
        
        return const_metadata.copy(input_shape=tuple(output_shape))
        
    def process(self, data): 
        # Perform the convolution 
        data = cx.ndimage.convolve1d(data, self.kernel, axis=0)
        # Crop the output data
        data = data[self.nAngles-1:-self.nAngles+1,:, :]
        return data    
    
    
class GenerateBmode():
    
    def __init__(self):
        pass
    
    def prepare(self):
        pass    
    
    def plotHistogram(self, data):
        data_cpu = data.get()
        data_cpu = np.abs(data_cpu).flatten()
        plt.hist(data_cpu, bins=128)
        return
    
    def displayBmode(self, frame, dB_range):
        frame = frame.get()
        frame = np.squeeze(frame)
        norm = plt.Normalize(dB_range[0], dB_range[1], True)
        frame_dim = frame.shape
        if(frame_dim[1] > frame_dim[0] ):
            frame = np.transpose(frame, [1,0])

        fig, ax0 = plt.subplots(1, 1, figsize=(8, 4))
        ax0.imshow(np.abs(frame), cmap='gray', norm=norm)
        ax0.set_title('Bmode [dB]')
        return

    def process(self, data):
        data = cp.squeeze(data)
        # Envelope detection
        data = cp.abs(data)
        # Log compression
        data[data==0] = 10**-10
        data = 20 * cp.log10(data)
        return data    

class ShearwaveDetection(Operation):
    # Expected input data format: 3D array [frame, x, z]
    # Output data format: 3D array [z, x, frame]
    def __init__(self, mode='kasai', packet_size=4, z_gate=4, fc=4.4e6, num_pkg=None, filter_pkg=None, **kwargs):
        # Capture params
        self.packet_size = packet_size
        self.mode = mode
        self.z_gate = z_gate 
        self.fc = fc
        self.xp = num_pkg
        self.filter_pkg = filter_pkg
        self.kwargs = kwargs 
        
    def set_pkgs(self, num_pkg, filter_pkg, **kwargs):
        self.xp = num_pkg
        self.filter_pkg = filter_pkg            
    
    def prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        #self.fc = _get_unique_pulse(const_metadata.context.sequence).center_frequency
        self.c  = const_metadata.context.sequence.speed_of_sound
        self.frame_pri = const_metadata.context.sequence.pri * 2
        self.fs = const_metadata.data_description.sampling_frequency
        
        # Metadata
        input_shape  = const_metadata.input_shape
        output_shape = (input_shape[2]-1, input_shape[1], input_shape[0]-1)
        
        return const_metadata.copy(input_shape=output_shape, dtype=cp.float32)

    def process(self, data):   
        # Rearrange data
        data = cp.moveaxis(data, [0,1], [2,1])
        
        # Shear wave detection
        P0 = data[:-1,:,:-1] * cp.conj(data[:-1,:,1:])
        P0 = cx.ndimage.convolve1d(P0, cp.ones(self.packet_size), axis=2)
        P0 = cx.ndimage.convolve1d(P0, cp.ones((self.z_gate)), axis=0)
        f_d = cp.angle(P0) / (2*cp.pi)
        
        if(self.mode=='loupas'):
            P1 = data[:-1,:,:] * cp.conj(data[1:,:,:])
            P1 = cx.ndimage.convolve1d(P1, cp.ones(self.packet_size), axis=2)
            P1 = cx.ndimage.convolve1d(P1, cp.ones((self.z_gate)), axis=0)
            # Estimated centeral frequency map
            fc_shift = cp.abs(cp.angle(P1) / (2*cp.pi/self.fs))
            fc_shift = fc_shift[:, :, 1:]
            fc_shift[fc_shift == 0] = 1e-9
            # Displacement
            ddata = self.c * f_d / (2*fc_shift*self.frame_pri)
        else:    
            ddata = self.c * f_d / (2*self.fc*self.frame_pri )  
        
        return ddata
            

class ShearwaveMotionDataFiltering(Operation):
    # Expected data format:  3D array [z, x, frame], float32
    # Output data format: 2x 3D array [2, z, x, frame]
    def __init__(self, sws_range, f_range, k_range, num_pkg=None, filter_pkg=None, **kwargs):
        self.sws_range = sws_range
        self.f_range   = f_range
        self.k_range   = k_range
        
        self.xp = num_pkg
        self.filter_pkg = filter_pkg
        self.kwargs = kwargs 

    def set_pkgs(self, num_pkg, filter_pkg, **kwargs):
        self.xp = num_pkg
        self.filter_pkg = filter_pkg    
        
    def prepare(self, const_metadata: arrus.metadata.ConstMetadata):
     
        self.fs = 1/(const_metadata.context.sequence.pri * 2)
        
        # Get input data shape
        self.data_shape = const_metadata.input_shape
         
        # Obtain buffer sizes
        def nextpow2(x):
            return int(np.ceil(np.log2(x)))
        
        self.nZ, self.nX, self.nFrames = self.data_shape
        nX_padded = 2**nextpow2(self.nX)
        nFrames_padded = 2**nextpow2(self.nFrames)
        self.kX = [nX_padded, nFrames_padded]
        
        ## Create masks for filtering
        # Basic directional mask
        mask = np.ones(self.kX)
        mask[:int(np.floor(self.kX[0]/2)), :int(np.floor(self.kX[1]/2))] = 0
        mask[int(np.floor(self.kX[0]/2)):, int(np.floor(self.kX[1]/2)):] = 0

        # Add filtering ranges
        vmin = self.sws_range[0]
        vmax = self.sws_range[1]

        # Precalc values
        w_s_max = np.ceil(self.f_range[1] * self.kX[1] / self.fs)
        w_s_min = np.ceil(self.f_range[0] * self.kX[1] / self.fs)
        k_s_max = np.ceil(self.k_range * self.kX[0]/2)

        # Experimantal filter functions parameters
        a_w1 = 2;
        a_w2 = 0.2;
        a_k  = 0.1;
        a_v1 = 10;
        a_v2 = 4; 

        mask_f = mask;
        for w in range(self.kX[1]):
            for k in range(self.kX[0]):
                w_s = np.abs(w - self.kX[1] / 2)
                k_s = np.abs(k - self.kX[0] / 2)
                if(k_s == 0):
                    k_s = 1e-6;

                v = np.abs(w_s / k_s)

                # Calc filter coefficients
                fw1 = 1 / (1 + np.exp(-a_w1 * (w_s - w_s_min)))
                fw2 = 1 - 1 / (1 + np.exp(-a_w2 * (w_s - w_s_max)))
                fw = min(fw1, fw2)

                fk = 1 - 1 / (1 + np.exp(-a_k * (k_s - k_s_max)))

                fv1 = 1 / (1 + np.exp(-a_v1 * (v - vmin)))
                fv2 = 1 - 1 / (1 + np.exp(-a_v2 * (v - vmax)))
                fv = min(fv1, fv2)

                mask_f[k, w] = mask_f[k, w] * fw * fk * fv

        mask_f[int(np.floor(self.kX[0]/2)), int(np.floor(self.kX[1]/2))] = 0;
        self.mask_RL = cp.asarray(mask_f)
        self.mask_LR = cp.asarray(np.flipud(mask_f))
        
        #self.plotFilterMasks()
        
        ## Metadata -update output dimensions
        input_shape  = const_metadata.input_shape
        output_shape = (2, input_shape[0], input_shape[1], input_shape[2])
        
        return const_metadata.copy(input_shape=output_shape)
    
    def process(self, data):
        # Perform the 2-D FFT
        X = cx.fft.fft2(data, s=self.kX, axes=(1, 2), overwrite_x=True)
        X = cx.fft.fftshift(X)        
        
        # Filtering in k-omega space
        X1 = X * self.mask_RL
        X2 = X * self.mask_LR
        # Perform the Inverse 2-D FFT
        X1 = cx.fft.fftshift(X1)
        X2 = cx.fft.fftshift(X2)
        X1 = cp.real(cx.fft.ifft2(X1, axes=(1, 2), overwrite_x=True))
        X2 = cp.real(cx.fft.ifft2(X2, axes=(1, 2), overwrite_x=True))
        X1 = X1[:, 0:self.nX, 0:self.nFrames] # L->R
        X2 = X2[:, 0:self.nX, 0:self.nFrames] # R->L
        
        # Pack data
        X1 = X1[np.newaxis, ...]
        X2 = X2[np.newaxis, ...]
        X1 = cp.real(X1)
        X2 = cp.real(X2)
        
        # Debug memory
        #mempool = cp.get_default_memory_pool()
        #pinned_mempool = cp.get_default_pinned_memory_pool()    
        #print("DIRFILT:Mempool used bytes [MB]: {:f}".format(mempool.used_bytes()/1024/1024))
        #print("DIRFILT:Mempool total bytes [MB]: {:f}".format(mempool.total_bytes()/1024/1024))

        
        return cp.concatenate((X1, X2), axis=0)
        
        
    def plotFilterMasks(self):
        mask_RL_cpu = self.mask_RL.get()
        mask_LR_cpu = self.mask_LR.get()
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7, 3))
        ax0.imshow(np.squeeze(mask_RL_cpu),   cmap='gray', aspect=0.5)
        ax1.imshow(np.squeeze(mask_LR_cpu),   cmap='gray', aspect=0.5)
        return
     
        
              
class SWS_Estimation(Operation):
    # Expected data format: 2x 3D array [2, z, x, frame]
    # Output format: 4D data: 2x 3D array: SWS: 3D array [dataset, z, x], SWS_r: 3D array [dataset, z, x]. Arrays concat'd along axis=0
    # For example: [SWV/r, LR/RL, z, x]

    def __init__(self, x_range, z_clip, frames_range, d, fri, interp_factor=5, interp_order=2, px_pitch=0.1e-3, sws_range=[0.5, 4.0],
                num_pkg=None, filter_pkg=None, **kwargs):
        # Params capture
        self.x_range = x_range # for example [[0, 200], [250, 400]], so x_ranges for LR and RL datasets: [[xL_LR, xR_LR], [xL_RL, xR_RL]]
        self.frames_range = frames_range # for example [0, 99]
        self.z_clip = z_clip # for example [10, 30] - how many pixels to clip from top and bottom
        self.d = d
        self.interp_factor = interp_factor
        self.interp_order = interp_order
        self.px_pitch = px_pitch
        self.sws_range = sws_range
        self.FRI = fri
        
        self.xp = num_pkg
        self.filter_pkg = filter_pkg
        self.kwargs = kwargs 
    
    def set_pkgs(self, num_pkg, filter_pkg, **kwargs):
        self.xp = num_pkg
        self.filter_pkg = filter_pkg       
    
    def prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        
        # Get input data shape
        data_shape = const_metadata.input_shape
        self.out_dim = [data_shape[1], data_shape[2], data_shape[3]] # of one sub-dataset
        
        # Process column ranges 
        self.xL = [0, 0]
        self.xR = [0, 0]       
        
        # LR
        self.xL[0] = self.x_range[0][0] - int(np.ceil(self.d/2)) 
        self.xR[0] = self.x_range[0][1] + int(np.ceil(self.d/2)) 
        if(self.xL[0] < 0):
            self.xL[0] = 0;

        if(self.xR[0] > self.out_dim[1]):
            self.xR[0] = self.out_dim[1] 
            
        # RL
        self.xL[1] = self.x_range[1][0] - int(np.ceil(self.d/2)) 
        self.xR[1] = self.x_range[1][1] + int(np.ceil(self.d/2)) 
        if(self.xL[1] < 0):
            self.xL[1] = 0;

        if(self.xR[1] > self.out_dim[1]):
            self.xR[1] = self.out_dim[1]             
            
        # Get window function
        N = (self.frames_range[1] - self.frames_range[0]) * self.interp_factor
        w = sp.signal.windows.tukey(N, alpha=0.2, sym=True)
        self.w = cp.asarray(w)
        
        # Other
        self.ds = self.px_pitch * self.d
        
        ## Metadata -update output dimensions
        input_shape  = const_metadata.input_shape
        output_shape = (2, 2, input_shape[1], input_shape[2])
        
        return const_metadata.copy(input_shape=output_shape)
    
    
    def process(self, data):
        
        # Create output buffer
        SWV = cp.zeros((2, 2, self.out_dim[0], self.out_dim[1]))  #[SWV/r, LR/RL,  z, x]
        
        # Process each sub-dataset (LR, and RL independently)
        for i in range(2):
            # Unpack data
            ddata = cp.squeeze(data[i])
        
            # Crop data
            ddata = ddata[self.z_clip[0]:-self.z_clip[1], self.xL[i]:self.xR[i], self.frames_range[0]:self.frames_range[1]]
            ddata[cp.isnan(ddata)]=0
        
            # Data interpolation along slow-time dimension
            ynew = cx.ndimage.zoom(input=ddata, zoom=(1, 1, self.interp_factor), output=data.dtype, order=self.interp_order)
        
            # DC offset cancellation
            m = cp.mean(ynew, axis=2)
            ynew = cp.transpose(ynew, [2,0,1])
            ynew = cp.subtract(ynew, m)
            ynew = cp.transpose(ynew, [1,2,0])
        
            # Apply window function
            ynew = ynew * self.w

            # Find normalization values
            txa = ynew[:, 0:ynew.shape[1]-self.d, :]
            txb = ynew[:, self.d:ynew.shape[1], :]
            Rxa = cp.sum(txa * txa, axis=2)
            Rxb = cp.sum(txb * txb, axis=2)

            # Compute the FFT along the slow-time dim
            X = cx.fft.rfft(ynew, axis=2, overwrite_x=True)

            # Correlation in frequency domain and back to time-domain
            Xa = X[:, 0:X.shape[1]-self.d, :]
            Xb = X[:, self.d:X.shape[1], :]
            c = cp.real(cx.fft.irfft(Xa * cp.conj(Xb), axis=2, overwrite_x=True))

            # Shift zero
            c = cp.concatenate((c[:, :, c.shape[2]//2:], c[:, :, :c.shape[2]//2]), axis=2)

            # Normalize the correlation values
            c = cp.transpose(c,[2, 0, 1])
            c = c / (cp.sqrt(Rxa) * cp.sqrt(Rxb))
            c = cp.transpose(c, [1,2,0])

            ## SWS estimation post-processing
            # find lag of max correlation
            r = cp.amax(c, axis=2)
            rmax_idx = cp.argmax(c, axis=2)

            # Zero shift
            rmax_idx = cp.abs(rmax_idx - c.shape[2]//2)
            rmax_idx[rmax_idx==0] = 1 

            # Calc SWS
            dt = self.FRI * rmax_idx / self.interp_factor
            sws_map = self.ds / dt

            # Limit the values
            sws_map = cp.clip(sws_map, self.sws_range[0], self.sws_range[1]) 
            
            # Aggregate the results
            SWV[0, i, self.z_clip[0]:-self.z_clip[1], int(cp.ceil(self.xL[i] + self.d/2)) : int(cp.ceil(self.xR[i]-self.d/2))] = sws_map
            SWV[1, i, self.z_clip[0]:-self.z_clip[1], int(cp.ceil(self.xL[i] + self.d/2)) : int(cp.ceil(self.xR[i]-self.d/2))] = r
            
            # Debug memory
            #mempool = cp.get_default_memory_pool()
            #pinned_mempool = cp.get_default_pinned_memory_pool()    
            #print("SWS:Mempool used bytes [MB]: {:f}".format(mempool.used_bytes()/1024/1024))
            #print("SWS:Mempool total bytes [MB]: {:f}".format(mempool.total_bytes()/1024/1024))

        return SWV    
        
                 
class SWS_Compounding(Operation):
    # Input data format: 4D data: 2x 3D array: SWS: 3D array [dataset, z, x], SWS_r: 3D array [dataset, z, x]. Arrays concat'd along axis=0
    # For example: [SWV/r, LR/RL, z, x]
    # Output data format: 3D data: 2x 2D array: SWS: 3D array [z, x], SWS_r: 3D array [z, x]. Arrays concat'd along axis=0
    # For example: [SWV/r, z, x]
    def __init__(self, num_pkg=None, filter_pkg=None, **kwargs):
        self.xp = num_pkg
        self.filter_pkg = filter_pkg
        self.kwargs = kwargs 
    
    def prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        ## Metadata -update output dimensions
        input_shape  = const_metadata.input_shape
        output_shape = (2, input_shape[2], input_shape[3])
        return const_metadata.copy(input_shape=output_shape)
    
    def set_pkgs(self, num_pkg, filter_pkg, **kwargs):
        self.xp = num_pkg
        self.filter_pkg = filter_pkg       
    
    def process(self, data):
        # Unpack data
        SWS   = cp.squeeze(data[0])
        SWS_r = cp.squeeze(data[1])
        
        # Weighted average
        SWScr = cp.squeeze(cp.sum(SWS_r, axis=0))
        SWScr[SWScr==0] = 10e-6
        SWSc = SWS * SWS_r / SWScr
        SWSc = cp.sum(SWSc, axis=0)
 
        # Pack data
        SWSc  = SWSc[np.newaxis, ...]
        SWScr = SWScr[np.newaxis, ...]
        return np.concatenate((SWSc, SWScr), axis=0) 
    
    
class MedianFiltering(Operation):
    # Expected data format: 3D data: 2x 2D array: SWS: 3D array [z, x], SWS_r: 3D array [z, x]. Arrays concat'd along axis=0
    def __init__(self, kernel_size, num_pkg=None, filter_pkg=None, **kwargs):
        self.kernel_size=kernel_size
        self.xp = num_pkg
        self.filter_pkg = filter_pkg
        self.kwargs = kwargs 
        
    def set_pkgs(self, num_pkg, filter_pkg, **kwargs):
        self.xp = num_pkg
        self.filter_pkg = filter_pkg            
    
    def prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        return const_metadata.copy() 
    
    def process(self, data):
        # Unpack data
        image = cp.squeeze(data[0])
        #Filter image
        image = cx.ndimage.median_filter(image, size=(self.kernel_size, self.kernel_size))
        # Pack data
        data[0, :, :] = image
        return data
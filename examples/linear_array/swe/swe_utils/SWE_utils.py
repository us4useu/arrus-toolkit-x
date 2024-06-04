import numpy as np
import scipy as sp
import cupy as cp
import cupyx.scipy as cx
import matplotlib.pyplot as plt

from scipy.signal import firwin, butter, buttord, freqz

  
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
    
    
# Operation classes ##########################################333    
    
class AngleCompounding():
    
    def __init__(self, nAngles, axis):
        self.nAngles = nAngles
        self.axis    = axis
        
    def prepare(self):
        # Create a kernel for the convolution
        self.kernel  = np.ones(self.nAngles) / self.nAngles
        return
        
    def process(self, data): 
        # Perform the convolution 
        data = sp.ndimage.convolve1d(data, self.kernel, axis=self.axis)
        # Crop the output data
        data = data[self.nAngles-1:-self.nAngles+1,:, :]
        return data
    
    def prepare_cp(self, nAngles):
        # Create a kernel for the convolution
        self.nAngles = nAngles
        self.kernel  = cp.ones(self.nAngles) / self.nAngles
        return
        
    def process_cp(self, data, axis): 
        # Perform the convolution 
        data = cx.ndimage.convolve1d(data, self.kernel, axis=axis)
        # Crop the output data
        data = data[self.nAngles-1:-self.nAngles+1,:, :]
        return data    
    
    
class GenerateBmode():
    
    def __init__(self):
        pass
    
    def prepare(self):
        pass    
    
    def plotHistogram(self, data):
        #%matplotlib widget
        data = np.abs(data).flatten()
        plt.hist(data, bins=128)
        return
    
    def displayBmode(self, frame, dB_range):
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
        data = np.squeeze(data)
        # Envelope detection
        data = np.abs(data)
        # Log compression
        data[data==0] = 10**-10
        data = 20 * np.log10(data)
        return data    
    
class ShearwaveDetection():
    # Expected data format: 3D array [z, x, frame]
    def __init__(self, mode='kasai', packet_size=4, z_gate=8, frame_pri=200e-3, c=1540, fc=4.4e6, fs=65e6):
        # Capture params
        self.packet_size = packet_size
        self.mode = mode
        self.z_gate = z_gate 
        self.frame_pri = frame_pri
        self.c = c
        self.fc = fc
        self.fs = fs
    
    def prepare(self):
        pass 
    
    def process(self, data):   
        P0 = data[:-1,:,:-1] * np.conj(data[:-1,:,1:])
        P0 = sp.ndimage.convolve1d(P0, np.ones(self.packet_size), axis=2)
        P0 = sp.ndimage.convolve1d(P0, np.ones((self.z_gate)), axis=0)
        f_d = np.angle(P0) / (2*np.pi)
        
        if(self.mode=='loupas'):
            P1 = data[:-1,:,:] * np.conj(data[1:,:,:])
            P1 = sp.ndimage.convolve1d(P1, np.ones(self.packet_size), axis=2)
            P1 = sp.ndimage.convolve1d(P1, np.ones((self.z_gate)), axis=0)
            # Estimated centeral frequency map
            fc_shift = np.abs(np.angle(P1) / (2*np.pi/self.fs))
            fc_shift = fc_shift[:, :, 1:]
            fc_shift[fc_shift == 0] = 1e-9
            # Displacement
            ddata = self.c * f_d / (2*fc_shift*self.frame_pri)
        else:    
            ddata = self.c * f_d / (2*self.fc*self.frame_pri )  
        
        return ddata
            

class ShearwaveMotionDataFiltering():
    # Expected data format: 3D array [z, x, frame]
    def __init__(self, data_shape, sws_range, f_range, k_range, fs):
        self.data_shape=data_shape
        self.sws_range = sws_range
        self.f_range   = f_range
        self.k_range   = k_range  
        self.fs        = fs     
    
    def prepare(self):
     
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
        #mask_RLraw = np.copy(mask)
        mask_RLraw = mask

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
                w_s = abs(w - self.kX[1] / 2)
                k_s = abs(k - self.kX[0] / 2)
                if(k_s == 0):
                    k_s = 1e-6;

                v = abs(w_s / k_s)

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
        self.mask_RL = mask_f
        self.mask_LR = np.flipud(mask_f);
        return  
    
    def process(self, data):
        # Perform zero padding of the input data
        #ddata_padded = np.pad(data, ((0,0), (0, self.kX[0] - self.nX), (0, self.kX[1] - self.nFrames)), 'constant')
        # Perform the 2-D FFT
        X = sp.fft.fft2(data, s=self.kX, axes=(1, 2), overwrite_x=False)
        X = sp.fft.fftshift(X)
        # Filtering in k-omega space
        X1 = X * self.mask_RL
        X2 = X * self.mask_LR
        # Perform the Inverse 2-D FFT
        X1 = sp.fft.fftshift(X1)
        X2 = sp.fft.fftshift(X2)
        X1 = np.real(sp.fft.ifft2(X1, axes=(1, 2)))
        X2 = np.real(sp.fft.ifft2(X2, axes=(1, 2)))
        X1 = X1[:, 0:self.nX, 0:self.nFrames] # L->R
        X2 = X2[:, 0:self.nX, 0:self.nFrames] # R->L
        return (X1, X2)
        
    def plotFilterMasks(self):
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7, 3))
        ax0.imshow(np.squeeze(self.mask_RL),   cmap='gray', aspect=0.5)
        ax1.imshow(np.squeeze(self.mask_LR),   cmap='gray', aspect=0.5)
        return
        
                
            
class SWS_Estimation():
    # Expected data format: 3D array [z, x, frame]
    def __init__(self, data_shape, x_range, d, interp_factor, interp_type='cubic', px_pitch=0.1e-3, FRI =200e-6, sws_range=[0.5, 4.0]):
        # Params capture
        self.out_dim = [data_shape[0], data_shape[1], data_shape[2]]
        self.x_range = x_range
        self.d = d
        self.interp_factor = interp_factor
        self.interp_type = interp_type
        self.px_pitch = px_pitch
        self.FRI = FRI
        self.sws_range = sws_range
    
    def prepare(self):
               
        # Process column range
        self.xL = self.x_range[0] - int(np.ceil(self.d/2));
        self.xR = self.x_range[1] + int(np.ceil(self.d/2));
        if(self.xL < 0):
            self.xL = 0;

        if(self.xR > self.out_dim[1]):
            self.xR = self.out_dim[1] 
            
        # Get window function
        N = (self.out_dim[2]-1) * self.interp_factor
        self.w = sp.signal.windows.tukey(N, alpha=0.2, sym=True)
        
        # Other
        self.ds = self.px_pitch * self.d
        
        return  
    
    def process(self, data):
        # Crop data
        data = data[:, self.xL:self.xR, :]
        
        # Data interpolation along slow-time dimension
        x = np.arange(0, data.shape[2], 1)
        data[np.isnan(data)]=0
        f = sp.interpolate.interp1d(x=x, y=data, kind=self.interp_type, axis=2)
        xnew = np.arange(0, data.shape[2]-1, 1/self.interp_factor)
        ynew = f(xnew) 
        
        # DC offset cancellation
        m = np.mean(ynew, axis=2)
        ynew = np.transpose(ynew, [2,0,1])
        ynew = np.subtract(ynew, m)
        ynew = np.transpose(ynew, [1,2,0])
        
        # Apply window function
        ynew = ynew * self.w
        
        # Compute the FFT along the slow-time dim
        X = sp.fft.fft(ynew, axis=2)

        # Correlation in frequency domain and back to time-domain
        Xa = X[:, 0:X.shape[1]-self.d, :]
        Xb = X[:, self.d:X.shape[1], :]
        c = np.real(sp.fft.ifft(Xa * np.conj(Xb), axis=2))

        # Shift zero
        c = np.concatenate((c[:, :, c.shape[2]//2:], c[:, :, :c.shape[2]//2]), axis=2)

        # Normalize the correlation values
        txa = ynew[:, 0:ynew.shape[1]-self.d, :]
        txb = ynew[:, self.d:ynew.shape[1], :]
        Rxa = np.sum(txa * txa, axis=2)
        Rxb = np.sum(txb * txb, axis=2)
        c = np.transpose(c,[2, 0, 1])
        c = c / (np.sqrt(Rxa) * np.sqrt(Rxb))
        c = np.transpose(c, [1,2,0])
        
        ## SWS estimation post-processing
        # find lag of max correlation
        r = np.amax(c, axis=2)
        rmax_idx = np.argmax(c, axis=2)

        # Zero shift
        rmax_idx = np.abs(rmax_idx - c.shape[2]//2)
        rmax_idx[rmax_idx==0] = 1 

        # Calc SWS
        dt = self.FRI * rmax_idx / self.interp_factor
        sws_map = self.ds / dt

        # Assign results
        SWV   = np.zeros(self.out_dim[:-1])
        SWV_r = np.zeros(self.out_dim[:-1])
        SWV[:, int(np.ceil(self.xL + self.d/2)) : int(np.ceil(self.xR-self.d/2))] = sws_map
        SWV_r[:, int(np.ceil(self.xL + self.d/2)) : int(np.ceil(self.xR-self.d/2))] = r

        # Limit the values
        SWV = np.clip(SWV, self.sws_range[0], self.sws_range[1])  
        
        return (SWV, SWV_r)
        
        
class SWS_Compounding():
    # Expected data format: 4D data: 2x 3D array: SWS: 3D array [dataset, z, x], SWS_r: 3D array [dataset, z, x]. Arrays concat'd along axis=0
    def __init__(self):
        pass
    
    def prepare(self):
        pass  
    
    def process(self, data):
        # Unpack data
        SWS   = np.squeeze(data[0])
        SWS_r = np.squeeze(data[1])
        # Weighted average
        SWScr = np.squeeze(np.sum(SWS_r, axis=0))
        SWScr[SWScr==0] = 10e-6
        SWSc = SWS * SWS_r / SWScr
        SWSc = np.sum(SWSc, axis=0)
        return (SWSc, SWScr)  
    
    
class MedianFiltering():
    # Expected data format: 2D data array: [z, x]
    def __init__(self, kernel_size):
        self.kernel_size=kernel_size
    
    def prepare(self):
        pass  
    
    def process(self, data):
        return sp.ndimage.median_filter(data, size=(self.kernel_size, self.kernel_size))   
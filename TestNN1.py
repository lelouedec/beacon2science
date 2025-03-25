import torch
from models.ESRGAN import *
import numpy as np 
import torchvision
torchvision.disable_beta_transforms_warning()
import models.unet2 as unet2
from natsort import natsorted 
import glob 


from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import mean_squared_error as MSE
from skimage.transform import resize
import pickle



import tqdm

import os


def normalize(img,rangev=2.5):
    
    vmax = np.median(img)+rangev*np.std(img)
    vmin = np.median(img)-rangev*np.std(img)

    img[img>vmax] = vmax
    img[img<vmin] = vmin

    img = (img-vmin)/( (vmax-vmin) + 1e-16)

    img[img>1.0] = 1.0
    return img


device = torch.device("cpu")
if(torch.backends.mps.is_available()):
    device = torch.device("mps")
elif(torch.cuda.is_available()):
    device = torch.device("cuda:0")


model2 = unet2.ResUnet(1,full_size=512)
# dict_gen  = torch.load("gan_gen_l14.pth",map_location=torch.device('cpu'))
dict_gen  = torch.load("PAPER_NN1.pth",map_location=torch.device('cpu'))

model2.load_state_dict(dict_gen)
model2.to(device)

imgs_paths = natsorted(glob.glob("/media/lelouedec/DATA/rdifs_beacon/*"))
print(len(imgs_paths),imgs_paths[0])

save_path = "results_enhanced_test"

cnt = 0
mse = 0
mse_beacon= 0
with torch.no_grad():
    for p in tqdm.tqdm(range(0,len(imgs_paths),1)):

        
        with open(imgs_paths[p], 'rb') as f:
            jplot_dict = pickle.load(f)
            data1 = jplot_dict['data']
            hdr = jplot_dict['header']
                
        data1 = np.nan_to_num(data1,np.nanmedian(data1)) 
        D2 = normalize(data1)
        D2_original = D2.copy()
        D2  =  resize(D2_original, (256,256),anti_aliasing=True,order=3,preserve_range=True)
        D2_tensor = torch.tensor(D2.copy()).float().unsqueeze(0).unsqueeze(1).to(device)

        sr = model2(D2_tensor,None)
        sr = sr[0,0,:,:].cpu().numpy()
        cnt+=1
        to_save = {'data': sr,
                'header':hdr
                    }
        pickle.dump(to_save, open("/media/lelouedec/DATA/E-beacon/"+imgs_paths[p].split("/")[-1][:-2]+".p", 'wb'))
        
       

imgs_paths = natsorted(glob.glob("/media/lelouedec/DATA/rdifs_beacon/*"))
print(len(imgs_paths),imgs_paths[0])

cnt = 0
PSNR_beacon = []
PSNR_enhanced = []
MSE_beacon = []
MSE_enhanced = []
SSIM_beacon = []
SSIM_enhanced = []
contrast = 5

magnitudes_science = []
magnitudes_beacon = []
magnitudes_enhanced = []


with torch.no_grad():
    for p in tqdm.tqdm(range(0,len(imgs_paths),1)):
        with open(imgs_paths[p], 'rb') as f:
            jplot_dict = pickle.load(f)
            data1 = jplot_dict['data']
            hdr = jplot_dict['header']
        
        with open("/media/lelouedec/DATA/E-beacon/"+imgs_paths[p].split("/")[-1][:-2]+".p", 'rb') as f:
            jplot_dict = pickle.load(f)
            enhanced = jplot_dict['data']
            hdr = jplot_dict['header']
        

        data1 = np.nan_to_num(data1,np.nanmedian(data1)) 
        D2 = normalize(data1)
        if(os.path.exists("/media/lelouedec/DATA/rdifs_science_120/"+imgs_paths[p].split("/")[-1])):
            with open("/media/lelouedec/DATA/rdifs_science_120/"+imgs_paths[p].split("/")[-1], 'rb') as f:
                jplot_dict = pickle.load(f)
                sciencerdif = resize(normalize(jplot_dict['data']),(512,512),preserve_range=True)
            D2 = resize(D2,(512,512),preserve_range=True)

            f_transform = np.fft.fft2(sciencerdif)
            f_transform_shifted = np.fft.fftshift(f_transform)  # Shift zero frequency to center
            magnitude_spectrum_science = np.log1p(np.abs(f_transform_shifted))

            magnitudes_science.append(magnitude_spectrum_science)

            f_transform = np.fft.fft2(D2)
            f_transform_shifted = np.fft.fftshift(f_transform)  # Shift zero frequency to center
            magnitude_spectrum_beacon = np.log1p(np.abs(f_transform_shifted))

            magnitudes_beacon.append(magnitude_spectrum_beacon)

            f_transform = np.fft.fft2(enhanced)
            f_transform_shifted = np.fft.fftshift(f_transform)  # Shift zero frequency to center
            magnitude_spectrum_enhanced = np.log1p(np.abs(f_transform_shifted))

            magnitudes_enhanced.append(magnitude_spectrum_enhanced)




            intercept    = -(0.5 * contrast) + 0.5
            D2           = contrast * D2  + intercept
            sciencerdif  = contrast * sciencerdif + intercept
            enhanced     = contrast * enhanced + intercept
            # baseline_img        = contrast * baseline_img + intercept



            D2 = np.where(D2 > 1,1,D2)
            D2 = np.where(D2 < 0,0,D2)

            sciencerdif = np.where(sciencerdif > 1,1,sciencerdif)
            sciencerdif = np.where(sciencerdif < 0,0,sciencerdif)

            enhanced = np.where(enhanced > 1,1,enhanced)
            enhanced = np.where(enhanced < 0,0,enhanced)




            psnr_beacon,mse_beacon       = PSNR(sciencerdif,D2,data_range=D2.max()-D2.min()),MSE(sciencerdif,D2)
            psnr_enhanced,mse_enhanced   = PSNR(sciencerdif,enhanced,data_range=enhanced.max()-enhanced.min()),MSE(sciencerdif,enhanced)    

            PSNR_beacon.append(psnr_beacon)
            PSNR_enhanced.append(psnr_enhanced)
            MSE_beacon.append(mse_beacon)
            MSE_enhanced.append(mse_enhanced)

            ssim_beacon ,imgssimbea  = ssim(sciencerdif,D2,full=True,gaussian_weights=True,data_range=D2.max()-D2.min())
            ssim_enhanced,imgssime   = ssim(sciencerdif,enhanced,full=True,gaussian_weights=True,data_range=enhanced.max()-enhanced.min())

            SSIM_beacon.append(ssim_beacon)
            SSIM_enhanced.append(ssim_enhanced)

print(np.array(PSNR_beacon).mean(),np.array(PSNR_enhanced).mean())
print(np.array(MSE_beacon).mean(),np.array(MSE_enhanced).mean())
print(np.array(SSIM_beacon).mean(),np.array(SSIM_enhanced).mean())

np.save("/media/lelouedec/DATA/magnitude_science.npy",np.array(magnitudes_science))
np.save("/media/lelouedec/DATA/magnitude_beacon.npy",np.array(magnitudes_beacon))
np.save("/media/lelouedec/DATA/magnitude_enhanced.npy",np.array(magnitudes_enhanced))

            
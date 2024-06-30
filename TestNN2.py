import torch
from ESRGAN import *
import torch

import numpy as np
from PIL import Image
import torchvision
torchvision.disable_beta_transforms_warning()
import FILM


from natsort import natsorted
import glob 
import cv2
from datetime import datetime,timedelta
import matplotlib.pyplot as plt 
import os 
from astropy.io import fits


def test():
    device = torch.device("cpu")
    if(torch.backends.mps.is_available()):
        device = torch.device("mps")
    elif(torch.cuda.is_available()):
        device = torch.device("cuda")

    
    print("THIS WILL RUN ON DEVICE:", device)

    # sudo rmmod nvidia_uvm
    # sudo modprobe nvidia_uvm


    model = FILM.Interpolator()
    model.load_state_dict(torch.load("FILM_model1.pth",map_location=torch.device('cpu')))

    model.to(device).eval()
    # model2.to(device).eval()


    dt1 = torch.ones((1)).to(device) * 0.33
    dt2 = torch.ones((1)).to(device) * 0.66

    imgs_paths = natsorted(glob.glob("../res_model_final3/*"))
    with torch.no_grad():
        for p in range(0,len(imgs_paths)-1):

            time1 = datetime.strptime(imgs_paths[p].split("/")[-1][:-8], '%Y-%m-%dT%H-%M-%S')
            time2 = datetime.strptime(imgs_paths[p+1].split("/")[-1][:-8], '%Y-%m-%dT%H-%M-%S')
            diff_time = (time2-time1).total_seconds()
            
            if(diff_time/3600<4):

                S1 = torch.tensor(np.asarray(Image.open(imgs_paths[p]).convert("L"))/255.0).float().unsqueeze(0).unsqueeze(1).to(device)
                S2 = torch.tensor(np.asarray(Image.open(imgs_paths[p+1]).convert("L"))/255.0).float().unsqueeze(0).unsqueeze(1).to(device)
            
                output1 = model(S1,S2,dt1.unsqueeze(1))[0,0,:,:].cpu().numpy()
                output2 = model(S1,S2,dt2.unsqueeze(1))[0,0,:,:].cpu().numpy()

                output1 = (output1 - output1.min())/(output1.max()-output1.min())
                output2 = (output2 - output2.min())/(output2.max()-output2.min())

                timeout1 = time1+timedelta(seconds=diff_time*0.33)
                timeout2 = time1+timedelta(seconds=diff_time*0.66)
                name1    = str(timeout1.year) +"-"+ '%02d' %timeout1.month +"-"+'%02d' % timeout1.day+"T"+'%02d' %timeout1.hour+"-"+'%02d' % timeout1.minute+"-"+'%02d' %timeout1.second+".000"
                name2    = str(timeout2.year) +"-"+ '%02d' %timeout2.month +"-"+'%02d' % timeout2.day+"T"+'%02d' %timeout2.hour+"-"+'%02d' % timeout2.minute+"-"+'%02d' %timeout2.second+".000"
                

                ## move img input to fits file with correct header
                name = '../test_fits/'+imgs_paths[p].split("/")[-1][:-3]+"fts"
                filea = fits.open(name)
                hdr = filea[0].header
                fits.writeto("../enhanced_fits/"+imgs_paths[p].split("/")[-1][:-3]+"fts", S1[0,0,:,:].cpu().numpy().astype(np.float32), hdr, output_verify='silentfix', overwrite=True)
                filea.close()

                filea = fits.open('../test_fits/'+imgs_paths[p+1].split("/")[-1][:-3]+"fts")
                hdr2 = filea[0].header
                filea.close()

                hdr1_2 = hdr.copy()
                hdr2_2 = hdr.copy()


                crval1 = hdr["crval1a"] + (hdr2["crval1a"] - hdr["crval1a"])*0.33
                crval2 = hdr["crval2a"] + (hdr2["crval2a"] - hdr["crval2a"])*0.33
                hdr1_2["crval1a"] = crval1
                hdr1_2["crval2a"] = crval2
                hdr1_2["DATE-END"] = name1


                crval1 = hdr["crval1a"] + (hdr2["crval1a"] - hdr["crval1a"])*0.66
                crval2 = hdr["crval2a"] + (hdr2["crval2a"] - hdr["crval2a"])*0.66
                hdr2_2["crval1a"] = crval1
                hdr2_2["crval2a"] = crval2
                hdr2_2["DATE-END"] = name2
                
                fits.writeto("../enhanced_fits/"+name1+".fts", output1.astype(np.float32), hdr, output_verify='silentfix', overwrite=True)
                fits.writeto("../enhanced_fits/"+name2+".fts", output2.astype(np.float32), hdr, output_verify='silentfix', overwrite=True)

            else:
                name = '../test_fits/'+imgs_paths[p].split("/")[-1][:-3]+"fts"
                filea = fits.open(name)
                hdr = filea[0].header
                fits.writeto("../enhanced_fits/"+imgs_paths[p].split("/")[-1][:-3]+"fts", S1[0,0,:,:].cpu().numpy().astype(np.float32), hdr, output_verify='silentfix', overwrite=True)
                filea.close()


        name = '../test_fits/'+imgs_paths[p+1].split("/")[-1][:-3]+"fts"
        filea = fits.open(name)
        hdr = filea[0].header
        fits.writeto("../enhanced_fits/"+imgs_paths[p+1].split("/")[-1][:-3]+"fts", S2[0,0,:,:].cpu().numpy().astype(np.float32), hdr, output_verify='silentfix', overwrite=True)
        filea.close()

           


        





if __name__ == "__main__":
    test()



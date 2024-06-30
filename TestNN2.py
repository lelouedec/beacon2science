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

    imgs_paths = natsorted(glob.glob("/Volumes/Data_drive/res_model_final3/*"))
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
                name1    = str(timeout1.year) +"-"+ '%02d' %timeout1.month +"-"+'%02d' % timeout1.day+"T"+'%02d' %timeout1.hour+"-"+'%02d' % timeout1.minute+"-"+'%02d' %timeout1.second
                name2    = str(timeout2.year) +"-"+ '%02d' %timeout2.month +"-"+'%02d' % timeout2.day+"T"+'%02d' %timeout2.hour+"-"+'%02d' % timeout2.minute+"-"+'%02d' %timeout2.second
                os.system("mv "+imgs_paths[p]+" /Volumes/Data_drive/res_NN2_1/"+imgs_paths[p].split("/")[-1])

                img = Image.fromarray((output1*255.0).astype(np.uint8))
                img.save("/Volumes/Data_drive/res_NN2_1/"+name1+".png")

                img = Image.fromarray((output2*255.0).astype(np.uint8))
                img.save("/Volumes/Data_drive/res_NN2_1/"+name2+".png")
            else:
                os.system("mv "+imgs_paths[p]+" /Volumes/Data_drive/res_NN2_1/"+imgs_paths[p].split("/")[-1])

           


        





if __name__ == "__main__":
    test()



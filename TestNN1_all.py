import torch
from models.ESRGAN import *
import torch

import numpy as np
from PIL import Image
import torchvision
torchvision.disable_beta_transforms_warning()
import models.ESRGAN as ESRGAN
import models.unet2 as unet2


from natsort import natsorted
import glob 
import cv2
from datetime import datetime
import matplotlib.pyplot as plt 



def test():
    device = torch.device("cpu")
    if(torch.backends.mps.is_available()):
        device = torch.device("mps")
    elif(torch.cuda.is_available()):
        device = torch.device("cuda")

    
    print("THIS WILL RUN ON DEVICE:", device)

    # sudo rmmod nvidia_uvm
    # sudo modprobe nvidia_uvm


    model = unet2.ResUnet(2,full_size=512)

    model.load_state_dict(torch.load("gan_gen_ssim3.pth", map_location=torch.device('cpu')))
    # model2.load_state_dict(torch.load("gan_gen_nod.pth", map_location=torch.device('cpu')))


    model.to(device).eval()
    # model2.to(device).eval()

    save_path = "res_model_final4"


    imgs_paths = natsorted(glob.glob("/Volumes/Data_drive/Test_images/*"))
    clahe_b = cv2.createCLAHE(clipLimit=10,tileGridSize=(10,10))
    results = [[] for i in range(len(imgs_paths))]
    with torch.no_grad():
        for p in range(0,len(imgs_paths)-1):


            time1 = datetime.strptime(imgs_paths[p].split("/")[-1][:-8], '%Y-%m-%dT%H-%M-%S')
            time2 = datetime.strptime(imgs_paths[p+1].split("/")[-1][:-8], '%Y-%m-%dT%H-%M-%S')
            diff_time = (time2-time1).total_seconds()
            
            if(diff_time/3600<12):
                # if not os.path.exists("/Volumes/Data_drive/res_model11/"+imgs_paths[p].split("/")[-1]):
                LR1 = torch.tensor(clahe_b.apply(np.asarray(Image.open(imgs_paths[p]).convert("L")))/255.0).float().unsqueeze(0).unsqueeze(1).to(device)
                LR2 = torch.tensor(clahe_b.apply(np.asarray(Image.open(imgs_paths[p+1]).convert("L")))/255.0).float().unsqueeze(0).unsqueeze(1).to(device)
            
                sr = model(LR1,LR2)

                sr1 = sr[0,0,:,:].detach().cpu().numpy()
                sr2 = sr[0,1,:,:].detach().cpu().numpy()

        #         if(len(results[p])==0):
        #             results[p].append(sr1)
        #         else:
        #             combined = Image.fromarray(( np.mean([results[p][0],sr1],0)*255.0).astype(np.uint8))
        #             combined.save("/Volumes/Data_drive/"+save_path+"/"+imgs_paths[p].split("/")[-1])

                
        #         results[p+1].append(sr2)
        #     else:
        #         combined = Image.fromarray(( np.mean(results[p],0)*255.0).astype(np.uint8))
        #         combined.save("/Volumes/Data_drive/"+save_path+"/"+imgs_paths[p].split("/")[-1])

        # img = Image.fromarray((results[p+1][0]*255.0).astype(np.uint8))
        # img.save("/Volumes/Data_drive/"+save_path+"/"+imgs_paths[p+1].split("/")[-1])

                img = Image.fromarray((sr1*255.0).astype(np.uint8))
                img.save("/Volumes/Data_drive/"+save_path+"/"+imgs_paths[p].split("/")[-1])

                img = Image.fromarray((sr2*255.0).astype(np.uint8))
                img.save("/Volumes/Data_drive/"+save_path+"/"+imgs_paths[p+1].split("/")[-1])


        





if __name__ == "__main__":
    test()



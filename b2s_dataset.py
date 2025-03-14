import tkinter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 


import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import numpy as np
from PIL import Image
import cv2
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import pickle
# from astropy.wcs import WCS
from scipy.ndimage import shift
import json
from datetime import datetime
from astropy.io import fits
import random
from skimage.transform import resize
from skimage import exposure,filters
from astropy.wcs import WCS,utils
from skimage.morphology import square

import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)

from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter('ignore', category=AstropyWarning)


class FinalDataset(Dataset):
    def __init__(self, l_resolution=128, r_resolution=256,path="../Dataset/training",training=True,validation=False,small=False):
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.path = path
        self.training = training
        self.validation = validation
       
       
        with open("dataset_train_final.json", "r") as final:
            self.data_json =  json.load(final)
      
        
        random.seed(42)
        np.random.seed(seed=42)
        all_indices = np.arange(0,len(self.data_json))
        indices_train = np.array(random.sample(range(0,all_indices.shape[0]), int(all_indices.shape[0]*0.90)))
        indices_train = np.sort(indices_train)

        validation_indices = np.setdiff1d(all_indices, indices_train,assume_unique=True)


        if(validation):
            self.data_json = [self.data_json[i] for i in validation_indices]
        elif(training and not validation):
            self.data_json = [self.data_json[i] for i in indices_train]
        
        print(len(self.data_json))

    def __len__(self):
        return len(self.data_json)
    
   
    def normalize(self,img,rangev=2.5):      
        vmax = np.median(img)+rangev*np.std(img)
        vmin = np.median(img)-rangev*np.std(img)

        img[img>vmax] = vmax
        img[img<vmin] = vmin

        img = (img-vmin)/(vmax-vmin)

        img[img>1.0] = 1.0
        return img

    def __getitem__(self, index):
        
        if(self.training ==False and self.validation==False):
            typeset= "Test"
        else:
            typeset= "Train"
        try:
            s1 = np.asarray(fits.open(self.path+typeset+"/science/"+self.data_json[index]["s1"].split('T')[0]+'/'+self.data_json[index]["s1"])[0].data)
            b1 = np.asarray(fits.open(self.path+typeset+"/beacon/"+self.data_json[index]["b1"].split('T')[0]+'/'+self.data_json[index]["b1"])[0].data)

            s1_h = fits.open(self.path+typeset+"/science/"+self.data_json[index]["s1"].split('T')[0]+'/'+self.data_json[index]["s1"])[0].header
            b1_h = fits.open(self.path+typeset+"/beacon/"+self.data_json[index]["b1"].split('T')[0]+'/'+self.data_json[index]["b1"])[0].header
        except:
            s1 = np.asarray(fits.open(self.path+typeset+"/science/"+self.data_json[index]["s2"].split('T')[0]+'/'+self.data_json[index]["s1"])[0].data)
            b1 = np.asarray(fits.open(self.path+typeset+"/beacon/"+self.data_json[index]["b2"].split('T')[0]+'/'+self.data_json[index]["b1"])[0].data)
        
        b2 = np.asarray(fits.open(self.path+typeset+"/beacon/"+self.data_json[index]["b2"].split('T')[0]+'/'+self.data_json[index]["b2"])[0].data)
        s2 = np.asarray(fits.open(self.path+typeset+"/science/"+self.data_json[index]["s2"].split('T')[0]+'/'+self.data_json[index]["s2"])[0].data)

        s2_h = fits.open(self.path+typeset+"/science/"+self.data_json[index]["s2"].split('T')[0]+'/'+self.data_json[index]["s2"])[0].header
        b2_h = fits.open(self.path+typeset+"/beacon/"+self.data_json[index]["b2"].split('T')[0]+'/'+self.data_json[index]["b2"])[0].header
        
       
      

        center      = b2_h['crpix1']-1, b2_h['crpix2']-1
        wcs = WCS(b2_h,key='A')
        center_prev = wcs.all_world2pix(b1_h["crval1a"],b1_h["crval2a"], 0)
        shift_arr = [center_prev[1]-center[1],(center_prev[0]-center[0])]


        center_science      = s2_h['crpix1']-1, s2_h['crpix2']-1
        wcsscience = WCS(s2_h,key='A')
        center_prev_science = wcsscience.all_world2pix(s1_h["crval1a"],s1_h["crval2a"], 0)
        shift_arr_science = [center_prev_science[1]-center_science[1],(center_prev_science[0]-center_science[0])]




        s1 = np.nan_to_num(s1,np.nanmedian(s1))
        s2 = np.nan_to_num(s2,np.nanmedian(s2))

        b1 = np.nan_to_num(b1,np.nanmedian(b1))
        b2 = np.nan_to_num(b2,np.nanmedian(b2))

        diff1   = np.float32(s2.copy()-shift(s1.copy(),shift_arr_science     , order=3,mode='nearest'))
        diff2   = np.float32(b2.copy()-shift(b1.copy(),shift_arr             , order=3,mode='nearest'))

        
        diff1 = np.nan_to_num(diff1,np.nanmedian(diff1))
        diff2 = np.nan_to_num(diff2,np.nanmedian(diff2))

        

        tr1 = np.array([shift_arr[1],shift_arr[0]])


        b1  =  resize(b1, (self.l_res , self.l_res),anti_aliasing=True,order=3,preserve_range=True)
        b2  =  resize(b2, (self.l_res , self.l_res),anti_aliasing=True,order=3,preserve_range=True)

        s1  =  resize(s1, (self.r_res , self.r_res),anti_aliasing=True,order=3,preserve_range=True)
        s2  =  resize(s2, (self.r_res , self.r_res),anti_aliasing=True,order=3,preserve_range=True)
       
        diff1 = resize(diff1, (self.r_res , self.r_res),anti_aliasing=True,order=3,preserve_range=True)
        diff2 = resize(diff2, (self.l_res , self.l_res),anti_aliasing=True,order=3,preserve_range=True)




        diff1 = self.normalize(diff1)
        diff2 = self.normalize(diff2)
        





        return {"LR1":torch.tensor(b1).unsqueeze(0).float(),
                "LR2":torch.tensor(b2).unsqueeze(0).float(),
                "HR1":torch.tensor(s1).unsqueeze(0).float(),
                "HR2":torch.tensor(s2).unsqueeze(0).float(),
                "diff1":torch.tensor(diff1).unsqueeze(0).float(),
                "diff2":torch.tensor(diff2).unsqueeze(0).float(),
                "time":self.data_json[index]["b2"],
                "tr1":tr1,           
                }
    


      

     



if __name__ == "__main__":
    dataset = FinalDataset(256,512,"../Dataset/test/",False,False,small=False)
    print(dataset.__len__())

    beta = 2.5
    for i in range(dataset.__len__()-100,dataset.__len__()):
        item = dataset.__getitem__(i)

        fig,ax= plt.subplots(2,3,figsize=(20,10))
        ax[0][0].imshow(item["HR1"][0].cpu().numpy(),cmap='gray',vmin=np.nanmedian(item["HR1"][0].cpu().numpy())-beta*np.nanstd(item["HR1"][0].cpu().numpy()),vmax=np.nanmedian(item["HR1"][0].cpu().numpy())+beta*np.nanstd(item["HR1"][0].cpu().numpy()))
        ax[0][0].axis("off")
        ax[0][0].set_title("HR1")
        ax[0][1].imshow(item["HR2"][0].cpu().numpy(),cmap='gray',vmin=np.nanmedian(item["HR2"][0].cpu().numpy())-beta*np.nanstd(item["HR2"][0].cpu().numpy()),vmax=np.nanmedian(item["HR2"][0].cpu().numpy())+beta*np.nanstd(item["HR2"][0].cpu().numpy()))
        ax[0][1].axis("off")
        ax[0][1].set_title("HR2")
        ax[0][2].imshow(item["diff1"][0].cpu().numpy(),cmap='gray')
        ax[0][2].axis("off")
        ax[0][2].set_title("HR1 s2 differences")

        ax[1][0].imshow(item["LR1"][0].cpu().numpy(),cmap='gray',vmin=np.nanmedian(item["LR1"][0].cpu().numpy())-beta*np.nanstd(item["LR1"][0].cpu().numpy()),vmax=np.nanmedian(item["LR1"][0].cpu().numpy())+beta*np.nanstd(item["LR1"][0].cpu().numpy()))
        ax[1][0].axis("off")
        ax[1][0].set_title("shift im_prev")
        ax[1][1].imshow(item["LR2"][0].cpu().numpy(),cmap='gray',vmin=np.nanmedian(item["LR2"][0].cpu().numpy())-beta*np.nanstd(item["LR2"][0].cpu().numpy()),vmax=np.nanmedian(item["LR2"][0].cpu().numpy())+beta*np.nanstd(item["LR2"][0].cpu().numpy()))
        ax[1][1].axis("off")
        ax[1][1].set_title("shift im")
        ax[1][2].imshow(item["diff2"][0].cpu().numpy(),cmap='gray')
        ax[1][2].axis("off")
        ax[1][2].set_title("diff differences")
        plt.savefig("test.png")
    

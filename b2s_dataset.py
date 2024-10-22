import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import matplotlib.pyplot as plt 
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

class FinalDataset(Dataset):
    def __init__(self, l_resolution=128, r_resolution=256,path="../Dataset/training",training=True,validation=False,small=False):
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.path = path
       
       
        if(training or validation):
            with open("dataset_train_final.json", "r") as final:
                self.data_json =  json.load(final)
        else:
            with open("dataset_test.json", "r") as final:
                self.data_json =  json.load(final)
        
        
        np.random.seed(seed=42)
        all_indices = np.arange(0,len(self.data_json))
        indices_train = np.array(random.sample(range(0,all_indices.shape[0]), int(all_indices.shape[0]*0.90)))
        validation_indices = np.setdiff1d(all_indices, indices_train,assume_unique=True)


        if(validation):
            self.data_json = [self.data_json[i] for i in validation_indices]
        elif(training and not validation):
            self.data_json = [self.data_json[i] for i in indices_train]
        
        print(len(self.data_json))

    def __len__(self):
        return len(self.data_json)
    
   


    def __getitem__(self, index):

       
        s1 = np.asarray(Image.open(self.path+"science/"+self.data_json[index]["s1"].replace("fts",".png")).convert("L"))/255.0
        s2 = np.asarray(Image.open(self.path+"science/"+self.data_json[index]["s2"].replace("fts",".png")).convert("L"))/255.0
        b1 = np.asarray(Image.open(self.path+"beacon/"+self.data_json[index]["b1"].replace("fts",".png")).convert("L"))/255.0
        b2 = np.asarray(Image.open(self.path+"beacon/"+self.data_json[index]["b2"].replace("fts",".png")).convert("L"))/255.0
        

        shift_arr = np.array(self.data_json[index]["shift"])
        


        diff1   = np.float32(s2-shift(s1,shift_arr, order=2,mode='nearest',prefilter=False))
        diff2   = np.float32(s1-shift(s2,-1.0*shift_arr, order=2,mode='nearest',prefilter=False))
        tr1 = np.array([shift_arr[1],shift_arr[0]])


        b1  =  cv2.resize(b1, (self.l_res , self.l_res),interpolation = cv2.INTER_CUBIC)
        b2  =  cv2.resize(b2, (self.l_res , self.l_res),interpolation = cv2.INTER_CUBIC)

        s1  =  cv2.resize(s1, (self.r_res , self.r_res),interpolation = cv2.INTER_CUBIC)
        s2  =  cv2.resize(s2, (self.r_res , self.r_res),interpolation = cv2.INTER_CUBIC)
       
        diff1 = cv2.resize(diff1, (self.r_res , self.r_res),interpolation = cv2.INTER_CUBIC)
        diff2 = cv2.resize(diff2, (self.r_res , self.r_res),interpolation = cv2.INTER_CUBIC)

     



        return {"LR1":torch.tensor(b1).unsqueeze(0).float(),
                "LR2":torch.tensor(b2).unsqueeze(0).float(),
                "HR1":torch.tensor(s1).unsqueeze(0).float(),
                "HR2":torch.tensor(s2).unsqueeze(0).float(),
                "diff1":torch.tensor(diff1).unsqueeze(0).float(),
                "diff2":torch.tensor(diff2).unsqueeze(0).float(),
                "tr1":tr1,           
                }
    


      

     



if __name__ == "__main__":
    dataset = FinalDataset(256,512,"/Volumes/Data_drive/",True)
    print(dataset.__len__())


    # shifts  = []
    for i in range(dataset.__len__()-100,dataset.__len__()):
        item = dataset.__getitem__(i)
        fig,ax= plt.subplots(2,3)
        ax[0][0].imshow(item["HR1"][0].cpu().numpy(),cmap='twilight_shifted')
        ax[0][0].axis("off")
        ax[0][0].set_title("HR1")
        ax[0][1].imshow(item["HR2"][0].cpu().numpy(),cmap='twilight_shifted')
        ax[0][1].axis("off")
        ax[0][1].set_title("HR2")
        ax[0][2].imshow(item["HR1"][0].cpu().numpy()-item["HR2"][0].cpu().numpy(),cmap='twilight_shifted')
        ax[0][2].axis("off")
        ax[0][2].set_title("HR1 s2 differences")

        ax[1][0].imshow(item["diff1"][0].cpu().numpy(),cmap='gray')
        ax[1][0].axis("off")
        ax[1][0].set_title("shift im_prev")
        ax[1][1].imshow(item["diff2"][0].cpu().numpy(),cmap='gray')
        ax[1][1].axis("off")
        ax[1][1].set_title("shift im")
        ax[1][2].imshow(item["diff1"][0].cpu().numpy()-item["diff2"][0].cpu().numpy(),cmap='gray')
        ax[1][2].axis("off")
        ax[1][2].set_title("diff differences")
        plt.show()
    

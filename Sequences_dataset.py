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

class FinalDatasetSequences(Dataset):
    def __init__(self, resolution=1024,path="../finals_test",training=True,validation=False):
        self.res = resolution
        self.path = path
       
       
        with open("sequences_dataset.json", "r") as final:
            self.data_json =  json.load(final)
       
        
        if(validation):
            self.data_json = self.data_json[int(len(self.data_json)*0.3): ]
        elif(training and not validation):
            self.data_json = self.data_json[:int(len(self.data_json)*0.98)]



    def __len__(self):
        return len(self.data_json)

    def __getitem__(self, index):

        self.clahe_s = cv2.createCLAHE(clipLimit=10,tileGridSize=(10,10))
       

        s1 = self.clahe_s.apply(np.asarray(Image.open(self.path+"images_science/"+self.data_json[index]["images"][0]).convert("L")))/255.0
        s3 = self.clahe_s.apply(np.asarray(Image.open(self.path+"images_science/"+self.data_json[index]["images"][1]).convert("L")))/255.0
        s4 = self.clahe_s.apply(np.asarray(Image.open(self.path+"images_science/"+self.data_json[index]["images"][2]).convert("L")))/255.0
        s2 = self.clahe_s.apply(np.asarray(Image.open(self.path+"images_science/"+self.data_json[index]["images"][3]).convert("L")))/255.0

        if(len(self.data_json[index])>4):
            s5 = self.clahe_s.apply(np.asarray(Image.open(self.path+"images_science/"+self.data_json[index]["images"][4]).convert("L")))/255.0
            s6 = self.clahe_s.apply(np.asarray(Image.open(self.path+"images_science/"+self.data_json[index]["images"][5]).convert("L")))/255.0

        

        
        time1 = datetime.strptime(self.data_json[index]["images"][0][:-8], '%Y-%m-%dT%H-%M-%S')
        time2 = datetime.strptime(self.data_json[index]["images"][3][:-8], '%Y-%m-%dT%H-%M-%S')
        time3 = datetime.strptime(self.data_json[index]["images"][1][:-8], '%Y-%m-%dT%H-%M-%S')
        time4 = datetime.strptime(self.data_json[index]["images"][2][:-8], '%Y-%m-%dT%H-%M-%S')


        diff_time  = (time2-time1).total_seconds()
        diff_time2 = (time3-time1).total_seconds()
        diff_time3 = (time4-time1).total_seconds()

        if(len(self.data_json[index])>4):
            time5 = datetime.strptime(self.data_json[index]["images"][4][:-8], '%Y-%m-%dT%H-%M-%S')
            time6 = datetime.strptime(self.data_json[index]["images"][5][:-8], '%Y-%m-%dT%H-%M-%S')

            diff_time4 = (time5-time2).total_seconds()
            diff_time5 = (time6-time2).total_seconds()
            ratio3 = diff_time4/diff_time
            ratio4 = diff_time5/diff_time


        ratio1 = diff_time2/diff_time
        ratio2 = diff_time3/diff_time

        shift_arr = np.array(self.data_json[index]["shift"])
        diff1   = np.float32(s3-shift(s1,shift_arr*ratio1, order=2,mode='nearest',prefilter=False))
        diff2   = np.float32(s4-shift(s3,shift_arr*ratio1, order=2,mode='nearest',prefilter=False))
        diff3   = np.float32(s2-shift(s4,shift_arr*ratio1, order=2,mode='nearest',prefilter=False))
        tr1 = np.array([shift_arr[1],shift_arr[0]])

        diff0   = np.float32(s2-shift(s1,shift_arr, order=2,mode='nearest',prefilter=False))

       

        s1  =  cv2.resize(s1, (self.res , self.res),interpolation = cv2.INTER_CUBIC)
        s2  =  cv2.resize(s2, (self.res , self.res),interpolation = cv2.INTER_CUBIC)
        s3  =  cv2.resize(s3, (self.res , self.res),interpolation = cv2.INTER_CUBIC)
        s4  =  cv2.resize(s4, (self.res , self.res),interpolation = cv2.INTER_CUBIC)

        diff1  =  cv2.resize(diff1, (self.res , self.res),interpolation = cv2.INTER_CUBIC)
        diff2  =  cv2.resize(diff2, (self.res , self.res),interpolation = cv2.INTER_CUBIC)
        diff3  =  cv2.resize(diff3, (self.res , self.res),interpolation = cv2.INTER_CUBIC)
       

     



        return {
                "IM1":torch.tensor(shift(s1,shift_arr, order=2,mode='nearest',prefilter=False)).unsqueeze(0).float(),
                "IM2":torch.tensor(s2).unsqueeze(0).float(),
                "IM3":torch.tensor(s3).unsqueeze(0).float(),
                "IM4":torch.tensor(s4).unsqueeze(0).float(),
                "ratio1":ratio1,
                "ratio2":ratio2,
                "diff1"   : torch.tensor(diff1).unsqueeze(0).float(),
                "diff2"   : torch.tensor(diff2).unsqueeze(0).float(),
                "diff3"   : torch.tensor(diff3).unsqueeze(0).float(),
                "diff0"   : torch.tensor(diff0).unsqueeze(0).float(),
                "shift"   :   tr1*ratio1
                }
    


      

     



if __name__ == "__main__":
    dataset = FinalDatasetSequences(1024,"/Volumes/Data_drive/",True)

    # shifts  = []
    for i in range(dataset.__len__()-100,dataset.__len__()):
        item = dataset.__getitem__(i)
        fig,ax= plt.subplots(2,4)
        ax[0][0].imshow(item["IM1"][0].cpu().numpy(),cmap='twilight_shifted')
        ax[0][0].axis("off")
        ax[0][0].set_title("IM1")
        ax[0][1].imshow(item["IM3"][0].cpu().numpy(),cmap='twilight_shifted')
        ax[0][1].axis("off")
        ax[0][1].set_title("IM3")
        ax[0][2].imshow(item["IM4"][0].cpu().numpy(),cmap='twilight_shifted')
        ax[0][2].axis("off")
        ax[0][2].set_title("IM4")
        ax[0][3].imshow(item["IM2"][0].cpu().numpy(),cmap='twilight_shifted')
        ax[0][3].axis("off")
        ax[0][3].set_title("IM2")

        ax[1][0].imshow(item["diff0"][0].cpu().numpy(),cmap='gray')
        ax[1][0].axis("off")
        ax[1][0].set_title("diff0")


        ax[1][1].imshow(item["diff1"][0].cpu().numpy(),cmap='gray')
        ax[1][1].axis("off")
        ax[1][1].set_title("diff1")
        ax[1][2].imshow(item["diff2"][0].cpu().numpy(),cmap='gray')
        ax[1][2].axis("off")
        ax[1][2].set_title("diff2")
        ax[1][3].imshow(item["diff3"][0].cpu().numpy(),cmap='gray')
        ax[1][3].axis("off")
        ax[1][3].set_title("diff3")
        plt.show()
    
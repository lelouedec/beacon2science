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
import random


class FinalDatasetSequences(Dataset):
    def __init__(self, resolution=1024,path="../finals_test",training=True,validation=False):
        self.res = resolution
        self.path = path
       
       
        with open("sequences_dataset_final.json", "r") as final:
            self.data_json =  json.load(final)
       
        np.random.seed(seed=42)
        all_indices = np.arange(0,len(self.data_json))
        indices_train = np.array(random.sample(range(0,all_indices.shape[0]), int(all_indices.shape[0]*0.90)))
        validation_indices = np.setdiff1d(all_indices, indices_train,assume_unique=True)
        
        print(all_indices.shape[0])
        print(indices_train.shape[0])
        print(validation_indices.shape[0])
                                         
        if(validation):
            self.data_json = [self.data_json[i] for i in indices_train]
        elif(training and not validation):
            self.data_json = [self.data_json[i] for i in validation_indices]



    def __len__(self):
        return len(self.data_json)

    def __getitem__(self, index):

        self.clahe_s = cv2.createCLAHE(clipLimit=10,tileGridSize=(10,10))
       

        s1 = self.clahe_s.apply(np.asarray(Image.open(self.path+"images_science/"+self.data_json[index]["images"][0]).convert("L")))/255.0
        s3 = self.clahe_s.apply(np.asarray(Image.open(self.path+"images_science/"+self.data_json[index]["images"][1]).convert("L")))/255.0
        s4 = self.clahe_s.apply(np.asarray(Image.open(self.path+"images_science/"+self.data_json[index]["images"][2]).convert("L")))/255.0
        s2 = self.clahe_s.apply(np.asarray(Image.open(self.path+"images_science/"+self.data_json[index]["images"][3]).convert("L")))/255.0

      
        time1 = datetime.strptime(self.data_json[index]["images"][0][:-9], '%Y-%m-%dT%H-%M-%S')
        time2 = datetime.strptime(self.data_json[index]["images"][3][:-9], '%Y-%m-%dT%H-%M-%S')
        
        time3 = datetime.strptime(self.data_json[index]["images"][1][:-9], '%Y-%m-%dT%H-%M-%S')
        time4 = datetime.strptime(self.data_json[index]["images"][2][:-9], '%Y-%m-%dT%H-%M-%S')
        
        # print(self.data_json[index]["images"][0],self.data_json[index]["images"][1],self.data_json[index]["images"][2],self.data_json[index]["images"][3])


        diff_time  = (time2-time1).total_seconds()
        diff_time2 = (time3-time1).total_seconds()
        diff_time3 = (time4-time1).total_seconds()

        

        ratio1 = diff_time2/diff_time
        ratio2 = diff_time3/diff_time

        shift_arr = np.array(self.data_json[index]["shift"])
        diff31   = np.float32(s3.copy()-shift(s1.copy(),shift_arr[1], order=2,mode='nearest',prefilter=False))
        diff43   = np.float32(s4.copy()-shift(s3.copy(),shift_arr[2], order=2,mode='nearest',prefilter=False))
        diff24   = np.float32(s2.copy()-shift(s4.copy(),shift_arr[3], order=2,mode='nearest',prefilter=False))
        
        
        
        
        tr43 = np.array([shift_arr[2][1],shift_arr[2][0]])
        tr31 = np.array([shift_arr[1][1],shift_arr[1][0]])
        tr24 = np.array([shift_arr[3][1],shift_arr[3][0]])
        if(time1.year<=2015):
            s1 = np.fliplr(s1)
            s2 = np.fliplr(s2)
            s3 = np.fliplr(s3)
            s4 = np.fliplr(s4)
            diff1 = np.fliplr(diff1)
            diff2 = np.fliplr(diff2)
            diff3 = np.fliplr(diff3)
            tr43[0] = -1*tr43[0]
            tr31[0] = -1*tr31[0]
            tr24[0] = -1*tr24[0]
        
       

        s1  =  cv2.resize(s1, (self.res , self.res),interpolation = cv2.INTER_CUBIC)
        s2  =  cv2.resize(s2, (self.res , self.res),interpolation = cv2.INTER_CUBIC)
        s3  =  cv2.resize(s3, (self.res , self.res),interpolation = cv2.INTER_CUBIC)
        s4  =  cv2.resize(s4, (self.res , self.res),interpolation = cv2.INTER_CUBIC)

        diff1  =  cv2.resize(diff1, (self.res , self.res),interpolation = cv2.INTER_CUBIC)
        diff2  =  cv2.resize(diff2, (self.res , self.res),interpolation = cv2.INTER_CUBIC)
        diff3  =  cv2.resize(diff3, (self.res , self.res),interpolation = cv2.INTER_CUBIC)
       

         



        return {
                "IM1":torch.tensor(s1).unsqueeze(0).float(),
                "IM2":torch.tensor(s2).unsqueeze(0).float(),
                "IM3":torch.tensor(s3).unsqueeze(0).float(),
                "IM4":torch.tensor(s4).unsqueeze(0).float(),
                "ratio1":ratio1,
                "ratio2":ratio2,
                "diff31"   : torch.tensor(diff31).unsqueeze(0).float(),
                "diff43"   : torch.tensor(diff43).unsqueeze(0).float(),
                "diff24"   : torch.tensor(diff24).unsqueeze(0).float(),
                "tr43"   : torch.tensor(tr43),
                "tr31"   : torch.tensor(tr31),
                "tr24"   : torch.tensor(tr24)
                }
    


      

     



if __name__ == "__main__":
    dataset = FinalDatasetSequences(resolution=1024,path="../",training=True,validation=False)
    print(dataset.__len__())

    # shifts  = []
    for i in range(dataset.__len__()-100,dataset.__len__()):
        item = dataset.__getitem__(i)

        diff120 =(item["IM2"]-item["IM1"])[0].cpu().numpy()
        diff120 = (diff120- diff120.min())/(diff120.max()-diff120.min())
        diff40  =(item["IM2"]-item["IM4"])[0].cpu().numpy()
        diff40 = (diff40- diff40.min())/(diff40.max()-diff40.min())
        error   = diff40-diff120
        fig,ax = plt.subplots(1,3)
        ax[0].imshow(diff120,cmap='gray')
        ax[1].imshow(diff40,cmap='gray')
        ax[2].imshow(error,cmap='bwr')
        plt.show()
    

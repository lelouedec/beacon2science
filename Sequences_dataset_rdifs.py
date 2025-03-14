import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import matplotlib.pyplot as plt 
import matplotlib 
import numpy as np
from PIL import Image
import cv2
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import pickle
# from astropy.wcs import WCS
from scipy.ndimage import shift
import json
from datetime import datetime,timedelta
import random
from skimage.transform import resize
from kornia import filters
from skimage import exposure
from astropy.io import fits
from astropy.wcs import WCS,utils



import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)

from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter('ignore', category=AstropyWarning)

import os 


class FinalDatasetSequences(Dataset):
    def __init__(self, resolution=1024,path="../finals_test",training=True,validation=False):
        self.res = resolution
        self.path = path
       
       
        with open("sequence_dataset_final_rdifs.json", "r") as final:
            self.data_json =  json.load(final)
       
        self.data_json = self.data_json#[-2000:]
        np.random.seed(seed=42)
        all_indices = np.arange(0,len(self.data_json))
        indices_train = np.array(random.sample(range(0,all_indices.shape[0]), int(all_indices.shape[0]*0.90)))
        validation_indices = np.setdiff1d(all_indices, indices_train,assume_unique=True)
        
        print(all_indices.shape[0])
        print(indices_train.shape[0])
        print(validation_indices.shape[0])
                                         
        if(validation):
            self.data_json = [self.data_json[i] for i in validation_indices]
        elif(training and not validation):
            self.data_json = [self.data_json[i] for i in indices_train]
        

        self.blur = filters.MedianBlur((3,3))


    def __len__(self):
        return len(self.data_json)

    
    def normalize(self,img,rangev=2.5):
    
        vmax = np.median(img)+rangev*np.std(img)
        vmin = np.median(img)-rangev*np.std(img)

        img[img>vmax] = vmax
        img[img<vmin] = vmin

        img = (img-vmin)/( vmax-vmin)

        img[img>1.0] = 1.0
        return img

    def open_fits(self,index1,index):
        name = self.data_json[index1]["images"][index]
        filename = name.replace("..png",".fts")
        day = name.split('T')[0]
        date = datetime.strptime(day, "%Y-%m-%d")
        typeset="Train"


        separation = [-3,-2,-1,0,1,2,3]

        for s in separation:
            day2 = date+timedelta(days=s)
            day2str = day2.strftime("%Y-%m-%d")

            path = self.path+typeset+"/science/"+day2str+'/'+filename
             
            if(os.path.isfile(path)):
                s1 = np.asarray(fits.open(path)[0].data)
                h1 = fits.open(path)[0].header
                return s1,h1

    def removenans(self,img):
        return np.nan_to_num(img,np.nanmedian(img))

    def compute_shift(self,h1,h2):

        center      = h2['crpix1']-1, h2['crpix2']-1
        wcs = WCS(h2,key='A')
        center_prev = wcs.all_world2pix(h1["crval1a"],h1["crval2a"], 0)
        shift_arr = [center_prev[1]-center[1],(center_prev[0]-center[0])]

        return shift_arr


    def __getitem__(self, index):

       

        s1,h1 = self.open_fits(index,0) 
        s2,h2 = self.open_fits(index,1) 
        s3,h3 = self.open_fits(index,2) 
        s4,h4 = self.open_fits(index,3) 
        s5,h5 = self.open_fits(index,4) 
        s6,h6 = self.open_fits(index,5) 
        s7,h7 = self.open_fits(index,6) 


        s1 = self.removenans(s1) 
        s2 = self.removenans(s2) 
        s3 = self.removenans(s3) 
        s4 = self.removenans(s4) 
        s5 = self.removenans(s5) 
        s6 = self.removenans(s6) 
        s7 = self.removenans(s7) 



        diff1   = np.float32(s4.copy()-shift(s1.copy(),self.compute_shift(h1,h4) , order=3,mode='nearest'))
        diff2   = np.float32(s7.copy()-shift(s4.copy(),self.compute_shift(h4,h7) , order=3,mode='nearest'))


        mid1   = np.float32(s5.copy()-shift(s2.copy(),self.compute_shift(h2,h5) , order=3,mode='nearest'))
        mid2   = np.float32(s6.copy()-shift(s3.copy(),self.compute_shift(h3,h6), order=3,mode='nearest'))



        diff1 = np.nan_to_num(diff1,np.nanmedian(diff1))
        diff2 = np.nan_to_num(diff2,np.nanmedian(diff2))

        mid1 = np.nan_to_num(mid1,np.nanmedian(mid1))
        mid2 = np.nan_to_num(mid2,np.nanmedian(mid2))



        time1 = datetime.strptime(self.data_json[index]["images"][0][:-9], '%Y-%m-%dT%H-%M-%S')

    


        if(time1.year<=2015):
            diff1 = np.fliplr(diff1)
            diff2 = np.fliplr(diff2)
            mid1 = np.fliplr(mid1)
            mid2 = np.fliplr(mid2)
           
      

        diff1  = resize(diff1, (self.res , self.res),anti_aliasing=True,order=3,preserve_range=True)
        diff2  = resize(diff2, (self.res , self.res),anti_aliasing=True,order=3,preserve_range=True)
        mid1   = resize(mid1, (self.res , self.res),anti_aliasing=True,order=3,preserve_range=True)
        mid2   = resize(mid2, (self.res , self.res),anti_aliasing=True,order=3,preserve_range=True)

        diff1 = self.normalize(diff1)
        diff2 = self.normalize(diff2)

      

        mid1 = self.normalize(mid1)
        mid2 = self.normalize(mid2)

    
       

        return {
                
                "ratio1":0.33,
                "ratio2":0.66,
                "diff1"   : torch.tensor(diff1).unsqueeze(0).float(),
                "diff2"   : torch.tensor(diff2).unsqueeze(0).float(),
                "mid1"    : torch.tensor(mid1).unsqueeze(0).float(),
                "mid2"    : torch.tensor(mid2).unsqueeze(0).float(),
                }
    


      

     



if __name__ == "__main__":
    dataset = FinalDatasetSequences(resolution=512,path="../Dataset/",training=True,validation=False)
    print(dataset.__len__())

    # shifts  = []
    for i in range(0,dataset.__len__()):
        item = dataset.__getitem__(i)

       
        fig,ax = plt.subplots(1,4)
        ax[0].imshow(item["diff1"][0],cmap='gray')
        ax[1].imshow(item["diff2"][0],cmap='gray')
        ax[2].imshow(item["mid1"][0],cmap='gray')
        ax[3].imshow(item["mid2"][0],cmap='gray')
        plt.savefig("test.png")
        # plt.show()
from natsort import natsorted
import glob
from astropy.io import fits
import functions
import os 
import numpy as np 
import multiprocessing as mp




def reduce_data(hduls,data,header,bflag,instru):
    clean_data,clean_header = functions.reduction(header[0]["DATE-END"],hduls,data,header,'A',instru,bflag,"test2/calibration/","test2/data/hi/",silent=True)
    return clean_data,clean_header

global_dates   = []
global_type    = "science"
global_typeset = "Train"
pathreduced     = "/Volumes/Data_drive/Reduced/"
instru= "hi_1"
def multiprocessing_reduction(i):
    date = global_dates[i]
    
    files = natsorted(glob.glob(date+"/"+global_type+"/*"))
    print(date+"/"+global_type+"/*")

    datas = []
    headers = []
    hduls = []
    for f in files:
        try:
            filea = fits.open(f)
            datas.append(filea[0].data)
            headers.append(filea[0].header)
            hduls.append(filea)
            filea.close()
        except:
            print("corrupted file")

    if(len(datas)>0 and len(headers)>0):
        data_red,clean_headers = reduce_data(hduls,datas,headers,global_type,instru)
        if(data_red is not None and clean_headers is not None):
            for d in range(0,len(data_red)):
                path = pathreduced+global_typeset+"/"+clean_headers[d]["DATE-END"][:10]+"/"+global_type
                if not os.path.exists(path):
                    os.makedirs(path)
                fits.writeto(path+"/"+clean_headers[d]["DATE-END"].replace(":","-")+".fts", data_red[d].astype(np.float32), clean_headers[d], output_verify='silentfix', overwrite=True)


def reduction_dataset(path,typeset,datatype):
    paths = natsorted(glob.glob(path))
   

    for j in range(0,len(paths),4):
        
        # if(purged_training_dates[j]>start_date):
        lstofdates = paths[j:j+4]

        global global_dates ,global_type,global_typeset
        global_dates = lstofdates
        global_type = datatype
        global_typeset = typeset

        pool=mp.get_context('fork').Pool(processes=4)
        pool.map(multiprocessing_reduction, np.arange(0,len(lstofdates),1))
        pool.close()
        pool.join()

def reduction_dates(paths,typeset,datatype):
     for j in range(0,len(paths),4):
        
        # if(purged_training_dates[j]>start_date):
        lstofdates = paths[j:j+4]

        global global_dates ,global_type,global_typeset
        global_dates = lstofdates
        global_type = datatype
        global_typeset = typeset

        pool=mp.get_context('fork').Pool(processes=4)
        pool.map(multiprocessing_reduction, np.arange(0,len(lstofdates),1))
        pool.close()
        pool.join()


if __name__ == "__main__":
    reduction_dataset("./HI1_data/Train/*","Train","science")

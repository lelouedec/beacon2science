from datetime import datetime,timedelta
from natsort import natsorted
from astropy.io import fits
import glob 
import numpy as np 
import skimage 
import os 
import Create_dataset
import multiprocessing as mp



def get_fits_date(date,path_reduced,typeset,type):
    data_background = []
    header_background = []
    
    prefix=str(date.strftime('%Y'))+"-"+str(date.strftime('%m'))+"-"+str(date.strftime('%d'))
    files_list = natsorted(glob.glob(path_reduced+typeset+'/'+prefix+"/"+type+"/*"))
    print(path_reduced+typeset+'/'+prefix+"/"+type+"/*",files_list)
    
    for f in files_list:
        filea = fits.open(f)
        data_background.append(filea[0].data)
        header_background.append(filea[0].header)
        filea.close()
    return data_background,header_background


def create_l2(d,type="beacon",typeset="Train",path_reduced='/Volumes/Data_drive/Reduced/',path_to_save="/Volumes/Data_drive/L2_data/",returned=False,bgtype="median"):
  
    print("processing date ", d)
    #### load and create background data from fits files of previous 5 days 
  
    dat1 = d
    data_background   = []
    header_background = []
    # while len(data_background)<36:
    for i in range(0,5):
        dat1 = dat1-timedelta(days=1)
        dat,hea = get_fits_date(dat1,path_reduced,typeset,type)
        data_background+=dat
        header_background+=hea

    w,h = data_background[0].shape
    for i in range(len(data_background)):
        nan_mask = np.isnan(data_background[i])
        data_background[i][nan_mask] = np.array(np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), data_background[i][~nan_mask]))
        data_background[i] = skimage.transform.resize(data_background[i],(w,h),preserve_range=True)
    data_background = np.array(data_background)

    background = np.median(data_background,axis=0)
   
  
    ##### load the data files and their headers and remove background from the data
    prefix=str(d.strftime('%Y'))+"-"+str(d.strftime('%m'))+"-"+str(d.strftime('%d'))
    files_list = natsorted(glob.glob(path_reduced+typeset+'/'+prefix+"/"+type+"/*"))
    datas   = []
    headers = []
    for f in files_list:
        filea = fits.open(f)
        datas.append(filea[0].data.copy()-skimage.transform.resize(background,filea[0].data.shape,preserve_range=True))
        headers.append(filea[0].header)
        filea.close()

    ### get last image from previous day and prepend it to the new L2 directory 
    dat1 = d-timedelta(days=1)
    prefix=str(dat1.strftime('%Y'))+"-"+str(dat1.strftime('%m'))+"-"+str(dat1.strftime('%d'))
    files_list = natsorted(glob.glob(path_reduced+typeset+'/'+prefix+"/"+type+"/*"))
    if(len(files_list)>0):
        files_list = files_list[-1]
        filea = fits.open(files_list)
        datas = [filea[0].data.copy()-background.copy()] + datas
        headers= [filea[0].header] + headers
        filea.close()
        

    for j in range(0,len(datas)):
        name = headers[j]["DATE-END"]
        name = name.replace(":","-")
        prefix=str(d.strftime('%Y'))+"-"+str(d.strftime('%m'))+"-"+str(d.strftime('%d'))
        path = path_to_save+typeset+"/"+type+"/"+prefix+"/"
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except:
            print("folder existed no creating it again ")
        if(returned==False):
            fits.writeto(path+name+".fts", np.float32(datas[j]), headers[j], output_verify='silentfix', overwrite=True)
    if(returned == False):
        return 
    else:
        return datas,headers 
    

### global variable I know not good but had to do it quick 
global_dates   = []
global_type    = "science"
global_typeset = "Train"
def multiprocessing_backgroundremoval(i):
    create_l2(global_dates[i],type=global_type,typeset=global_typeset)



def L1_to_L2():
    test_dates,_     = Create_dataset.create_dates_test()

    training_dates   = Create_dataset.get_dates_training()

    start_date = datetime(2006,4,1)

    cnt = 0
    purged_training_dates = []
    for td in training_dates:
        if(td not in test_dates and td not in purged_training_dates ):
            purged_training_dates.append(td)
            cnt+=1
    

    for j in range(0,len(test_dates),1):
        # lstofdates = test_dates[j:j+1]
        create_l2(test_dates[j],type="beacon",typeset="Test")
        # global global_dates ,global_type,global_typeset
        # global_dates = lstofdates
        # global_type = "beacon"
        # global_typeset = "Test"

        # pool=mp.get_context('fork').Pool(processes=1)
        # pool.map(multiprocessing_backgroundremoval, np.arange(0,len(lstofdates),1))
        # pool.close()
        # pool.join()


    # for j in range(0,len(purged_training_dates),4):
        
    #     if(purged_training_dates[j]>start_date):
    #         lstofdates = purged_training_dates[j:j+4]

    #         global global_dates ,global_type,global_typeset
    #         global_dates = lstofdates
    #         global_type = "beacon"
    #         global_typeset = "Test"

    #         pool=mp.get_context('fork').Pool(processes=4)
    #         pool.map(multiprocessing_backgroundremoval, np.arange(0,len(lstofdates),1))
    #         pool.close()
            # pool.join()


if __name__ == "__main__":  
    L1_to_L2()


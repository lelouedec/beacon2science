
import functions
import os 
import wget
import multiprocessing as mp
import numpy as np 
from datetime import datetime


folder_data = "./WunderbarDataset/"
# folder_data = "./HI2_data/"

global_type = "science"
global_urls1 = None
global_test = "Test"
def multi_processes_dl(i):
    newpath = folder_data+global_test+"/"+global_urls1[i][0].split("_")[0]+"/"+global_type+"/"

    if not os.path.exists(newpath):
        os.makedirs(newpath)

    if not os.path.isfile(newpath+global_urls1[i][0]):
        print(newpath+global_urls1[i][0])
        wget.download(global_urls1[i][1],newpath+global_urls1[i][0])


def Download(date,data_type,set_type="test",instru="Hi_1"):
    """
    Download and reduce data from a given date

    @param date: date string in the format of YYYYMMDD
    @param data_type: data source, either beacon data or science
    """
    date = str(date).split(" ")[0].replace("-","")
    print("DATE: ", str(date),"\n \n ",'https://stereo-ssc.nascom.nasa.gov/pub/ins_data/secchi/L0/a/img/'+instru+'/'+  str(date))
    if(data_type=="beacon"):
        if(set_type=='train'):
            url1 = 'https://stereo-ssc.nascom.nasa.gov/pub/ins_data/secchi/L0/a/img/'+instru+'/'+  str(date)
        else:
            url1 = 'https://stereo-ssc.nascom.nasa.gov/pub/beacon/ahead/secchi/img/'+instru+'/'+  str(date)     
        if(instru=="hi_1"):   
            urls1 = functions.listfd(url1, 's7h1A.fts')
        else:
            urls1 = functions.listfd(url1, 's7h2A.fts')
    else:
        url1 = 'https://stereo-ssc.nascom.nasa.gov/pub/ins_data/secchi/L0/a/img/'+instru+'/'+  str(date)
        if(instru=="hi_1"):
            urls1 = functions.listfd(url1, 's4h1A.fts')
        else:
            urls1 = functions.listfd(url1, 's4h2A.fts')


    newpath = folder_data+set_type+"/"+str(date)+"/"+data_type+"/"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
   
    global global_urls1 
    global_urls1 = urls1
    global global_type 
    global_type = data_type
    global global_test 
    global_test = set_type

    pool=mp.get_context('fork').Pool(processes=12)
    pool.map(multi_processes_dl, np.arange(0,len(urls1),1))
    pool.close()
    pool.join()


if __name__ == "__main__":
    dates = [datetime(2020,4,9),datetime(2020,4,10)]#,datetime(2020,4,13),datetime(2020,4,14),datetime(2020,4,15),datetime(2020,4,16),datetime(2020,4,17)]
    for d in dates:
        Download(d,"science",set_type="Train",instru='hi_2')
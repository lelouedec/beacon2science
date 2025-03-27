
# file management and os
import os 
from natsort import natsorted
import glob 

# web stuff 
import wget
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
from requests.adapters import HTTPAdapter, Retry
import requests
from bs4 import BeautifulSoup

# multiprocessing
import multiprocessing as mp

#numpy and array
import numpy as np
from astropy.io import fits

#utils
import functions
from datetime import datetime,timedelta
import skimage



datapath = "/scratch/aswo/jlelouedec/WunderbarDataset"


def listfd(input_url, extension):
    """
    Provides list of urls and corresponding file names to download.

    @param input_url: URL of STEREO-HI image files
    @param extension: File ending of STEREO-HI image files
    @return: List of URLs and corresponding filenames to be downloaded
    """

    disable_warnings(InsecureRequestWarning)

    
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)

    session.mount('http://', adapter)
    session.mount('https://', adapter)

    output_urls = []

    page = session.get(input_url).text
    #page = requests.get(input_url, verify=False).text

    soup = BeautifulSoup(page, 'html.parser')
    url_found = [input_url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(extension)]
    filename = [node.get('href') for node in soup.find_all('a') if node.get('href').endswith(extension)]

    for i in range(len(filename)):
        output_urls.append((filename[i], url_found[i]))

    return output_urls




global_type = "beacon"
global_urls1 = None
global_test = "forecast"
global_dates   = []


def multi_processes_dl(i):
    newpath = "./"+datapath+"/"+global_test+"/"+global_urls1[i][0].split("_")[0]+"/"+global_type+"/"

    if not os.path.exists(newpath):
        os.makedirs(newpath)

    if not os.path.isfile(newpath+global_urls1[i][0]):
        print(newpath+global_urls1[i][0])
        wget.download(global_urls1[i][1],newpath+global_urls1[i][0])




def Download(date,data_type,set_type="test"):
    """
    Download and reduce data from a given date

    @param date: date string in the format of YYYYMMDD
    @param data_type: data source, either beacon data or science
    """
        
    if(data_type=="beacon"):
        if(set_type=='train'):
            url1 = 'https://stereo-ssc.nascom.nasa.gov/pub/ins_data/secchi/L0/a/img/hi_1/' + str(date)
        else:
            url1 = 'https://stereo-ssc.nascom.nasa.gov/pub/beacon/ahead/secchi/img/hi_1/' + str(date)        
        urls1 = listfd(url1, 's7h1A.fts')
    else:
        url1 = 'https://stereo-ssc.nascom.nasa.gov/pub/ins_data/secchi/L0/a/img/hi_1/' + str(date)
        urls1 = listfd(url1, 's4h1A.fts')

    newpath = "./"+datapath+"/"+set_type+"/"+str(date)+"/"+data_type+"/"
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





def reduce_data(hduls,data,header,bflag):
    clean_data,clean_header = functions.reduction(header[0]["DATE-END"],hduls,data,header,'A',"hi_1",bflag,"./test2/calibration/","./test2/data/hi/",silent=True)
    return clean_data,clean_header




def multiprocessing_reduction(i):
    date = global_dates[i]
    
    files = natsorted(glob.glob(datapath+"/"+global_test+"/"+date+"/"+global_type+"/*"))
    

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
        data_red,clean_headers = reduce_data(hduls,datas,headers,global_type)
        if(data_red is not None and clean_headers is not None):
            for d in range(0,len(data_red)):
                path = "/scratch/aswo/jlelouedec/Reduced/"+global_typeset+"/"+clean_headers[d]["DATE-END"][:10].replace("-","")+"/"+global_type
                if not os.path.exists(path):
                    os.makedirs(path)
                fits.writeto(path+"/"+clean_headers[d]["DATE-END"].replace(":","")+".fts", data_red[d].astype(np.float32), clean_headers[d], output_verify='silentfix', overwrite=True)



def reduction_date(dates,datatype,typeset):
    print("downloading calcfiles")
    ## check for calfile and point files in test2 folder and download them if needed
    functions.check_calfiles("./test2/")
    functions.check_pointfiles("./test2/")
    print("reducing")
    for j in range(0,len(dates),4):
        
        # if(purged_training_dates[j]>start_date):
        lstofdates = dates[j:j+4]

        global global_dates ,global_type,global_typeset
        global_dates = lstofdates
        global_type = datatype
        global_typeset = typeset

        pool=mp.get_context('fork').Pool(processes=4)
        pool.map(multiprocessing_reduction, np.arange(0,len(lstofdates),1))
        pool.close()
        pool.join()

def get_x_last_days(x=3):
    dates = []
    now  = datetime.now() - timedelta(days = x+4)
    for i in range(0,x+5): ## add 5 days for background removal
        next_date = str(now.year)+str('%02d' % now.month)+str('%02d' % now.day)
        dates.append(next_date)
        now = now + timedelta(days=1)
    
    return dates


def get_fits_date(date,path_reduced,typeset,type):
    data_background = []
    header_background = []
    
    prefix=str(date.strftime('%Y'))+str(date.strftime('%m'))+str(date.strftime('%d'))
    files_list = natsorted(glob.glob(path_reduced+typeset+'/'+prefix+"/"+type+"/*"))
    
    for f in files_list:
        filea = fits.open(f)
        data_background.append(filea[0].data)
        header_background.append(filea[0].header)
        filea.close()
    return data_background,header_background




def create_l2(d,type="beacon",typeset="forecast",returned=False,bgtype="median"):
  
    path_reduced = "/scratch/aswo/jlelouedec/Reduced/"
    path_to_save = "/scratch/aswo/jlelouedec/L2_data/"

    d = datetime.strptime(d,'%Y%m%d')
    print("processing date ", d)
    dat1 = d
    data_background   = []
    header_background = []
    # while len(data_background)<36:
    for i in range(0,5):
        dat1 = dat1-timedelta(days=1)
        dat,hea = get_fits_date(dat1,path_reduced,typeset,type)
        data_background+=dat
        header_background+=hea

    if(len(data_background)>0):
        w,h = data_background[0].shape
        for i in range(len(data_background)):
            nan_mask = np.isnan(data_background[i])
            data_background[i][nan_mask] = np.array(np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), data_background[i][~nan_mask]))
            data_background[i] = skimage.transform.resize(data_background[i],(w,h),preserve_range=True)
        data_background = np.array(data_background)
    


        if(bgtype=="median"):
            background = np.median(data_background,axis=0)
        else:
            background = np.percentile(data_background,5,0)
    
        ##### load the data files and their headers and remove background from the data
        prefix=str(d.strftime('%Y'))+str(d.strftime('%m'))+str(d.strftime('%d'))
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
        prefix=str(dat1.strftime('%Y'))+str(dat1.strftime('%m'))+str(dat1.strftime('%d'))
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
            prefix=str(d.strftime('%Y'))+str(d.strftime('%m'))+str(d.strftime('%d'))
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
    

def run_all():
     #### Define days you want to download + add 5 days for background
    dates = ["20240505","20240506","20240507","20240508","20240509","20240510"]
    #### or get last x days
    dates = get_x_last_days(7)

    
    ## download the corresponding dates of data + background
    for d in dates:
        Download(d,"beacon","forecast")

    # reduce the corresponding dates using Maike's code (we keep a frozen version of it to avoid problems)
    reduction_date(dates,"beacon","forecast")

    ## we want to get L2 data for only the last dates/without background dates
    dates = dates[5:]
    # Create L2 data from the reduced data 
    for d in dates:
        create_l2(d,
                  type="beacon",
                  typeset="forecast",
                  returned=False, ## in case you want to get L2 arrays returned
                  bgtype="median") ## what kind of background
        
        
if __name__ == "__main__":
   run_all()

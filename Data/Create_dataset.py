from datetime import datetime,timedelta
from STEREOHI import functions
import os 
import multiprocessing as mp
import wget
import numpy as np 
import json
from natsort import natsorted
from astropy.io import fits
import glob 
import matplotlib.pyplot as plt 
import cv2
import pickle 
from PIL import Image
import matplotlib.colors as colors
import skimage 
import time 
from astropy.wcs import WCS,utils
from collections import Counter
# import Convert_reduce_fits
from sunpy.map import Map
from sunpy.coordinates import get_body_heliographic_stonyhurst
import requests 
from bs4 import BeautifulSoup
from scipy.ndimage import shift
import tqdm
import matplotlib
from skimage import exposure
import matplotlib
from scipy import stats
import tqdm


matplotlib.rcParams['backend'] = 'Qt5Agg' 





def create_dates_test():
    Test_events_selected = [ 
                        "30/08/2008",
                        "18/12/2008",
                        "09/01/2009",
                        "22/01/2009",
                        "31/01/2009",
                        "09/05/2009",
                        "16/06/2009",
                        "23/06/2009",
                        "18/10/2009",
                        "22/12/2009",
                        "08/04/2010",
                        "19/04/2010",
                        "08/05/2010",
                        "16/06/2010",
                        "21/06/2010",
                        "01/08/2010",
                        "07/10/2010",
                        "24/11/2010",
                        "23/12/2010",
                        "20/01/2011",
                        "30/01/2011",
                        "14/02/2011",
                        "25/03/2011",
                        "04/04/2011",
                        "04/08/2011",
                        "07/09/2011",
                        "27/10/2011",
                        "12/07/2012",
                        "12/03/2019",
                        "21/03/2019",
                        "11/05/2019",
                        "22/05/2019",
                        "09/07/2020",
                        "30/09/2020",
                        "10/10/2020",
                        "07/12/2020",
                        "08/12/2020",
                        "11/02/2021",
                        "20/02/2021",
                        "25/04/2021",
                        "09/06/2021",
                        "24/11/2021",
                        "10/03/2022",
                        "28/03/2022",
                        "09/04/2022",
                        "02/06/2022",
                        "13/06/2022",
                        "27/06/2022",
                    ]

    print("Number of events",len(Test_events_selected))
    new_events_list = []
    event_lists = []
    for i in range(0,len(Test_events_selected)):
        date = datetime.strptime(Test_events_selected[i],'%d/%m/%Y')
        dates = [
                date-timedelta(days=3),
                date-timedelta(days=2),
                date-timedelta(days=1),
                date,
                date+timedelta(days=1),
                date+timedelta(days=2),
                date+timedelta(days=3)
                ]
        event_lists.append(dates)
        for d in range(0,len(dates)):
            found = False
            for e in new_events_list:
                if(e==d):
                    found = True
            if (found==False):
                new_events_list.append(dates[d])
    return new_events_list,event_lists




def check_for_leaks():

    paths_science = natsorted(glob.glob("/Volumes/Data_drive/L2_data/Test/science/*"))

    for p in paths_science:
        date = p.split("/")[-1]
        if os.path.exists("/Volumes/Data_drive/L2_data/Train/science/"+date):
            print("p is in /Volumes/Data_drive/L2_data/Train/science/"+date)

    paths_beacon = natsorted(glob.glob("/Volumes/Data_drive/L2_data/Test/beacon/*"))

    for p in paths_beacon:
        date = p.split("/")[-1]
        if os.path.exists("/Volumes/Data_drive/L2_data/Train/beacon/"+date):
            print("p is in /Volumes/Data_drive/L2_data/Train/beacon/"+date)

   
        

def get_dates_training():
    f = open('helcat.json')
    data = json.load(f)
    data = data["data"]

    dates = []
    for d in data:
        if(d[2]=="A" and d[1].split(" ")[0]):
            dates.append(d[1].split(" ")[0])
    
    print("events training",len(dates))
    dates_final = []
    for i in range(len(dates)):
        dates_final.append(datetime(int(dates[i][:4]),int(dates[i][5:7]),int(dates[i][8:])))
        dates_final.append(datetime(int(dates[i][:4]),int(dates[i][5:7]),int(dates[i][8:]))+ timedelta(days=1))
        dates_final.append(datetime(int(dates[i][:4]),int(dates[i][5:7]),int(dates[i][8:]))+ timedelta(days=2))


    return dates_final



def get_day_and_background(day,len_bckg=7):
    dates = []
    # date = datetime.strptime(day,'%Y%m%d')
    dates.append(day)
    for i in range(1,len_bckg+1):
        dates.append(day-timedelta(days=i))

    # for i in range(len(dates)):
    #     dates[i] = dates[i].strftime('%Y%m%d')
    #     print(dates[i])
    return dates




def Download_dataset():
    _,test_dates     = create_dates_test()
    training_dates = get_dates_training()
    print(len(training_dates),len(test_dates))

    start_date = datetime(2010,7,31)

    purged_training_dates = []
    for td in training_dates:
        if(td not in test_dates and td not in purged_training_dates ):
            purged_training_dates.append(td)

    print("purged training events",len(purged_training_dates))
    exit()
    
    for j in range(0,len(purged_training_dates)):
        dates = get_day_and_background(purged_training_dates[j])
        for d in dates:
            Download(d,"science",set_type="Train")
            Download(d,"beacon",set_type="Train")
   

def create_dataset_images_pairs():
    days = natsorted(glob.glob('/Volumes/Data_drive/L2_data/Test/beacon/*'))
    pairs = []
    for i in tqdm.tqdm(range(0,len(days))):
        d = days[i]
        files = natsorted(glob.glob(d+"/*"))
        for j in range(0,len(files)-1):
                e  = files[j]
                e2 = files[j+1]
                pair = {}
                pair["b1"] = e.split("/")[-1]
                pair["b2"] = e2.split("/")[-1]
                paths = natsorted(glob.glob(d.replace("beacon","science")+"/"+e.split("/")[-1].split("T")[0]+"*"))
                if(e.split("/")[-1].split("T")[0]!=e2.split("/")[-1].split("T")[0]):
                    paths+=natsorted(glob.glob(d.replace("beacon","science")+e2.split("/")[-1].split("T")[0]+"*"))
                for p in paths:
                    if(p.split("/")[-1]==e.split("/")[-1]):
                        pair["s1"] = p.split("/")[-1]
                    elif(p.split("/")[-1]==e2.split("/")[-1]):
                        pair["s2"] = p.split("/")[-1]
                        
                if(len(pair.keys())==4):
                    pairs.append(pair)

                
    print(len(pairs))
    with open("dataset_test_final.json", "w") as final:
        json.dump(pairs, final)

def compute_shift_dataset():
    with open("dataset_test_final.json", "r") as final:
        data_json =  json.load(final)

    for i,d in enumerate(data_json):
        print(d)
        header_s1 = list("/Volumes/Data_drive/L2_data/Test/"+"science/"+d["s1"].split("T")[0]+"/"+d["s1"])
        header_s1 = "".join(header_s1)
        header_s2 = list("/Volumes/Data_drive/L2_data/Test/"+"science/"+d["s2"].split("T")[0]+"/"+d["s2"])
        header_s2 = "".join(header_s2)
        
        header_s1 = fits.open(header_s1)[0].header
        header_s2 = fits.open(header_s2)[0].header

        center      = header_s2['crpix1']-1, header_s2['crpix2']-1
        print(center)
        wcs = WCS(header_s2,key='A')
        center_prev = wcs.all_world2pix(header_s1["crval1a"],header_s1["crval2a"], 0)
        shift_arr1 = [center_prev[1]-center[1],(center_prev[0]-center[0])]
        print("shift science",shift_arr1)
        data_json[i]["shift"] = shift_arr1

        header_b1 = list("/Volumes/Data_drive/L2_data/Test/"+"beacon/"+d["b1"].split("T")[0]+"/"+d["s1"])
        header_b1 = "".join(header_b1)
        header_b2 = list("/Volumes/Data_drive/L2_data/Test/"+"beacon/"+d["b2"].split("T")[0]+"/"+d["s2"])
        header_b2 = "".join(header_b2)
        
        header_b1 = fits.open(header_b1)[0].header
        header_b2 = fits.open(header_b2)[0].header

        center      = header_b2['crpix1']-1, header_b2['crpix2']-1
        print(center)
        wcs = WCS(header_b2,key='A')
        center_prev = wcs.all_world2pix(header_b1["crval1a"],header_b1["crval2a"], 0)
        shift_arr = [center_prev[1]-center[1],(center_prev[0]-center[0])]
        print("shift beacon", shift_arr)
        data_json[i]["shift"] = shift_arr

        print("ratio" ,shift_arr1[0]/shift_arr[0],shift_arr1[1]/shift_arr[1])
        exit()
    
    with open("dataset_test_final.json", "w") as final:
        json.dump(data_json, final)


def get_shift(header1,header2):
    center      = header2['crpix1']-1, header2['crpix2']-1
    wcs = WCS(header2,key='A')
    center_prev = wcs.all_world2pix(header1["crval1a"],header1["crval2a"], 0)
    shift_arr = [center_prev[1]-center[1],(center_prev[0]-center[0])]
    return shift_arr


def create_dataset_sequences():
    files = natsorted(glob.glob("../Dataset/training/science/*"))
    #"2007-04-20T07-59-40.013
    sequences = []
    for i in tqdm.tqdm(range(0,len(files)-7,7)):
        dates  = [datetime.strptime(f.split("/")[-1][:-5],"%Y-%m-%dT%H-%M-%S.%f")for f in files[i:i+7]]
        diff   = np.array([(dates[j]- dates[j-1]).total_seconds()/60 for j in range(1,len(dates))])

        if(np.all(diff-40.0<10.0)):
            tiles_to_save = [f.split("/")[-1] for f in files[i:i+7]]
            sequence = {}
            sequence["images"] = tiles_to_save

            header_s1 = list("/Volumes/Data_drive/Reduced/Train/"+tiles_to_save[0].split("T")[0]+"/"+"science/"+tiles_to_save[0][:-4]+"fts")
            header_s1 = "".join(header_s1)

            header_s2 = list("/Volumes/Data_drive/Reduced/Train/"+tiles_to_save[1].split("T")[0]+"/"+"science/"+tiles_to_save[1][:-4]+"fts")
            header_s2 = "".join(header_s2)

            header_s3 = list("/Volumes/Data_drive/Reduced/Train/"+tiles_to_save[2].split("T")[0]+"/"+"science/"+tiles_to_save[2][:-4]+"fts")
            header_s3 = "".join(header_s3)

            header_s4 = list("/Volumes/Data_drive/Reduced/Train/"+tiles_to_save[3].split("T")[0]+"/"+"science/"+tiles_to_save[3][:-4]+"fts")
            header_s4 = "".join(header_s4)

            header_s5 = list("/Volumes/Data_drive/Reduced/Train/"+tiles_to_save[4].split("T")[0]+"/"+"science/"+tiles_to_save[4][:-4]+"fts")
            header_s5 = "".join(header_s5)

            header_s6 = list("/Volumes/Data_drive/Reduced/Train/"+tiles_to_save[5].split("T")[0]+"/"+"science/"+tiles_to_save[5][:-4]+"fts")
            header_s6 = "".join(header_s6)

            header_s7 = list("/Volumes/Data_drive/Reduced/Train/"+tiles_to_save[6].split("T")[0]+"/"+"science/"+tiles_to_save[6][:-4]+"fts")
            header_s7 = "".join(header_s7)


            

            header_s1 = fits.open(header_s1)[0].header
            header_s2 = fits.open(header_s2)[0].header
            header_s3 = fits.open(header_s3)[0].header
            header_s4 = fits.open(header_s4)[0].header
            header_s5 = fits.open(header_s5)[0].header
            header_s6 = fits.open(header_s6)[0].header
            header_s7 = fits.open(header_s7)[0].header
             
            


            shift1 = get_shift(header_s1,header_s2)
            shift2 = get_shift(header_s2,header_s3)
            shift3 = get_shift(header_s3,header_s4)
            shift4 = get_shift(header_s4,header_s5)
            shift5 = get_shift(header_s5,header_s6)
            shift6 = get_shift(header_s6,header_s7)
            sequence["shifts"] = [shift1,shift2,shift3,shift4,shift5,shift6]
            sequences.append(sequence)
    



    with open("sequence_dataset_final_rdifs.json", "w") as final:
        json.dump(sequences, final)

            

    
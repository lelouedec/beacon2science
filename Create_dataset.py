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
from skimage import exposure
import pickle 
from PIL import Image
import matplotlib.colors as colors
import skimage 
import time 
from astropy.wcs import WCS
from collections import Counter

def reduce_data(data,header,bflag):

    
    ftpsc = 'A'
    bad_indices = []
    indices     = np.arange(0,len(data))
    print("start",len(indices))
    
    n_images = [header[i]['n_images'] for i in range(len(header))]

    if bflag == 'science':
        norm_img = 30
    if bflag == 'beacon':
        norm_img = 1


    bad_indices+=[i for i in range(len(data)) if  n_images[i] != norm_img]

    indices = np.setdiff1d(indices, np.array(bad_indices))
    print("norm_img error  ",len(indices))


    crval1_test = [int(np.sign(header[i]['crval1'])) for i in indices]

    if(len(crval1_test)==0):
        return None,None

    

    median_crval = np.nanmedian(crval1_test)
    std_crval = np.nanstd(crval1_test)
    
    # bad_indices+=[indices[i] for i in range(0,len(indices)) if  np.abs(crval1_test[i]-median_crval)>4*std_crval]
    
    common_crval = Counter(crval1_test)
    com_val, count = common_crval.most_common()[0]

    print(com_val)
    

    bad_indices+=[indices[i] for i in range(0,len(indices)) if  crval1_test[i] != com_val]

    indices = np.setdiff1d(indices, np.array(bad_indices))
    print("CRVAL error  ",len(indices))


    if bflag == 'science':
        datamin_test = [header[i]['DATAMIN'] for i in indices]
        bad_indices+= [i for i in range(len(datamin_test)) if datamin_test[i] != norm_img]
        
    if bflag == 'beacon':
        test_data = np.array([data[i] for i in indices])
        bad_indices+= [indices[i] for i in range(0,len(test_data)) if test_data[i].sum()==0]


        indices = np.setdiff1d(indices, np.array(bad_indices))
        # test_data = np.array([data[i] for i in indices])

        # vals = []
        # for i in range(0,test_data.shape[0]):
        #     vals.append(test_data[i].sum())
        # median = np.nanmedian(np.array(vals))
        # std    = np.nanstd(np.array(vals))
        # bad_indices+= [indices[i] for i in range(0,len(test_data)) if (test_data[i].sum()-median)>std]
        
    indices = np.setdiff1d(indices, np.array(bad_indices))
    # print("Sum error  ",len(indices))

    missing_ind = np.array([header[i]['NMISSING'] for i in indices])
    bad_indices+= [i for i in range(len(missing_ind)) if missing_ind[i] > 0]
    indices = np.setdiff1d(indices, np.array(bad_indices))
    print("end ",len(indices))
    clean_headers = [header[i] for i in indices]
    clean_data    = np.array([data[i] for i in indices])


    
    if bflag == 'beacon':
        for i in range(len(clean_headers)):
            functions.hi_fix_beacon_date(clean_headers[i])


    crval1 = [clean_headers[i]['crval1'] for i in range(len(clean_headers))]

    if ftpsc == 'A':    
        post_conj = [int(np.sign(crval1[i])) for i in range(len(crval1))]

    if ftpsc == 'B':    
        post_conj = [int(-1*np.sign(crval1[i])) for i in range(len(crval1))]
        
    if len(set(post_conj)) == 1:

        post_conj = post_conj[0]

        if post_conj == -1:
            post_conj = False
        if post_conj == 1:
            post_conj = True

    else:
        print('Corrupted CRVAL1 in header. Exiting...')
        return None,None
        

    dateavg = [clean_headers[i]['date-avg'] for i in range(len(clean_headers))]

    timeavg = [datetime.strptime(dateavg[i], '%Y-%m-%dT%H:%M:%S.%f') for i in range(len(dateavg))]
    start_time = time.time()
    data_trim = np.array([functions.scc_img_trim(clean_data[i], clean_headers[i]) for i in range(len(clean_data))])
    print("TIME trim", time.time()-start_time)


    start_time = time.time()
    data_sebip = [functions.scc_sebip(data_trim[i], clean_headers[i], True) for i in range(len(data_trim))]
    print("TIME sebip", time.time()-start_time)


    # maps are created from corrected data
    # header is saved into separate list
    start_time = time.time()
    biasmean = [functions.get_biasmean(clean_headers[i]) for i in range(len(clean_headers))]
    biasmean = np.array(biasmean)
    print("TIME biasmean", time.time()-start_time)

    for i in range(len(biasmean)):

        if biasmean[i] != 0:
            clean_headers[i].header['OFFSETCR'] = biasmean[i]

    data_sebip = data_sebip - biasmean[:, None, None]



    # saturated pixels are removed
    # calls function hi_remove_saturation from functions.py
    start_time = time.time()
    data_desat = np.array([functions.hi_remove_saturation(data_sebip[i, :, :], clean_headers[i]) for i in range(len(data_sebip))])
    print("TIME hi_remove_saturation", time.time()-start_time)
    # data_desat = data_sebip.copy()


    dstart1 = [clean_headers[i]['dstart1'] for i in range(len(clean_headers))]
    dstart2 = [clean_headers[i]['dstart2'] for i in range(len(clean_headers))]
    dstop1 = [clean_headers[i]['dstop1'] for i in range(len(clean_headers))]
    dstop2 = [clean_headers[i]['dstop2'] for i in range(len(clean_headers))]

    naxis1 = [clean_headers[i]['naxis1'] for i in range(len(clean_headers))]
    naxis2 = [clean_headers[i]['naxis2'] for i in range(len(clean_headers))]

    exptime = [clean_headers[i]['exptime'] for i in range(len(clean_headers))]
    n_images = [clean_headers[i]['n_images'] for i in range(len(clean_headers))]
    cleartim = [clean_headers[i]['cleartim'] for i in range(len(clean_headers))]
    ro_delay = [clean_headers[i]['ro_delay'] for i in range(len(clean_headers))]
    ipsum = [clean_headers[i]['ipsum'] for i in range(len(clean_headers))]

    rectify = [clean_headers[i]['rectify'] for i in range(len(clean_headers))]
    obsrvtry = [clean_headers[i]['obsrvtry'] for i in range(len(clean_headers))]

    for i in range(len(obsrvtry)):

        if obsrvtry[i] == 'STEREO_A':
            obsrvtry[i] = True

        else:
            obsrvtry[i] = False

    line_ro = [clean_headers[i]['line_ro'] for i in range(len(clean_headers))]
    line_clr = [clean_headers[i]['line_clr'] for i in range(len(clean_headers))]

    header_int = np.array(
        [[dstart1[i], dstart2[i], dstop1[i], dstop2[i], naxis1[i], naxis2[i], n_images[i], post_conj] for i in
            range(len(dstart1))])

    header_flt = np.array(
        [[exptime[i], cleartim[i], ro_delay[i], ipsum[i], line_ro[i], line_clr[i]] for i in range(len(exptime))])

    header_str = np.array([[rectify[i], obsrvtry[i]] for i in range(len(rectify))])

    start_time = time.time()
    data_desm = [functions.hi_desmear(data_desat[i, :, :], header_int[i], header_flt[i], header_str[i]) for i in
                    range(len(data_desat))]
    print("TIME hi_desmear", time.time()-start_time)
    



    data_desm = np.array(data_desm)


    ipkeep = [clean_headers[k]['IPSUM'] for k in range(len(clean_headers))]

    start_time = time.time()
    calimg = [functions.get_calimg('hi_1', ftpsc, clean_headers[k], 'test2/calibration/', post_conj, False) for k in range(len(clean_headers))]
    calimg = np.array(calimg)
    print("TIME get_calimg", time.time()-start_time)

    start_time = time.time()
    calfac = [functions.get_calfac(clean_headers[k], timeavg[k]) for k in range(len(clean_headers))]
    calfac = np.array(calfac)
    print("TIME get_calfac", time.time()-start_time)

    start_time = time.time()
    diffuse = [functions.scc_hi_diffuse(clean_headers[k], ipkeep[k]) for k in range(len(clean_headers))]
    diffuse = np.array(diffuse)
    print("TIME scc_hi_diffuse", time.time()-start_time)

    data_red =  data_desm   * diffuse * calfac[:, None, None]  * calimg
    

    # print('Calibrating pointing...')
    start_time = time.time()
    for i in range(len(clean_headers)):
        functions.hi_fix_pointing(clean_headers[i], 'test2/data' + '/' + 'hi/', ftpsc, 'hi_1', post_conj, silent_point=True)
    print("TIME hi_fix_pointing", time.time()-start_time)



    return data_red,clean_headers




global_type = "science"
global_urls1 = None
global_test = "Test"
def multi_processes_dl(i):
    newpath = "./Raw_data/"+global_test+"/"+global_urls1[i][0].split("_")[0]+"/"+global_type+"/"

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
        
    print("DATE: ", date,"\n \n ")
    if(data_type=="beacon"):
        if(set_type=='train'):
            url1 = 'https://stereo-ssc.nascom.nasa.gov/pub/ins_data/secchi/L0/a/img/hi_1/' + str(date)
        else:
            url1 = 'https://stereo-ssc.nascom.nasa.gov/pub/beacon/ahead/secchi/img/hi_1/' + str(date)        
        urls1 = functions.listfd(url1, 's7h1A.fts')
    else:
        url1 = 'https://stereo-ssc.nascom.nasa.gov/pub/ins_data/secchi/L0/a/img/hi_1/' + str(date)
        urls1 = functions.listfd(url1, 's4h1A.fts')


    newpath = "./Raw_data/"+set_type+"/"+str(date)+"/"+data_type+"/"
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

def create_dates_test():
    Test_events_selected = [ 
                        "03/09/2009",
                        "03/04/2010",
                        "08/04/2010",
                        "23/05/2010",
                        "16/06/2010",
                        "01/08/2010",
                        "15/12/2010",
                        "30/01/2011",
                        "14/02/2011",
                        "23/06/2011",
                        "02/08/2011",
                        "06/09/2011",
                        "22/10/2011",
                        "12/07/2012",
                        "15/01/2013",
                        "30/09/2013",
                        "15/04/2020",
                        "23/06/2020",
                        "09/07/2020",
                        "20/07/2020",
                        "30/09/2020",
                        "26/10/2020",
                        "07/12/2020",
                        "11/02/2021",
                        "20/02/2021",
                        "10/04/2021",
                        "22/04/2021",
                        "09/05/2021",
                        "29/05/2021",
                        "23/08/2021",
                        "13/09/2021",
                        "09/10/2021",
                        "04/04/2022",
                        "11/04/2022",
                        "27/06/2022",
                        "03/07/2022",
                        "03/11/2022",
                        "31/12/2022",
                        "16/04/2023",
                        "21/04/2023"
                    ]

    new_events_list = []
    for i in range(0,len(Test_events_selected)):
        date = datetime.strptime(Test_events_selected[i],'%d/%m/%Y')
        dates = [
                datetime.strptime(Test_events_selected[i],'%d/%m/%Y')-timedelta(days=3),
                datetime.strptime(Test_events_selected[i],'%d/%m/%Y')-timedelta(days=2),
                datetime.strptime(Test_events_selected[i],'%d/%m/%Y')-timedelta(days=1),
                datetime.strptime(Test_events_selected[i],'%d/%m/%Y'),
                datetime.strptime(Test_events_selected[i],'%d/%m/%Y')+timedelta(days=1),
                datetime.strptime(Test_events_selected[i],'%d/%m/%Y')+timedelta(days=2),
                datetime.strptime(Test_events_selected[i],'%d/%m/%Y')+timedelta(days=3)
                ]
        for d in range(0,len(dates)):
            found = False
            for e in new_events_list:
                if(e==d):
                    found = True
            if (found==False):
                new_events_list.append(dates[d])
    return new_events_list


def create_l2(d,type="beacon",typeset="Train"):
    datas   = []
    headers = []
    print("processing date ", d)
    #### load and create background data from fits files of previous 7 days 
    background = [
        d-timedelta(days=7),
        d-timedelta(days=6),
        d-timedelta(days=5),
        d-timedelta(days=4),
        d-timedelta(days=3),
        d-timedelta(days=2),
        d-timedelta(days=1),
    ]
    data_background = []
    header_background = []
    for b in background:
        prefix=str(b.strftime('%Y'))+"-"+str(b.strftime('%m'))+"-"+str(b.strftime('%d'))
        files_list = natsorted(glob.glob('/Volumes/Data_drive/Reduced/'+typeset+'/'+prefix+"/"+type+"/*"))
        for f in files_list:
            filea = fits.open(f)
            data_background.append(filea[0].data)
            header_background.append(filea[0].header)
            filea.close()



    data_background = np.array(data_background)
    background = np.nanmedian(data_background,0)
  
    ##### load the data files and their headers and remove background from the data
    prefix=str(d.strftime('%Y'))+"-"+str(d.strftime('%m'))+"-"+str(d.strftime('%d'))
    files_list = natsorted(glob.glob('/Volumes/Data_drive/Reduced/'+typeset+'/'+prefix+"/"+type+"/*"))
    for f in files_list:
        filea = fits.open(f)
        datas.append(filea[0].data-background)
        headers.append(filea[0].header)
        filea.close()


    for j in range(0,len(datas)):
        name = headers[j]["DATE-END"]
        name = name.replace(":","-")
        fits.writeto("/Volumes/Data_drive/L2_data/"+typeset+"/"+type+"/"+name+".fts", datas[j], headers[j], output_verify='silentfix', overwrite=True)


### global variable I know not good but had to do it quick 
global_dates   = []
global_type    = "science"
global_typeset = "Train"
def multiprocessing_backgroundremoval(i):
    create_l2(global_dates[i],type=global_type,typeset=global_typeset)

def multiprocessing_reduction(i):
    date = global_dates[i]
    da = str(date.year)+str(date.year)+"%02d" % (date.month)+"%02d" % (date.day)
    
    files = natsorted(glob.glob("./WunderbarDataset/"+global_typeset+"/"+da+"/"+global_type+"/*"))
    datas = []
    headers = []
    for f in files:
        filea = fits.open(f)
        datas.append(filea[0].data)
        headers.append(filea[0].header)
        filea.close()

    reduce_data(datas,headers,global_type)


def reduction_dataset():
    test_dates     = create_dates_test()
    training_dates = get_dates_training()


    purged_training_dates = []
    for td in training_dates:
        if(td not in test_dates and td not in purged_training_dates ):
            purged_training_dates.append(td)


    for j in range(0,len(purged_training_dates),4):
        
        # if(purged_training_dates[j]>start_date):
        lstofdates = purged_training_dates[j:j+4]

        global global_dates ,global_type,global_typeset
        global_dates = lstofdates
        global_type = "beacon"
        global_typeset = "Train"

        pool=mp.get_context('fork').Pool(processes=4)
        pool.map(multiprocessing_reduction, np.arange(0,len(lstofdates),1))
        pool.close()
        pool.join()


def L1_to_L2():
    test_dates     = create_dates_test()
    training_dates = get_dates_training()

    start_date = datetime(2011,5,1)

    purged_training_dates = []
    for td in training_dates:
        if(td not in test_dates and td not in purged_training_dates ):
            purged_training_dates.append(td)

    
    for j in range(0,len(purged_training_dates),4):
        
        # if(purged_training_dates[j]>start_date):
        lstofdates = purged_training_dates[j:j+4]

        global global_dates ,global_type,global_typeset
        global_dates = lstofdates
        global_type = "beacon"
        global_typeset = "Train"

        pool=mp.get_context('fork').Pool(processes=4)
        pool.map(multiprocessing_backgroundremoval, np.arange(0,len(lstofdates),1))
        pool.close()
        pool.join()


def normalize_img(img,minstd=2,maxstd=2):

    vmin = np.nanmedian(img) - minstd * np.nanstd(img)
    vmax = np.nanmedian(img) + maxstd * np.nanstd(img)

    img[img<vmin] = vmin
    img[img>vmax] = vmax

    img = (img-vmin)/(vmax-vmin)

    img[img>1.0] = 1.0

    return img

list_imgs   = []
path_to_save = "../Dataset/"
def mp_savepngs(i):
    b = list_imgs[i]
    if(int(b.split("/")[-1][:4])>=2011):
        print(b)
        filea = fits.open(b)
        data   = filea[0].data
        filea.close()
        data = normalize_img(data,1,2.5)
        img = Image.fromarray(data*255.0).convert("L")
        img.save(path_to_save+"training/science/"+b.split("/")[-1][:-3]+".png")

def create_pngs():
    
    beacon_paths  = natsorted(glob.glob("/Volumes/Data_drive/L2_data/Train/beacon/*"))
    science_paths = natsorted(glob.glob("/Volumes/Data_drive/L2_data/Train/science/*"))

    # for b in beacon_paths:
    #     filea = fits.open(b)
    #     data   = filea[0].data
    #     filea.close()
    #     data = normalize_img(data,2.5,2.5)
    #     img = Image.fromarray(data*255.0).convert("L")
    #     img.save(path_to_save+"training/beacon/"+b.split("/")[-1][:-3]+".png")

    global list_imgs 
    list_imgs = science_paths
   

    pool=mp.get_context('fork').Pool(processes=6)
    pool.map(mp_savepngs, np.arange(0,len(science_paths),1))
    pool.close()
    pool.join()

   



def display_data():
    
    beacon_paths = natsorted(glob.glob("/Volumes/Data_drive/L2_data/Train/beacon/2008-12-12T23-*"))
    clahe = cv2.createCLAHE(10,(10,10))
    clahe2 = cv2.createCLAHE(10,(4,4))
    for b in beacon_paths:
        filea = fits.open(b)
        data   = filea[0].data.copy()
        
        raw_beacon0 = filea[0].data
        raw_beacon0 = np.nan_to_num(raw_beacon0,np.nanmedian(raw_beacon0))
        
        raw_beacon1 = clahe.apply( (normalize_img(raw_beacon0.copy(),1,2.5)*255.0).astype(np.uint8))/255.0
        raw_beacon2 = clahe2.apply( (normalize_img(raw_beacon0.copy(),1,2.5)*255.0).astype(np.uint8))/255.0

        raw_beacon = normalize_img(raw_beacon0,2.5,2.5)
        # data2 = clahe.apply((normalize_img(data,2.5,2.5)*255.0).astype(np.uint8))
        # data = clahe.apply((normalize_img(data,1,2.5)*255.0).astype(np.uint8))
        # data2 = normalize_img(data,2.5,2.5)
        # data = normalize_img(data,1,2.5)
        header = filea[0].header
        filea.close()


        name = b.split("/")[-1]
        if(os.path.isfile("/Volumes/Data_drive/L2_data/Train/science/"+name)):
            filea = fits.open("/Volumes/Data_drive/L2_data/Train/science/"+name)
            print(name)
            data_science   = filea[0].data.copy()

            raw_science = filea[0].data
            raw_science = np.nan_to_num(raw_science,np.nanmedian(raw_science))
            
            raw_science1 =  clahe2.apply((normalize_img(raw_science.copy(),1,1)*255.0).astype(np.uint8))/255.0
            raw_science2 = clahe2.apply((normalize_img(raw_science.copy(),1,2.5)*255.0).astype(np.uint8))/255.0
           
            # p2,p98 = np.nanpercentile(raw_science1, (5, 99))
            # raw_science1 = exposure.rescale_intensity(raw_science1, in_range=(p2, p98))
            # raw_science1 = exposure.adjust_log(raw_science1, 1)
            header_science  = filea[0].header
            filea.close()

            # kernel = skimage.morphology.square(21)
            # medianed1 = skimage.filters.median(raw_science1,kernel)
            # deviation = np.abs(raw_science1 - medianed1)

            # raw_science1[raw_science1>2*deviation] = medianed1[raw_science1>2*deviation]
            plt.imshow(raw_science1,cmap='twilight')
            plt.show()
            exit()
            fig,ax = plt.subplots(4,2)
            ax[0][0].imshow(raw_science1,cmap='gist_heat',norm=colors.LogNorm(vmin=0.15,vmax=1,clip=True))
            ax[0][1].imshow(raw_science2,cmap='gist_heat')
            ax[1][0].imshow(raw_beacon1,cmap='gist_heat')
            ax[1][1].imshow(raw_beacon2,cmap='gist_heat')

            

           

            data = np.nan_to_num(data,np.nanmedian(data))
            Y,bins = np.histogram(data.reshape((data.shape[0]*data.shape[1])),bins=1000,range=(np.nanmedian(data) - 2.5 * np.nanstd(data),np.nanmedian(data) + 2.5 * np.nanstd(data)))

            template = np.ones(data.shape)*len(bins)
            for i in range(0,len(bins)-1):
                template[np.logical_and(data>bins[i], data<bins[i+1])] = i  
            template[data>bins[-1]] = i+1  



            data_science = np.nan_to_num(data_science,np.nanmedian(data_science))
            Y2,bins2 = np.histogram(data_science.reshape((data_science.shape[0]*data_science.shape[1])),bins=1000,range=(np.nanmedian(data_science) - 2.5 * np.nanstd(data_science),np.nanmedian(data_science) + 2.5 * np.nanstd(data_science)))
            template2 = np.ones(data_science.shape)*len(bins2)
            for i in range(0,len(bins2)-1):
                template2[np.logical_and(data_science>bins2[i], data_science<bins2[i+1])] = i  

            template2[data_science>bins2[-1]] = i+1  

            ax[3][0].imshow(exposure.equalize_adapthist(template/len(bins),clip_limit=0.2),cmap='coolwarm')
            ax[3][1].imshow(exposure.equalize_adapthist(template2/len(bins2),clip_limit=0.2),cmap='coolwarm') 

            cm = plt.cm.get_cmap('coolwarm')

            x_span = bins.max()-bins.min()
            C = [cm(((x-bins.min())/x_span)) for x in  bins]
            ax[2][0].bar(bins[:-1],Y,color=C,width=bins[1]-bins[0])
            ax[2][0].axvline(x = np.nanmedian(data) - 2.5 * np.nanstd(data), color = 'b', label = 'axvline - full height')
            ax[2][0].axvline(x = np.nanmedian(data) -  np.nanstd(data), color = 'r', label = 'axvline - full height')
            ax[2][0].axvline(x = np.nanmedian(data) -  0.5 * np.nanstd(data), color = 'y', label = 'axvline - full height')
            ax[2][0].axvline(x = np.nanmedian(data) + 2.5 * np.nanstd(data), color = 'b', label = 'axvline - full height')
            ax[2][0].set_ylim([0, Y.max()/100])



            x_span = bins2.max()-bins2.min()
            C = [cm(((x-bins2.min())/x_span)) for x in  bins2]
            ax[2][1].bar(bins2[:-1],Y2,color=C,width=bins2[1]-bins2[0])
            ax[2][1].axvline(x = np.nanmedian(data_science) - 2.5  * np.nanstd(data_science), color = 'b', label = 'axvline - full height')
            ax[2][1].axvline(x = np.nanmedian(data_science) - np.nanstd(data_science), color = 'r', label = 'axvline - full height')
            ax[2][1].axvline(x = np.nanmedian(data_science) - 0.5 * np.nanstd(data_science), color = 'y', label = 'axvline - full height')
            ax[2][1].axvline(x = np.nanmedian(data_science) + 2.5 * np.nanstd(data_science), color = 'b', label = 'axvline - full height')
            ax[2][1].set_ylim([0, Y2.max()/100])

            plt.show()

        

def get_dates_training():
    f = open('helcat.json')
    data = json.load(f)
    data = data["data"]

    dates = []
    for d in data:
        if(d[2]=="A" and d[1].split(" ")[0]):
            dates.append(d[1].split(" ")[0])
    
    dates_final = []
    for i in range(len(dates)):
        dates_final.append(datetime(int(dates[i][:4]),int(dates[i][5:7]),int(dates[i][8:])))
        dates_final.append(datetime(int(dates[i][:4]),int(dates[i][5:7]),int(dates[i][8:]))+ timedelta(days=1))
        dates_final.append(datetime(int(dates[i][:4]),int(dates[i][5:7]),int(dates[i][8:]))+ timedelta(days=2))


    return dates_final



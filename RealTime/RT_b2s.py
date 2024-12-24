import glob 
import data_pipeline
from natsort import natsorted 
import matplotlib.pyplot as plt 
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import shift
from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from sunpy.coordinates import Helioprojective
from sunpy.coordinates.ephemeris import get_horizons_coord
from astropy.coordinates import SkyCoord
from astropy import units as u


from skimage.transform import resize, rotate

import matplotlib
matplotlib.rcParams['backend'] = 'Qt5Agg' 
from datetime import datetime,timedelta
import numpy as np
from skimage import exposure,filters
import torch
import sys
sys.path.insert(0, '..')
import unet2
import functions


def normalize(img,rangev=2.5):      
    vmax = np.median(img)+rangev*np.std(img)
    vmin = np.median(img)-rangev*np.std(img)

    img[img>vmax] = vmax
    img[img<vmin] = vmin

    img = (img-vmin)/(vmax-vmin)

    img[img>1.0] = 1.0
    return img


def ecliptic_cut(data, header, ftpsc, post_conj, datetime_data, datetime_series):
    
    xsize = 256
    ysize = 256
   
    x = np.linspace(0, xsize-1, xsize)
    y = np.linspace(ysize-1, 0, ysize)

    xv, yv = np.meshgrid(x, y)

    wcoord = [WCS(header[i]) for i in range(len(header))]

    dat = [header[i]['DATE-END'] for i in range(len(header))]

    earth = [get_body_heliographic_stonyhurst('earth', dat[i]) for i in [0, -1]]

    if ftpsc == 'A': 
        stereo = get_horizons_coord('STEREO-A', [dat[0], dat[-1]])

    if ftpsc == 'B':
        stereo = get_horizons_coord('STEREO-B', [dat[0], dat[-1]])


    e_hpc = [SkyCoord(earth[i]).transform_to(Helioprojective(observer=stereo[i])) for i in range(len(earth))]
    
    e_x = np.array([e_hpc[i].Tx.to(u.deg).value for i in range(len(e_hpc))])*np.pi/180
    e_y = np.array([e_hpc[i].Ty.to(u.deg).value for i in range(len(e_hpc))])*np.pi/180

    e_x_interp = np.linspace(e_x[0], e_x[1], len(dat))
    e_y_interp = np.linspace(e_y[0], e_y[1], len(dat))

    e_pa = np.arctan2(-np.cos(e_y_interp)*np.sin(e_x_interp), np.sin(e_y_interp))
    
    dif_cut = []
    elongation = []
    
    width_cut = 1
    date_steps = len(datetime_series)

    for i in range(len(wcoord)):
                
        thetax, thetay = wcoord[i].all_pix2world(xv, yv, 0)

        tx = thetax*np.pi/180
        ty = thetay*np.pi/180
        
        pa_reg = np.arctan2(-np.cos(ty)*np.sin(tx), np.sin(ty))
        elon_reg = np.arctan2(np.sqrt((np.cos(ty)**2)*(np.sin(tx)**2)+(np.sin(ty)**2)), np.cos(ty)*np.cos(tx))

        pa_reg = resize(pa_reg,(512,512))
        elon_reg = resize(elon_reg,(512,512))
        
        delta_pa = e_pa[i]

        e_val = [(delta_pa)-1*np.pi/180, (delta_pa)+1*np.pi/180]

    

        if ftpsc == 'A':
            farside = -1 if post_conj else 0

        if ftpsc == 'B':
            farside = 0 if post_conj else -1

        data_rot = rotate(data[i], -delta_pa, preserve_range=True, mode='constant',order=0, cval=np.median(data[i]))
        elon_rot = rotate(elon_reg, -delta_pa, preserve_range=True, mode='edge')
        pa_rot = rotate(pa_reg, -delta_pa, preserve_range=True, mode='edge')

        

        farside_ids = np.argwhere(
                                    np.logical_and(
                                                    (pa_rot[:, farside].flatten() >= min(e_val)),
                                                    (pa_rot[:,farside].flatten() <= max(e_val))
                                    )
                        )
        

        farside_ids = farside_ids.flatten()
        min_id_farside = min(farside_ids)

        if i == 0:
            max_id_farside = max(farside_ids)            
            width_cut = max_id_farside - min_id_farside
            dif_cut = np.zeros((512,date_steps, width_cut))
            el_cut  = np.zeros((512,date_steps, width_cut))
            dif_cut[:] = np.nan
            el_cut[:] = np.nan
            arr_ind = 0

        else:
            max_id_farside = min_id_farside + width_cut
            arr_ind = (np.abs(datetime_series - datetime_data[i])).argmin()
            
        diff_slice = data_rot[min_id_farside:max_id_farside, :]

        dif_cut[:,arr_ind,:] = diff_slice.T
        el_cut[:,arr_ind,:]  = elon_rot[min_id_farside:max_id_farside, :].T*180/np.pi
          

    return dif_cut, el_cut





def create_jplot_from_differences(differences,headers,cadence=120):
    
    date1 = datetime.strptime(headers[0]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')


    if(date1.year>2015 and date1.year<2023):
        postconj= True
    else:
        postconj = False


    datetime_data = [datetime.strptime(t['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f') for t in headers]
    datetime_series = np.arange(np.min(datetime_data), np.max(datetime_data) + timedelta(minutes=cadence), timedelta(minutes=cadence)).astype(datetime)
    cuts,elongations = functions.ecliptic_cut(differences,headers,"beacon",'A',postconj,datetime_data,datetime_series,"no_median")

    cuts = np.where(np.isnan(cuts), np.nanmedian(cuts), cuts)
    elongations = np.abs(elongations)
   
    return cuts,datetime_series,elongations

def create_enhanced_jplots(differences,headers,cadence=120):
    date1 = datetime.strptime(headers[0]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')


    if(date1.year>2015 and date1.year<2023):
        postconj= True
    else:
        postconj = False


    datetime_data = [datetime.strptime(t['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f') for t in headers]
    datetime_series = np.arange(np.min(datetime_data), np.max(datetime_data) + timedelta(minutes=cadence), timedelta(minutes=cadence)).astype(datetime)
    cuts,elongations = ecliptic_cut(differences,headers,"A",postconj,datetime_data,datetime_series)

    cuts = np.where(np.isnan(cuts), np.nanmedian(cuts), cuts)
    elongations = np.abs(elongations)
   
    return cuts,datetime_series,elongations


def processjplot(cuts,dates,elongations,medianed=False):
    if(medianed):
        cuts = np.median(cuts,2)
    else:
        a,b,c = cuts.shape
        cuts  = cuts.reshape((a,b*c))
    
    p2, p98 = np.nanpercentile(cuts, (2, 98))
    cuts = exposure.rescale_intensity(cuts, in_range=(p2, p98))
    cuts = np.where(np.isnan(cuts), np.nanmedian(cuts), cuts)
    vmin = np.nanmedian(cuts) - 2 *  np.nanstd(cuts)
    vmax = np.nanmedian(cuts) + 2 *  np.nanstd(cuts)

    elongations = np.asarray(elongations)
    elongations = [np.nanmin(elongations), np.nanmax(elongations)]

    return cuts,vmin,vmax,elongations

typeset= "forecast"
type = "beacon"
device = torch.device("cpu")
if(torch.backends.mps.is_available()):
    device = torch.device("mps")
elif(torch.cuda.is_available()):
    device = torch.device("cuda:0")




dates = data_pipeline.get_x_last_days(7)
dates = dates[5:]

datas   = []
headers = []
for d in dates:
    path = "L2_data/"+typeset+"/"+type+"/"+d+"/*"
    files = natsorted(glob.glob(path))
    for f in files:
        filea  = fits.open(f)
        data   = filea[0].data
        header = filea[0].header
        filea.close()
        datas.append(data)
        headers.append(header)

maxgap  = -3.5
cadence = 120

diffs = []
nonprocesseddiffs = []
headers2= []
for i in range(1,len(datas)-1):
    time1 = datetime.strptime(headers[i-1]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')
    time2 = datetime.strptime(headers[i]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')

    if( np.abs((time2-time1).total_seconds()/60.0)<= -maxgap * cadence and np.abs((time2-time1).total_seconds()/60.0) >= (cadence-5)):

        im1 = np.float32(datas[i-1])
        nan_mask = np.isnan(im1)
        im1[nan_mask] = np.array(np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), im1[~nan_mask]))


        im2 = np.float32(datas[i])
        nan_mask = np.isnan(im2)
        im2[nan_mask] = np.array(np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), im2[~nan_mask]))

        hdr = headers[i-1]
        hdr2 = headers[i]


        center = hdr2['crpix1']-1, hdr2['crpix2']-1
        wcs = WCS(hdr2,key='A')
        center_prev = wcs.all_world2pix(hdr["crval1a"],hdr["crval2a"], 0)
        shift_arr = np.array([center_prev[1]-center[1],center_prev[0]-center[0]])
        shift_arr = shift_arr


        diff = np.float32(im2-shift(im1,shift_arr, mode='nearest'))
        nonprocesseddiffs.append(diff.copy())

        diff = exposure.equalize_adapthist(normalize(diff),clip_limit=0.02,kernel_size=diff.shape[0]//10)

        diffs.append(diff)
        headers2.append(hdr2)

model = unet2.ResUnet(1,full_size=512)
dict_gen  = torch.load("gan_gen_l13.pth",map_location=torch.device('cpu'))
model.load_state_dict(dict_gen)
model.to(device)

enhanced = []
with torch.no_grad():
    for diff in diffs:


        D2 = torch.tensor(diff).float().unsqueeze(0).unsqueeze(1).to(device)

        sr = model(D2,None)
        sr = sr[0,0,:,:].cpu().numpy()

        enhanced.append(sr)


cuts_beacon,dates_beacon,elongations_beacon = create_jplot_from_differences(nonprocesseddiffs,headers2,120)
cuts,dates,elongations = create_enhanced_jplots(enhanced,headers2,120)


cuts,vmin,vmax,elongations_beacon = processjplot(cuts,dates,elongations,True)
cuts_beacon,vmin_beacon,vmax_beacon,elongations = processjplot(cuts_beacon,dates_beacon,elongations_beacon,True)



fig,ax = plt.subplots(2,1,figsize=(20,10))
ax[0].imshow(cuts_beacon, cmap='gray', aspect='auto',interpolation='none',origin='upper', extent=[dates_beacon[0], dates_beacon[-1],elongations_beacon[0] , elongations_beacon[1]],vmin=vmin_beacon,vmax=vmax_beacon)
ax[0].title.set_text('Beacon JPlot')
ax[1].imshow(cuts, cmap='gray', aspect='auto',interpolation='none',origin='upper', extent=[dates[0], dates[-1],elongations[0] , elongations[1]],vmin=vmin,vmax=vmax)
ax[1].title.set_text('Enhanced Beacon JPlot')
now  = datetime.now()
plt.savefig(str(now.year)+str('%02d' % now.month)+str('%02d' % now.day)+".png")


        

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
from PIL import Image
from PIL import ImageDraw 
from PIL import ImageFont

from skimage.transform import resize, rotate

import matplotlib
# matplotlib.rcParams['backend'] = 'Qt5Agg' 
from datetime import datetime,timedelta
import numpy as np
from skimage import exposure,filters
import torch
import sys
sys.path.insert(0, '..')
import models.unet2 as unet2
import models.RIFE as RIFE

import functions
import os 
import pickle

plt.rcParams.update({'font.size': 20})

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



contrast = 4.0

def create_jplot_from_differences(differences,headers,cadence=120):
    
    date1 = datetime.strptime(headers[0]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')


    if(date1.year>2015 and date1.year<2023):
        postconj= True
    else:
        postconj = False


    for i in range(0,len(differences)):
        dif = normalize(differences[i],2.5)
        intercept = -(0.5 * contrast) + 0.5
        dif   = contrast * dif  + intercept

        dif = np.where(dif > 1,1,dif)
        dif = np.where(dif < 0,0,dif)
        differences[i] = dif


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

    for i in range(0,len(differences)):
        dif = differences[i]
        
        intercept = -(0.5 * contrast) + 0.5
        dif   = contrast * dif  + intercept

        dif = np.where(dif > 1,1,dif)
        dif = np.where(dif < 0,0,dif)
        differences[i] = dif



    datetime_data = [datetime.strptime(t['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f') for t in headers]
    datetime_series = np.arange(np.min(datetime_data), np.max(datetime_data) + timedelta(minutes=cadence), timedelta(minutes=cadence)).astype(datetime)
    cuts,elongations = ecliptic_cut(differences,headers,"A",postconj,datetime_data,datetime_series)

    cuts = np.where(np.isnan(cuts), np.nanmedian(cuts), cuts)
    elongations = np.abs(elongations)
   
    return cuts,datetime_series,elongations

def resistant_mean(inputData, Cut=3.0, axis=None, dtype=None):
    """
    Robust estimator of the mean of a data set.  Based on the
    resistant_mean function from the AstroIDL User's Library.

    .. versionchanged:: 1.0.3
        Added the 'axis' and 'dtype' keywords to make this function more
        compatible with numpy.mean()
    """
    epsilon = 1.0e-20
    if axis is not None:
        fnc = lambda x: resistant_mean(x, dtype=dtype)
        dataMean = np.apply_along_axis(fnc, axis, inputData)
    else:
        data = inputData.ravel()
        if type(data).__name__ == "MaskedArray":
            data = data.compressed()
        if dtype is not None:
            data = data.astype(dtype)

        data0 = np.nanmedian(data)
        maxAbsDev = np.nanmedian(np.abs(data - data0)) / 0.6745
        if maxAbsDev < epsilon:
            maxAbsDev = np.nanmean(np.abs(data - data0)) / 0.8000

        cutOff = Cut * maxAbsDev
        good = np.where(np.abs(data - data0) <= cutOff)
        good = good[0]
        dataMean = np.nanmean(data[good])
        dataSigma = np.sqrt(np.nansum((data[good] - dataMean) ** 2.0) / len(good))

        if Cut > 1.0:
            sigmaCut = Cut
        else:
            sigmaCut = 1.0
        if sigmaCut <= 4.5:
            dataSigma = dataSigma / (
            -0.15405 + 0.90723 * sigmaCut - 0.23584 * sigmaCut ** 2.0 + 0.020142 * sigmaCut ** 3.0)

        cutOff = Cut * dataSigma
        good = np.where(np.abs(data - data0) <= cutOff)
        good = good[0]
        dataMean = np.nanmean(data[good])
        if len(good) > 3:
            dataSigma = np.sqrt(np.nansum((data[good] - dataMean) ** 2.0) / len(good))

        if Cut > 1.0:
            sigmaCut = Cut
        else:
            sigmaCut = 1.0
        if sigmaCut <= 4.5:
            dataSigma = dataSigma / (
            -0.15405 + 0.90723 * sigmaCut - 0.23584 * sigmaCut ** 2.0 + 0.020142 * sigmaCut ** 3.0)

        dataSigma = dataSigma / np.sqrt(len(good) - 1)

    return dataMean


def processjplot(cuts,dates,elongations,medianed=False):
    # if(medianed):
    # cuts = np.median(cuts,2)
    cuts = resistant_mean(cuts,axis=2)
    # else:
        # a,b,c = cuts.shape
        # cuts  = cuts.reshape((a,b*c))
    
    # p2, p98 = np.nanpercentile(cuts, (2, 98))
    # cuts = exposure.rescale_intensity(cuts, in_range=(p2, p98))


    cuts = np.where(np.isnan(cuts), np.nanmedian(cuts), cuts)
    vmin = np.nanmedian(cuts) - 2.7 *  np.nanstd(cuts)
    vmax = np.nanmedian(cuts) + 3.0 *  np.nanstd(cuts)

    elongations = np.asarray(elongations)
    elongations = [np.nanmin(elongations), np.nanmax(elongations)]

    return cuts,vmin,vmax,elongations


def enhance_latest():
    typeset= "forecast"
    type = "beacon"
    device = torch.device("cpu")
    if(torch.backends.mps.is_available()):
        device = torch.device("mps")
    elif(torch.cuda.is_available()):
        device = torch.device("cuda:0")

    now  = datetime.now()




    dates = data_pipeline.get_x_last_days(7)
    dates = dates[5:]


    datas   = []
    headers = []
    for d in dates:
        path = "/scratch/aswo/jlelouedec/L2_data/"+typeset+"/"+type+"/"+d+"/*"
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
    times = []
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

            diff = normalize(diff)

            diffs.append(diff)
            headers2.append(hdr2)
            times.append(time2)

    
    cuts_beacon,dates_beacon,elongations_beacon = create_jplot_from_differences(nonprocesseddiffs,headers2,120)
    # dict_beacon = {
    #         'data':cuts_beacon,
    #         'dates':dates_beacon,
    #         'elongations':elongations_beacon
    # }
    # pickle.dump(dict_beacon, open("latest_jplot_beacon.p", "wb"))  # save it into a file named save.p
    # pickle.dump(dict_beacon, open("/perm/aswo/jlelouedec/beacon2science/"+str(now.year)+str('%02d' % now.month)+str('%02d' % now.day)+"_jplot_beacon.p", "wb"))  # save it into a file named save.p
    # pickle.dump(dict_beacon, open("/perm/aswo/jlelouedec/beacon2science/latest_jplot_beacon.p", "wb"))  # save it into a file named save.p


    cuts_beacon,vmin_beacon,vmax_beacon,elongations_beacon = processjplot(cuts_beacon,dates_beacon,elongations_beacon,False)

    ###########################################
    ###########################################
    ### We apply the Fist neural network 

    model = unet2.ResUnet(1,full_size=512)
    dict_gen  = torch.load("PAPERNN1.pth",map_location=torch.device('cpu'))
    model.load_state_dict(dict_gen)
    model.to(device)

    enhanced = []
    with torch.no_grad():
        for diff in diffs:


            D2 = torch.tensor(diff).float().unsqueeze(0).unsqueeze(1).to(device)

            sr = model(D2,None)
            sr = sr[0,0,:,:].cpu().numpy()

            enhanced.append(sr)

    


    cuts,dates,elongations = create_enhanced_jplots(enhanced,headers2,120)
    # dict_enhanced = {
    #         'data':cuts,
    #         'dates':dates,
    #         'elongations':elongations
    # }
    # pickle.dump(dict_enhanced, open("latest_jplot_enhance.p", "wb"))  # save it into a file named save.p

    cuts,vmin,vmax,elongations = processjplot(cuts,dates,elongations,False)

    
    ###########################################
    ###########################################
    ### We apply the second neural network 
    model = RIFE.Model()
    model.flownet.to(device)

    model.load_model("PAPER_NN2.pth")

    interpolated_rdifs = []
    interpolated_headers = []

    with torch.no_grad():
        for p in range(0,len(enhanced)-1,1):
            

            time1 = times[p]
            time2 = times[p+1]
            diff_time = (time2-time1).total_seconds()

            if(diff_time/3600<=4.0):
                
                data1 = enhanced[p]
                data2 = enhanced[p+1]

                if(time1.year<=2015):
                    data1 = np.fliplr(data1)
                    data2 = np.fliplr(data2)


                S1 = torch.tensor(data1.copy()).float().unsqueeze(0).unsqueeze(1).to(device)
                S2 = torch.tensor(data2.copy()).float().unsqueeze(0).unsqueeze(1).to(device)


                output1 = model.inference(S1,S2,timestep=torch.tensor([0.33]))
                output2 = model.inference(S1,S2,timestep=torch.tensor([0.66]))
            

                output1 = output1[0,0,:,:].cpu().numpy()
                output2 = output2[0,0,:,:].cpu().numpy()

               
                output1 = exposure.match_histograms(output1, data1)
                output2 = exposure.match_histograms(output2, data1)

                if(time1.year<=2015):
                    data1 = np.fliplr(data1)
                    data2 = np.fliplr(data2)
                    output1 = np.fliplr(output1)
                    output2 = np.fliplr(output2)

                header1 = headers2[p]
                header2 = headers2[p+1]

                hdr3 = header2.copy()
                hdr4 = header2.copy()

                timeout1 = time1+timedelta(minutes=40)
                timeout2 = time1+timedelta(minutes=80)
                
                hdr3["DATE-END"] = timeout1.strftime('%Y-%m-%dT%H:%M:%S')+".000"
                hdr4["DATE-END"] = timeout2.strftime('%Y-%m-%dT%H:%M:%S')+".000"


                interpolated_rdifs = interpolated_rdifs + [data1,output1,output2]
                interpolated_headers = interpolated_headers + [header1,hdr3,hdr4]
            else:
                interpolated_rdifs = interpolated_rdifs + [enhanced[p]]
                interpolated_headers = interpolated_headers + [headers2[p]]

    interpolated_rdifs = interpolated_rdifs + [enhanced[p+1]]
    interpolated_headers = interpolated_headers + [headers2[p+1]]


    # for i in range(0,len(diffs)):
    #     dif = diffs[i]
    #     intercept = -(0.5 * contrast) + 0.5
    #     dif   = contrast * dif  + intercept

    #     dif = np.where(dif > 1,1,dif)
    #     dif = np.where(dif < 0,0,dif)

    #     img = Image.fromarray(np.flipud(dif)*255.0).convert("L")
    #     draw = ImageDraw.Draw(img)
    #     font = ImageFont.truetype("SourceSansPro-Bold.otf",17)
    #     draw.text((10, 20),headers2[i]["DATE-END"].replace("T"," ")[:-4],font=font, fill=255)
    #     img.save("tmp/"+str(i)+".png")
    
    # os.system("ffmpeg -y -framerate 5 -i tmp/%d.png -pix_fmt rgb24 hi1_beacon_current.mp4")
    # os.system("rm -rf tmp/*")

    time1 = datetime.strptime(headers[i-1]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')
    time2 = datetime.strptime(headers[i]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')



    index_beacon = 0

    for h in headers2:
        print(h["DATE-END"])


    for i in range(0,len(interpolated_rdifs)):
        dif        = interpolated_rdifs[i]
        time = datetime.strptime(interpolated_headers[i]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')
        time2 = datetime.strptime(headers2[index_beacon+1]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')
        if(np.abs(time-time2).total_seconds()/60 <10.0):
            index_beacon = index_beacon + 1

        dif_beacon = diffs[index_beacon]
        time_beacon = headers2[index_beacon]["DATE-END"]


        intercept = -(0.5 * contrast) + 0.5
        dif   = contrast * dif  + intercept
        dif = np.where(dif > 1,1,dif)
        dif = np.where(dif < 0,0,dif)
        img = Image.fromarray(np.flipud(dif)*255.0).convert("L")


        dif_beacon   = contrast * dif_beacon  + intercept
        dif_beacon = np.where(dif_beacon > 1,1,dif_beacon)
        dif_beacon = np.where(dif_beacon < 0,0,dif_beacon)
        dif_beacon = resize(dif_beacon,(512,512))
        img_beacon = Image.fromarray(np.flipud(dif_beacon)*255.0).convert("L")

        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("SourceSansPro-Bold.otf",35)
        draw.text((10, 20),interpolated_headers[i]["DATE-END"].replace("T"," ")[:-4],font=font, fill=255)

        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("SourceSansPro-Bold.otf",25)
        draw.text((10, 470),"Beacon2Science",font=font, fill=255)

        draw2 = ImageDraw.Draw(img_beacon)
        font = ImageFont.truetype("SourceSansPro-Bold.otf",35)
        draw2.text((10, 20),headers2[index_beacon]["DATE-END"].replace("T"," ")[:-4],font=font, fill=255)

        draw2 = ImageDraw.Draw(img_beacon)
        font = ImageFont.truetype("SourceSansPro-Bold.otf",25)
        draw2.text((10, 470),"Beacon",font=font, fill=255)

       

        img_total = Image.fromarray(np.hstack([img_beacon,np.asarray(img)])).convert("L")


        img_total.save("tmp/"+str(i)+".png", dpi=(1000, 1000))
    
    os.system("ffmpeg -y -framerate 10 -i tmp/%d.png -pix_fmt yuv420p -vb 20M /perm/aswo/ops/hi/hi1_current_interpolated.mp4")
    os.system("rm -rf tmp/*")
    os.system('ffmpeg -y -i /perm/aswo/ops/hi/hi1_current_interpolated.mp4 -filter_complex "fps=9,scale=350:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=35[p];[s1][p]paletteuse=dither=bayer" /perm/aswo/ops/hi/hi1_current_interpolated.gif')

                
    cuts_interpolated,dates_interpolated,elongations_interpolated = create_enhanced_jplots(interpolated_rdifs,interpolated_headers,40)
    dict_interpolated = {
            'data':cuts_interpolated,
            'dates':dates_interpolated,
            'elongations':elongations_interpolated
    }
    pickle.dump(dict_interpolated, open("latest_jplot_enhance.p", "wb"))  # save it into a file named save.p
    pickle.dump(dict_interpolated, open("/perm/aswo/ops/hi/"+str(now.year)+str('%02d' % now.month)+str('%02d' % now.day)+"_jplot_interpolated.p", "wb"))  # save it into a file named save.p
    pickle.dump(dict_interpolated, open("/perm/aswo/ops/hi/latest_jplot_interpolated.p", "wb"))  # save it into a file named save.p

    cuts_interpolated,vmin_interpolated,vmax_interpolated,elongations_interpolated = processjplot(cuts_interpolated,dates_interpolated,elongations_interpolated,False)
    
    






    fig,ax = plt.subplots(2,1,figsize=(20,10))
    ax[0].imshow(cuts_beacon.astype(np.float32), cmap='gray', aspect='auto',interpolation='none',origin='upper', extent=[np.datetime64(dates_beacon[0]), np.datetime64(dates_beacon[-1]) ,elongations_beacon[0].astype(np.float32) , elongations_beacon[1].astype(np.float32)],vmin=vmin_beacon.astype(np.float32),vmax=vmax_beacon.astype(np.float32))
    ax[0].title.set_text('Beacon JPlot')
    # ax[1].imshow(cuts.astype(np.float32), cmap='gray', aspect='auto',interpolation='none',origin='upper', extent=[np.datetime64(dates[0]), np.datetime64(dates[-1]),elongations[0] , elongations[1]],vmin=vmin,vmax=vmax)
    # ax[1].title.set_text('Enhanced Beacon JPlot')
    ax[1].imshow(cuts_interpolated.astype(np.float32), cmap='gray', aspect='auto',interpolation='none',origin='upper', extent=[np.datetime64(dates_interpolated[0]), np.datetime64(dates_interpolated[-1]),elongations_interpolated[0] , elongations_interpolated[1]],vmin=vmin_interpolated,vmax=vmax_interpolated)
    ax[1].title.set_text('Interpolated Enhanced Beacon JPlot')
    
    # plt.savefig(str(now.year)+str('%02d' % now.month)+str('%02d' % now.day)+".png")
    # plt.figtext(0.05,0.00, "Le Louëdec, Justin et al., 2025", fontsize=8, va="top", ha="left")
    ax[1].text(0.00,-0.20, 'Le Louëdec, Justin et al., 2025',  color='black', fontsize=15, style='italic', horizontalalignment='left',verticalalignment='top', transform=ax[1].transAxes)
    # plt.subplots_adjust(bottom=0.01, right=0.8, top=0.9)
    plt.savefig("/perm/aswo/ops/hi/latest.png")
    plt.savefig("/perm/aswo/ops/hi/"+str(now.year)+str('%02d' % now.month)+str('%02d' % now.day)+".png")


        
if __name__ == '__main__':
    data_pipeline.run_all()	
    enhance_latest()
from natsort import natsorted
import glob 
from datetime import datetime,timedelta
import numpy as np
from PIL import Image,  ImageTransform
from astropy.io import fits

from astropy.wcs import WCS
from scipy.ndimage import shift
import matplotlib.pyplot as plt 
from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from sunpy.coordinates import Helioprojective
from sunpy.coordinates.ephemeris import get_horizons_coord
from astropy.coordinates import SkyCoord

from astropy import units as u
from skimage.transform import  resize,rotate
from skimage import exposure
import skimage
from skimage.draw import line
import matplotlib.dates as mdates
import cv2
import os 
from matplotlib.ticker import MultipleLocator
import matplotlib
# matplotlib.rcParams['backend'] = 'Qt5Agg' 




def rotate_via_numpy(xy, radians,center):
    """Use numpy to build a rotation matrix and take the dot product."""
    x, y = xy
    xc,yc = center
    x= x-xc
    y= y-yc

    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, [x, y])

    x= float(m.T[0])+xc
    y= float(m.T[1])+yc

    return x, y


def insert_cut(dates,dtime,diff,cuts,header,size):
    wcoord = WCS(header)
    # diff  = (diff-diff.min())/(diff.max()-diff.min())
    for i,d in enumerate(dates):
        if(np.abs((d-dtime).total_seconds()/60.0)<20.0):

            earth = get_body_heliographic_stonyhurst('earth', dtime)
            stereo = get_horizons_coord('STEREO-A', dtime)
            e_hpc = SkyCoord(earth).transform_to(Helioprojective(observer=stereo))
            
            e_x = e_hpc.Tx.to(u.deg).value *np.pi/180
            e_y = e_hpc.Ty.to(u.deg).value *np.pi/180
            e_pa = np.arctan2(-np.cos(e_y)*np.sin(e_x), np.sin(e_y))
            delta_pa = e_pa
            e_val = [(delta_pa)-1*np.pi/180, (delta_pa)+1*np.pi/180]


            if(size==512):
                newsize = 256
            else:
                newsize = size
            x = np.linspace(0, newsize-1, newsize)
            y = np.linspace(newsize-1, 0, newsize)
            xv, yv = np.meshgrid(x, y)
            thetax, thetay = wcoord.all_pix2world(xv, yv, 0)
            tx = thetax*np.pi/180
            ty = thetay*np.pi/180
            pa_reg = np.arctan2(-np.cos(ty)*np.sin(tx), np.sin(ty))

            elon_reg = np.arctan2(np.sqrt((np.cos(ty)**2)*(np.sin(tx)**2)+(np.sin(ty)**2)), np.cos(ty)*np.cos(tx))


            elon_reg = resize(elon_reg, (size,size), anti_aliasing=True)           
            pa_reg = resize(pa_reg, (size,size), anti_aliasing=True)
         
      
            diff_begin = diff.copy()
            diff_begin = np.repeat(diff_begin[:,:,None],3,2)

            minid = np.abs(pa_reg[:,0]-min(e_val))
            minid = np.argmin(minid)

            maxid = np.abs(pa_reg[:,0]-max(e_val))
            maxid = np.argmin(maxid)
           

            minid2 = np.abs(pa_reg[:,-1]-min(e_val))
            minid2 = np.argmin(minid2)

            maxid2 = np.abs(pa_reg[:,-1]-max(e_val))
            maxid2 = np.argmin(maxid2)

            if(dtime.year >2015):
                tmp = minid
                minid = maxid
                maxid = tmp

                tmp = minid2
                minid2 = maxid2
                maxid2 = tmp

            size1 = maxid-minid
            center1 = minid + int(size1/2)


            size2 = maxid2-minid2
            center2 = minid2 + int(size2/2)
            

            #### displaying lines for the ecliptic in green
            rr, cc = line(maxid,0,maxid2,size-1)
            diff_begin[rr,cc] = [0.0,1.0,0.0]

            rr, cc = line(minid,0,minid2,size-1)
            diff_begin[rr,cc] = [0.0,1.0,0.0]



           
            if(dtime.year <=2015):
                minid2 = center2 - int(size1/2)
                maxid2 = center2 + int(size1/2)
            else:
                minid = center1 - int(size2/2)
                maxid = center1 + int(size2/2)

            

            ### displaying lines for larger size of ecliptic cone 
            rr, cc = line(minid,0,minid2,size-1)
            diff_begin[rr,cc] = [1.0,0.0,0.0]

            rr, cc = line(maxid,0,maxid2,size-1)
            diff_begin[rr,cc] = [1.0,0.0,0.0]




            # mask = np.zeros(diff.shape)
            rr, cc = skimage.draw.polygon([minid,maxid,maxid2,minid2],[0,0,size-1,size-1])
            # mask[rr,cc] = 1.0

            if(dtime.year<=2015):
                angle = np.arctan2(minid2-minid,size-1)*180/np.pi
            else:
                angle = np.arctan2(minid-minid2,size-1)*180/np.pi
            
            diff     = rotate(diff,angle,preserve_range=True)
            elon_reg = rotate(elon_reg,angle,preserve_range=True)


            # mask = rotate(mask,angle)
            minid,maxid = rotate_via_numpy((0,minid),angle*np.pi/180,(size//2,size//2))
            maxid = int(maxid)



            diff = np.repeat(diff[:,:,None],3,2)
            diff2 = np.zeros(diff.shape)
           
            if(dtime.year<=2015):
                data_med = resize(diff[maxid:maxid+size1,:,0].T,(size,cuts.shape[2]))
                elon_med = resize(elon_reg[maxid:maxid+size1,:].T,(size,cuts.shape[2]))
                diff2[maxid:maxid+size1,:] = diff[maxid:maxid+size1,:]
            else:
                data_med = resize(diff[maxid:maxid+size2,:,0].T,(size,cuts.shape[2]))
                elon_med = resize(elon_reg[maxid:maxid+size2,:].T,(size,cuts.shape[2]))
                diff2[maxid:maxid+size2,:] = diff[maxid:maxid+size2,:]


            rr, cc = line(maxid,0,maxid,size-1)
            diff[rr,cc] = [1.0,0.0,0.0]

            if(dtime.year <=2015):
                rr, cc = line(maxid+size1,0,maxid+size1,size-1)
                diff[rr,cc] = [1.0,0.0,0.0]
            else:
                rr, cc = line(maxid+size2,0,maxid+size2,size-1)
                diff[rr,cc] = [1.0,0.0,0.0]

                 
        
            
            # fig,ax = plt.subplots(1,3)
            # ax[0].imshow(diff_begin,cmap='gray')
            # ax[1].imshow(diff,cmap='gray')
            # ax[2].imshow(diff2,cmap='gray')
            # plt.show()
            
            # data_mask = np.where((pa_reg > min(e_val)) & (pa_reg < max(e_val)), diff, np.nan)
           

            # data_med = np.nanmedian(data_mask, 0)
          
            # elon_mask = np.where((pa_reg > min(e_val)) & (pa_reg < max(e_val)), elon_reg, np.nan)


            cuts[:,i,:]= data_med
            break
    return cuts,[np.nanmin(elon_med*180/np.pi),np.nanmax(elon_med*180/np.pi)]





def compute_running_difference(datas,headers,shift_multiplier,median=True):
    differences = []
    headers_differences = []

    if(datas[0].shape[0]==256):
        kernel = 3
    else:
        kernel = 5 
    for i in range(1,len(datas)):
        time1 = datetime.strptime(headers[i-1]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')
        time2 = datetime.strptime(headers[i]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')
        if( ((time2-time1).total_seconds()/60.0)<400.0):
            im1 = datas[i-1]
            im2 = datas[i]

            head1 = headers[i-1]
            head2 = headers[i]

            print(head1["DATE-END"],head2["DATE-END"])
            print(head1["crval1a"],head1["crval2a"])


            center = [head2['crpix1']-1,head2['crpix2']-1]
            wcs = WCS(head2,key='A')
            center_prev = wcs.all_world2pix(head1["crval1a"],head1["crval2a"], 0)
            shift_arr = np.array([center_prev[1]-center[1],center_prev[0]-center[0]])


            if (median):
                diff = cv2.medianBlur(np.float32(im2-shift(im1,shift_arr*shift_multiplier, mode='nearest',prefilter=False)), kernel)
            else :
                diff = np.float32(im2-shift(im1,shift_arr*shift_multiplier, mode='nearest',prefilter=False))
            differences.append(diff)
            headers_differences.append(head2)

    return differences,headers_differences
            
def create_jplot_from_differences(differences,headers,cadence=120):
    date1 = datetime.strptime(headers[0]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')
    date2  = datetime.strptime(headers[-1]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')


    dates2 = []
    current_date = date1
    while current_date <= date2:
        dates2.append(current_date)
        current_date = current_date + timedelta(minutes=cadence)
    dates2.append(date2)


    

    if(differences[0].shape[0]==1024 or differences[0].shape[0]==512):
        size = 64
    else:
        size = 32
    cuts = np.zeros((differences[0].shape[0],len(dates2),size))
    elongations = []
    for i in range(0,len(differences)):
        cuts,elong = insert_cut(dates2,datetime.strptime(headers[i]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f'),differences[i],cuts,headers[i],size=differences[0].shape[0])
        elongations.append(elong)
    cuts = np.nan_to_num(cuts,np.nanmedian(cuts))
    return cuts,dates2,elongations


def create_l2_beacon(dates):
    datas   = []
    headers = []
    cnt = 0
    for d in dates:
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
            files_list = natsorted(glob.glob('/Volumes/Data_drive/Reduced/Test/'+prefix+"/beacon/*"))
            for f in files_list:
                filea = fits.open(f)
                data_background.append(filea[0].data)
                header_background.append(filea[0].header)
                filea.close()

        if(cnt==0):
            datas.append(data_background[-1])
            headers.append(header_background[-1])

        data_background = np.array(data_background)
        background = np.nanmedian(data_background,0)
        
        if(cnt==0):
            datas[0] = datas[0]-background

        

        ##### load the data files and their headers and remove background from the data
        prefix=str(d.strftime('%Y'))+"-"+str(d.strftime('%m'))+"-"+str(d.strftime('%d'))
        files_list = natsorted(glob.glob('/Volumes/Data_drive/Reduced/Test/'+prefix+"/beacon/*"))
        for f in files_list:
            filea = fits.open(f)
            datas.append(filea[0].data-background)
            headers.append(filea[0].header)
            filea.close()

        cnt+=1
    return datas,headers

def load_enhanced_beacon(dates,path='/Volumes/Data_drive/res_model_final3/'):
    imgs_list    = []
    imgs_headers = []
    for e in dates:
        prefix=str(e.strftime('%Y'))+"-"+str(e.strftime('%m'))+"-"+str(e.strftime('%d'))
        imgs = natsorted(glob.glob(path+prefix+"*"))
        for j,im in enumerate(imgs):
            name = '/Volumes/Data_drive/Reduced/Test/'+prefix+"/beacon/"+im.split("/")[-1][:-3]+"fts"
            if(os.path.exists(name) ):
                filea = fits.open(name)
                imgs_headers.append(filea[0].header)
                filea.close()
            else:
                imgs_headers.append(None)
            imgs_list.append(np.asarray(Image.open(im).convert("L"))/255.0)

    return imgs_list,imgs_headers

def load_final_enhanced(dates,path):
    imgs_list    = []
    imgs_headers = []
    for e in dates:
        prefix=str(e.strftime('%Y'))+"-"+str(e.strftime('%m'))+"-"+str(e.strftime('%d'))
        imgs = natsorted(glob.glob(path+prefix+"*"))
        for im in imgs:
            filea = fits.open(im)
            imgs_headers.append(filea[0].header)
            imgs_list.append(filea[0].data)
            filea.close()
    return imgs_list,imgs_headers



events_selected = [ 
                    # "03/09/2009",
                    # "03/04/2010",
                    # "08/04/2010",
                    # "23/05/2010",
                    # "16/06/2010",
                    # "01/08/2010",
                    # "15/12/2010",
                    # "30/01/2011",
                    # "14/02/2011",
                    # "23/06/2011",
                    # "02/08/2011",
                    # "06/09/2011",
                    # "22/10/2011",
                    # "12/07/2012",
                    # "15/01/2013",
                    # "30/09/2013",
                    # "15/04/2020",
                    # "23/06/2020",
                    # "09/07/2020",
                    # "20/07/2020",
                    # "30/09/2020",
                    # "26/10/2020",
                    # "07/12/2020",
                    # "11/02/2021",
                    # "20/02/2021",
                    # "10/04/2021",
                    # "22/04/2021",
                    # "09/05/2021",
                    # "29/05/2021",
                    # "23/08/2021",
                    # "13/09/2021",
                    # "09/10/2021",
                    # "04/04/2022",
                    # "11/04/2022",
                    # "27/06/2022",
                    # "03/07/2022",
                    # "03/11/2022",
                    # "31/12/2022",
                    # "16/04/2023",
                    "21/04/2023"
                ]

new_events_list = []
for i in range(0,len(events_selected)):
    date = datetime.strptime(events_selected[i],'%d/%m/%Y')
    dates = [
             datetime.strptime(events_selected[i],'%d/%m/%Y')-timedelta(days=3),
             datetime.strptime(events_selected[i],'%d/%m/%Y')-timedelta(days=2),
             datetime.strptime(events_selected[i],'%d/%m/%Y')-timedelta(days=1),
             datetime.strptime(events_selected[i],'%d/%m/%Y'),
             datetime.strptime(events_selected[i],'%d/%m/%Y')+timedelta(days=1),
             datetime.strptime(events_selected[i],'%d/%m/%Y')+timedelta(days=2),
             datetime.strptime(events_selected[i],'%d/%m/%Y')+timedelta(days=3)
             ]
    new_events_list.append(dates)





for i in range(0,len(new_events_list)):

    # beacon_datas, beacon_headers = create_l2_beacon(new_events_list[i])
    # differences,headers_differences  = compute_running_difference(beacon_datas,beacon_headers,1)
    # cuts_beacon,dates,elongations = create_jplot_from_differences(differences,headers_differences)

    # elongations = np.asarray(elongations)#*180/np.pi
    # elongations = [np.nanmin(elongations), np.nanmax(elongations)]

    # if(datetime.strptime(beacon_headers[0]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f').year<2015):
    #     origin ='upper'
    # else:
    #     origin = 'lower'

    # fig,ax = plt.subplots(3,2,figsize=(20,10))
    # p2, p96 = np.nanpercentile(cuts_beacon, (5, 98))
    # cuts_beacon2 = exposure.rescale_intensity(cuts_beacon, in_range=(p2, p96))
    # vmin_h1 = np.nanmedian(cuts_beacon2) - 0.5 * np.nanstd(cuts_beacon2)
    # vmax_h1 = np.nanmedian(cuts_beacon2) + 2 * np.nanstd(cuts_beacon2)
    # ax[0][0].imshow(cuts_beacon2.reshape((cuts_beacon2.shape[0],cuts_beacon2.shape[1]*cuts_beacon2.shape[2])),aspect="auto",origin=origin,interpolation='none',extent = [dates[0],dates[-1],elongations[0],elongations[-1]],cmap='afmhot',vmin=vmin_h1,vmax=vmax_h1)


    # cuts_beacon = np.median(cuts_beacon,2)
    # p2, p96 = np.nanpercentile(cuts_beacon, (5, 98))
    # cuts_beacon = exposure.rescale_intensity(cuts_beacon, in_range=(p2, p96))
    # vmin_h1 = np.nanmedian(cuts_beacon) - 0.5 * np.nanstd(cuts_beacon)
    # vmax_h1 = np.nanmedian(cuts_beacon) + 2 * np.nanstd(cuts_beacon)
    # ax[0][1].imshow(cuts_beacon,aspect="auto",origin=origin,interpolation='none',extent = [dates[0],dates[-1],elongations[0],elongations[-1]],cmap='afmhot',vmin=vmin_h1,vmax=vmax_h1)




    # enhanced_data, enhanced_headers,has_header = load_enhanced_beacon(new_events_list[i])
    # differences,headers_differences  = compute_running_difference(enhanced_data,enhanced_headers,2,False)
    # cuts_enhanced,dates,elongations2 = create_jplot_from_differences(differences,headers_differences)
    # elongations2 = np.asarray(elongations2)#*180/np.pi
    # elongations2 = [np.nanmin(elongations2), np.nanmax(elongations2)]


    # # p2, p96 = np.nanpercentile(cuts_enhanced, (5, 95))
    # # cuts_enhanced2 = exposure.rescale_intensity(cuts_enhanced, in_range=(p2, p96))
    # vmin_h1 = np.nanmedian(cuts_enhanced) - 0.5 * np.nanstd(cuts_enhanced)
    # vmax_h1 = np.nanmedian(cuts_enhanced) + 2 * np.nanstd(cuts_enhanced)
    # ax[1][0].imshow(cuts_enhanced.reshape((cuts_enhanced.shape[0],cuts_enhanced.shape[1]*cuts_enhanced.shape[2])),aspect="auto",origin=origin,interpolation='none',extent = [dates[0],dates[-1],elongations2[0],elongations2[-1]],cmap='afmhot',vmin=vmin_h1,vmax=vmax_h1)


    # cuts_enhanced = np.median(cuts_enhanced,2)
    # p2, p96 = np.nanpercentile(cuts_enhanced, (5, 95))
    # cuts_enhanced = exposure.rescale_intensity(cuts_enhanced, in_range=(p2, p96))
    # vmin_h1 = np.nanmedian(cuts_enhanced) - 0.5 * np.nanstd(cuts_enhanced)
    # vmax_h1 = np.nanmedian(cuts_enhanced) + 2 * np.nanstd(cuts_enhanced)
    # ax[1][1].imshow(cuts_enhanced,aspect="auto",origin=origin,interpolation='none',extent = [dates[0],dates[-1],elongations2[0],elongations2[-1]],cmap='afmhot',vmin=vmin_h1,vmax=vmax_h1)


    fig,ax = plt.subplots(1,2)

   

    enhanced_data2, enhanced_headers2 = load_final_enhanced(new_events_list[i],"../enhanced_fits/")
    differences2,headers_differences2  = compute_running_difference(enhanced_data2,enhanced_headers2,2,False)
    cuts_enhanced2,dates2,elongations3 = create_jplot_from_differences(differences2,headers_differences2,cadence=40)
    elongations3 = np.asarray(elongations3)#*180/np.pi
    elongations3 = [np.nanmin(elongations3), np.nanmax(elongations3)]


     # p2, p96 = np.nanpercentile(cuts_enhanced, (5, 95))
    # cuts_enhanced2 = exposure.rescale_intensity(cuts_enhanced, in_range=(p2, p96))
    vmin_h1 = np.nanmedian(cuts_enhanced2) - 0.5 * np.nanstd(cuts_enhanced2)
    vmax_h1 = np.nanmedian(cuts_enhanced2) + 2 * np.nanstd(cuts_enhanced2)
    ax[0].imshow(cuts_enhanced2.reshape((cuts_enhanced2.shape[0],cuts_enhanced2.shape[1]*cuts_enhanced2.shape[2])),aspect="auto",origin='lower',interpolation='none',extent = [dates2[0],dates2[-1],elongations3[0],elongations3[-1]],cmap='afmhot',vmin=vmin_h1,vmax=vmax_h1)
    



    cuts_enhanced2 = np.median(cuts_enhanced2,2)
    p2, p96 = np.nanpercentile(cuts_enhanced2, (5, 95))
    cuts_enhanced2 = exposure.rescale_intensity(cuts_enhanced2, in_range=(p2, p96))
    vmin_h1 = np.nanmedian(cuts_enhanced2) - 0.5 * np.nanstd(cuts_enhanced2)
    vmax_h1 = np.nanmedian(cuts_enhanced2) + 2 * np.nanstd(cuts_enhanced2)
    ax[1].imshow(cuts_enhanced2,aspect="auto",origin='lower',interpolation='none',extent = [dates2[0],dates2[-1],elongations3[0],elongations3[-1]],cmap='afmhot',vmin=vmin_h1,vmax=vmax_h1)





    for a in ax.reshape(-1): 
        a.yaxis.set_minor_locator(MultipleLocator(2))
        a.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 24)))
        a.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
        a.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6)))




    plt.savefig("../test.png")
    # plt.show()
    # plt.savefig("jplots_res_compscience_V2/"+new_events_list[i][3].strftime("%Y-%m-%d").replace(":","-")+"_2.png")



# center = [126.5,126.5]
# center_science = [511.5,511.5]
# resolution_enhanced = 512
# resolution_science = 1024
# resolution_beacon = 256
# clahe_b = cv2.createCLAHE(clipLimit=10,tileGridSize=(10,10))

# for ev in new_events_list:
#     imgs_list = []
#     beacon_list = []
#     science_list = []
#     imgs_headers = []
#     science_headers = []

#     for e in ev:
#         prefix=str(e.strftime('%Y'))+"-"+str(e.strftime('%m'))+"-"+str(e.strftime('%d'))
#         imgs = natsorted(glob.glob('/Volumes/Data_drive/res_model_final2/'+prefix+"*"))
#         for im in imgs:
#             name = '/Volumes/Data_drive/Reduced/Test/'+prefix+"/beacon/"+im.split("/")[-1][:-3]+"fts"
#             filea = fits.open(name)
#             imgs_headers.append(filea[0].header)
#             filea.close()
#             imgs_list.append(np.asarray(Image.open(im).convert("L"))/255.0)
#             beacon_list.append(np.asarray(Image.open('/Volumes/Data_drive/Test_images/'+im.split("/")[-1]).convert("L"))/255.0)
        
        
#         imgs_science = natsorted(glob.glob('/Volumes/Data_drive/Test_images_science/'+prefix+"*"))
#         for im2 in imgs_science:
#             science_list.append(np.asarray(Image.open(im2).convert("L"))/255.0)
#             name = '/Volumes/Data_drive/Reduced/Test/'+prefix+"/science/"+im2.split("/")[-1][:-3]+"fts"
#             if(os.path.exists(name) ):
#                 filea = fits.open(name)
#             else:
#                 filea = fits.open('/Volumes/Data_drive/Reduced/Train/'+prefix+"/science/"+im2.split("/")[-1][:-3]+"fts")
#             science_headers.append(filea[0].header)
#             filea.close()
     


#     date1 = datetime.strptime(imgs_headers[1]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')
#     date2  = datetime.strptime(imgs_headers[-1]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')

#     dates2 = []
#     current_date = date1
#     while current_date <= date2:
#         dates2.append(current_date)
#         current_date = current_date + timedelta(minutes=120)
#     dates2.append(date2)

#     # cuts_s2.fill(np.nan)
#     cuts_enhancedbeacon = np.zeros((resolution_enhanced,len(dates2),64))

#     cuts_beacon = np.zeros((resolution_beacon,len(dates2),64))


#     # date1 = datetime.strptime(science_headers[1]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')
#     # date2  = datetime.strptime(science_headers[-1]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')

#     # dates3 = []
#     # current_date = date1
#     # while current_date <= date2:
#     #     dates3.append(current_date)
#     #     current_date = current_date + timedelta(minutes=120)
#     # dates3.append(date2)
#     cuts_science = np.zeros((resolution_science,len(dates2),64))




#     multipliers  = {

#         resolution_beacon: 1,
#         resolution_science: 4, 
#         resolution_enhanced: 2,


#     }


#     for i in range(1,len(imgs_list)):

#         time1 = datetime.strptime(imgs_headers[i-1]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')
#         time2 = datetime.strptime(imgs_headers[i]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')
#         if( ((time2-time1).total_seconds()/60.0)<124.0):
#             im1 = imgs_list[i-1]
#             im2 = imgs_list[i]

#             im1_beacon = beacon_list[i-1]
#             im2_beacon = beacon_list[i]


#             head1 = imgs_headers[i-1]
#             head2 = imgs_headers[i]

#             print(head1["DATE-END"],head2["DATE-END"])


        
#             wcs = WCS(head2,key='A')
#             center_prev = wcs.all_world2pix(head1["crval1a"],head1["crval2a"], 0)
#             shift_arr = np.array([center_prev[1]-center[1],center_prev[0]-center[0]])
#             shift_arr2 = shift_arr.copy()

#             difference   = np.float32(im2-shift(im1,shift_arr*multipliers[resolution_enhanced], mode='nearest',prefilter=False)) 
#             difference_beacon   = np.float32(im2_beacon-shift(im1_beacon,shift_arr*multipliers[resolution_beacon], mode='nearest',prefilter=False)) 


#             # fig,ax = plt.subplots(1,2)
#             # ax[0].imshow(difference,cmap='gray')
#             # ax[1].imshow(difference_beacon,cmap='gray')
#             # plt.show()


#             difference_beacon = (difference_beacon - difference_beacon.min())/ (difference_beacon.max()-difference_beacon.min())

#             cuts_enhancedbeacon = insert_cut(dates2,datetime.strptime(imgs_headers[i]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f'),difference,cuts_enhancedbeacon,head2,size=resolution_enhanced)
#             cuts_beacon = insert_cut(dates2,datetime.strptime(imgs_headers[i]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f'),clahe_b.apply((difference_beacon*255).astype(np.uint8))/255.0,cuts_beacon,head2,size=resolution_beacon)


#     # for i in range(0,len(science_list)):
#     #     time1 = datetime.strptime(science_headers[i-1]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')
#     #     time2 = datetime.strptime(science_headers[i]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')
#     #     if( ((time2-time1).total_seconds()/60.0)<300.0):
#     #         science_header1 = science_headers[i-1]
#     #         science_header2 = science_headers[i]

#     #         im1_science = science_list[i-1]
#     #         im2_science = science_list[i]

#     #         wcs = WCS(science_header2,key='A')
#     #         center_prev = wcs.all_world2pix(science_header1["crval1a"],science_header1["crval2a"], 0)
#     #         shift_arr_science = np.array([center_prev[1]-center_science[1],center_prev[0]-center_science[0]])
#     #         shift_arr_science = shift_arr_science.copy()

#     #         difference_science   = np.float32(im2_science-shift(im1_science,shift_arr_science, mode='nearest',prefilter=False)) 

#     #         cuts_science = insert_cut(dates2,datetime.strptime(science_headers[i]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f'),difference_science,cuts_science,science_header2,size=resolution_science)


#     cuts_science[resize(cuts_enhancedbeacon,(1024,cuts_science.shape[1]))==0.0] = 0.0
#     # Contrast stretching
#     p2, p96 = np.nanpercentile(cuts_science, (5, 98))
#     cuts_science = exposure.rescale_intensity(cuts_science, in_range=(p2, p96))

#     # cuts_s2 = (cuts_s2 - cuts_s2.min())/(cuts_s2.max() - cuts_s2.min())

#     p2, p96 = np.nanpercentile(cuts_beacon, (5, 96))
#     cuts_beacon = exposure.rescale_intensity(cuts_beacon, in_range=(p2, p96))

#     # p2, p96 = np.nanpercentile(cuts_enhancedbeacon, (5, 98))
#     # cuts_enhancedbeacon = exposure.rescale_intensity(cuts_enhancedbeacon, in_range=(p2, p96))

#     # cuts_beacon = (cuts_beacon - cuts_beacon.min())/(cuts_beacon.max() - cuts_beacon.min())

#     if(datetime.strptime(imgs_headers[0]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f').year<2015):
#         origin='upper'
#     else:
#         origin = 'lower'

#     dates2 = mdates.date2num(dates2)

#     fig,ax = plt.subplots(3,2,figsize=(20,10))
#     ax[0][0].imshow(np.median(cuts_beacon,2),aspect="auto",origin=origin,interpolation='none',extent = [dates2[0],dates2[-1],4,24],cmap='gray')
#     ax[0][1].imshow(cuts_beacon.reshape((resolution_beacon,cuts_beacon.shape[1]*cuts_beacon.shape[2])),aspect="auto",origin=origin,interpolation='none',extent = [dates2[0],dates2[-1],4,24],cmap='gray')

#     ax[1][0].imshow(np.median(cuts_enhancedbeacon,2),aspect="auto",interpolation='none',origin=origin,extent = [dates2[0],dates2[-1],4,24],cmap='gray')
#     ax[1][1].imshow(cuts_enhancedbeacon.reshape((resolution_enhanced,cuts_enhancedbeacon.shape[1]*cuts_enhancedbeacon.shape[2])),aspect="auto",origin=origin,interpolation='none',extent = [dates2[0],dates2[-1],4,24],cmap='gray')

#     ax[2][0].imshow(np.median(cuts_science,2),aspect="auto",interpolation='none',origin=origin,extent = [dates2[0],dates2[-1],4,24],cmap='gray')
#     ax[2][1].imshow(cuts_science.reshape((resolution_science,cuts_science.shape[1]*cuts_science.shape[2])),aspect="auto",origin=origin,interpolation='none',extent = [dates2[0],dates2[-1],4,24],cmap='gray')
    
#     ax[0][0].xaxis_date()
#     ax[0][1].xaxis_date()
#     ax[1][0].xaxis_date()
#     ax[1][1].xaxis_date()

#     ax[2][0].xaxis_date()
#     ax[2][1].xaxis_date()

#     ax[0][0].title.set_text("Beacon medianed")
#     ax[0][1].title.set_text("Beacon slice")
#     ax[1][0].title.set_text("Enhanced Beacon medianed")
#     ax[1][1].title.set_text("Enhanced Beacon slice")
#     ax[2][0].title.set_text("Science medianed")
#     ax[2][1].title.set_text("Science slice")

    

#     plt.show()
#     # print(len(imgs_list))
#     # plt.savefig("jplots_res_compscience/"+ev[3].strftime("%Y-%m-%d").replace(":","-")+"_2.png")
#     # exit()

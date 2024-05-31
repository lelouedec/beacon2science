from astropy.io import fits
from natsort import natsorted
import glob
import numpy as np 
from datetime import datetime
import os
import json
from astropy.wcs import WCS
import astropy
from astropy.constants import iau2015 as const
import matplotlib.patches as patches
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.pyplot as plt 
import matplotlib 
import seaborn as sns
from matplotlib.gridspec import GridSpec
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
try:
    import astrospice
except:
    print("no astrospice")
import sunpy.coordinates
matplotlib.rcParams['backend'] = 'Qt5Agg' 
plt.rcParams.update({'font.size': 20})


def convert_size():
    paths = natsorted(glob.glob("/Volumes/Data_drive/Reduced/Train/*/*/*"))
    start = datetime(2021,7,2)


    for p in paths:
        print(p)
        d = p.split("/")[5].replace("-",'') 
        date = datetime(int(d[:4]),int(d[4:6]),int(d[6:]))
        if(date>=start):
            filea = fits.open(p)
            data = filea[0].data
            header = filea[0].header
            fits.writeto(p, data.astype(np.float32), header, output_verify='silentfix', overwrite=True)

def convert_end():
    paths = natsorted(glob.glob("/media/lelouedec/Data_drive/Reduced/Test/*/*/*"))

    for p in paths:
        if(p[-3:]!="fts"):
            if os.path.exists(p+".fts"):
                os.remove(p) 
                print("deleting",p)
            else:
                os.rename(p,p+".fts")
                print("renaming",p)
            # exit()


def unfolder_images():
    files = natsorted(glob.glob('/media/lelouedec/Data_drive/Test_events_beacon/*/*'))
    for f in files:
        p = f.split("/")[-1]
        p2 = list(p)
        p2[-11] = "-"
        p2[-14] = "-"
        p2 = "".join(p2)
        print(p2)
        # exit()
        os.system("cp "+f +" /media/lelouedec/Data_drive/Test_images_beacon/"+p2)


def create_dataset_images_pairs():
    files = natsorted(glob.glob('/media/lelouedec/Data_drive/Test_images_beacon/*'))
    current_date = datetime.strptime(files[0].split("/")[-1].split("T")[0],'%Y-%m-%d')
    days = {}
    event = []
    for i in range(1,len(files)):
        date = datetime.strptime(files[i].split("/")[-1].split("T")[0],'%Y-%m-%d')
        if((date-current_date).total_seconds()/(3600*24))>1:
            first_date = datetime.strptime(event[0].split("/")[-1].split("T")[0],'%Y-%m-%d')
            days[first_date] = event
            event = []
            event.append(files[i])
        else:
            event.append(files[i])
        current_date = date
    print(len(days))

    pairs = []
    for k in days.keys():
        event_data = days[k]
        print(k)
        for j in range(0,len(event_data)-1,2):
            e  = event_data[j]
            e2 = event_data[j+1]
            pair = {}
            pair["b1"] = e.split("/")[-1]
            pair["b2"] = e2.split("/")[-1]
            paths = natsorted(glob.glob('/media/lelouedec/Data_drive/Test_images_science/'+e.split("/")[-1].split("T")[0]+"*"))
            if(e.split("/")[-1].split("T")[0]!=e2.split("/")[-1].split("T")[0]):
                paths+=natsorted(glob.glob('/media/lelouedec/Data_drive/Test_images_science/'+e2.split("/")[-1].split("T")[0]+"*"))
            in_betweens = []
            found = False
            for p in paths:
                if(p.split("/")[-1]==e.split("/")[-1]):
                    # print("  ",e,p)
                    found = True
                    pair["s1"] = p.split("/")[-1]
                elif(p.split("/")[-1]==e2.split("/")[-1]):
                    # print("  ",e2,p)
                    pair["s2"] = p.split("/")[-1]
                    found = False
                elif(found):
                    in_betweens.append(p.split("/")[-1])
            pair["mid"] = in_betweens

            # print("     ",pair)
            if(len(pair.keys())==5 and len(in_betweens)>=2):
                pairs.append(pair)
            # print(len(pairs),len(pair.keys()),len(in_betweens))
        # exit()
    print(len(pairs))
    with open("dataset_test.json", "w") as final:
        json.dump(pairs, final)


def compute_shift_dataset():
    with open("dataset_test.json", "r") as final:
        data_json =  json.load(final)

    for i,d in enumerate(data_json):
        print(d)
        header_s1 = list("/Volumes/Data_drive/Reduced/Test/"+d["s1"].split("T")[0]+"/science/"+d["s1"][:-3]+"fts")
        header_s1[-4] = "."
        header_s1[-11] = "-"
        header_s1[-14] = "-"
        header_s1[-8] = "."
        header_s1 = "".join(header_s1)
        header_s2 = list("/Volumes/Data_drive/Reduced/Test/"+d["s2"].split("T")[0]+"/science/"+d["s2"][:-3]+"fts")
        header_s2[-4] = "."
        header_s2[-8] = "."
        header_s2[-11] = "-"
        header_s2[-14] = "-"
        header_s2 = "".join(header_s2)
        
        # header_b1 = "/Volumes/Data_drive/Reduced/Train/"+d["b1"].split("T")[0]+"/beacon/"+d["b1"][:-3]+"fts"
        # header_b2 = "/Volumes/Data_drive/Reduced/Train/"+d["b2"].split("T")[0]+"/beacon/"+d["b2"][:-3]+"fts"


        header_s1 = fits.open(header_s1)[0].header
        header_s2 = fits.open(header_s2)[0].header

        center      = header_s2['crpix1']-1, header_s2['crpix2']-1
        wcs = WCS(header_s2,key='A')
        center_prev = wcs.all_world2pix(header_s1["crval1a"],header_s1["crval2a"], 0)
        shift_arr = [center_prev[1]-center[1],(center_prev[0]-center[0])]
        print(shift_arr)
        data_json[i]["shift"] = shift_arr


    with open("dataset_test.json", "w") as final:
        json.dump(data_json, final)
        
def rename_this_shit():
    paths = natsorted(glob.glob("/media/jlelouedec/Data_drive/images_beacon/*"))
    for p in paths:
        print(p[-11],p[-14])
        p2 = list(p)
        p2[-11] = "-"
        p2[-14] = "-"
        p2 = "".join(p2)
        print(p,p2)
        os.system('mv '+p +' '+ p2)
        # exit() 


def get_poses_dataset():
    with open("dataset_bis.json", "r") as final:
        data_json =  json.load(final)

    poses = []
    dates = []

    for i,d in enumerate(data_json):
        header_s1 = list("/Volumes/Data_drive/Reduced/Train/"+d["s1"].split("T")[0]+"/science/"+d["s1"][:-3]+"fts")
        header_s1[-4] = "."
        header_s1[-11] = "-"
        header_s1[-14] = "-"
        header_s1[-8] = "."
        header_s1 = "".join(header_s1)
        header_s2 = list("/Volumes/Data_drive/Reduced/Train/"+d["s2"].split("T")[0]+"/science/"+d["s2"][:-3]+"fts")
        header_s2[-4] = "."
        header_s2[-8] = "."
        header_s2[-11] = "-"
        header_s2[-14] = "-"
        header_s2 = "".join(header_s2)
        
      

        header_s1 = fits.open(header_s1)[0].header
        header_s2 = fits.open(header_s2)[0].header

        poses.append([header_s1["HEQX_OBS"],header_s1["HEQY_OBS"],header_s1["HEQZ_OBS"]])
        poses.append([header_s2["HEQX_OBS"],header_s2["HEQY_OBS"],header_s2["HEQZ_OBS"]])
        dates.append(header_s1["DATE-END"])
        dates.append(header_s2["DATE-END"])

    print(len(dates),len(poses),len(data_json))
    poses = np.array(poses)
    np.save("positions.npy", poses)

    dates = np.array(dates)
    np.save("dates.npy", dates)

def cartesian2polar(x,y):
    
    theta = np.arctan2(y,x)
    r  = np.sqrt(x**2 + y**2)
    return theta, r

AU = const.au.value
R_sun = 696340000
minr = (R_sun/AU)*12

def plot_for_range(ax,events_selected,dates,colors):
   
    R = 0.5
    R2 = 0.3
    min_theta = 10.0
    max_theta = 0.0
    for j,p in enumerate(events_selected):
        # if( (int(dates[j][:4])>=begin and int(dates[j][:4])<=end)  and cnt <max_shots):
        path = ("/Volumes/Data_drive/Reduced/Train/"+p.strftime("%Y-%m-%d"))
        if(os.path.exists):
            first = natsorted(glob.glob(path+"/science/*"))[0]
            filea = fits.open(first)
            header = filea[0].header
            filea.close()

            ps = [header["HEQX_OBS"],header["HEQY_OBS"],header["HEQZ_OBS"]]
            p1_x = ps[0]/AU
            p1_y = ps[1]/AU

            thetas,rs = cartesian2polar(p1_x,p1_y)
            # ax.scatter(thetas,rs)


            if np.abs(thetas)<np.abs(min_theta):
                min_theta = thetas

            if np.abs(thetas)>np.abs(max_theta):
                max_theta = thetas

            dx,dy = p1_x,p1_y
            dxr,dyr = -dy,dx
            d = np.sqrt(dx**2+dy**2)

            rho = minr/d
            ad = rho**2
            bd = rho*np.sqrt(1-rho**2)
            if(int(p.year)>=2015):
                T1x = ad*dx + bd*dxr
                T1y = ad*dy + bd*dyr
                ### Rotate 20 degrees 
                p2_x = (T1x-p1_x)* np.cos(-20*np.pi/180)  - (T1y-p1_y)  * np.sin(-20*np.pi/180) + p1_x
                p2_y = (T1x-p1_x)* np.sin(-20*np.pi/180)  + (T1y-p1_y)  * np.cos(-20*np.pi/180) + p1_y
            else:
                T1x = ad*dx - bd*dxr
                T1y = ad*dy - bd*dyr
                ### Rotate 20 degrees 
                p2_x = (T1x-p1_x)* np.cos(20*np.pi/180) - (T1y-p1_y)  * np.sin(20*np.pi/180) + p1_x
                p2_y = (T1x-p1_x)* np.sin(20*np.pi/180)  + (T1y-p1_y) * np.cos(20*np.pi/180) + p1_y

            

            ### vector direction for p1
            vec1 = np.array([minr-p1_x,-p1_y])
            vec1 = vec1/np.linalg.norm(vec1)

            ### vector direction for p2
            vec2 = np.array([p2_x-p1_x,p2_y-p1_y])
            vec2 = vec2/np.linalg.norm(vec2)

            p1 = p1_x+R*vec1[0],p1_y+R*vec1[1]
            p2 = p1_x+R*vec2[0],p1_y+R*vec2[1]
            
           

            t1,r1 = cartesian2polar(p1[0],p1[1])
            t2,r2 = cartesian2polar(p2[0],p2[1])

            tetas = []
            rss = []
            nb = 500
            for perc in range(0,nb):
                if(int(p.year)>=2015):
                    p1a_x = (p1[0]-p1_x)* np.cos((-20*perc/nb)*np.pi/180)  - (p1[1]-p1_y) * np.sin((-20*perc/nb)*np.pi/180) + p1_x
                    p1a_y = (p1[0]-p1_x)* np.sin((-20*perc/nb)*np.pi/180)  + (p1[1]-p1_y) * np.cos((-20*perc/nb)*np.pi/180) + p1_y
                else:
                    p1a_x = (p1[0]-p1_x)* np.cos((20*perc/nb)*np.pi/180)  - (p1[1]-p1_y) * np.sin((20*perc/nb)*np.pi/180) + p1_x
                    p1a_y = (p1[0]-p1_x)* np.sin((20*perc/nb)*np.pi/180)  + (p1[1]-p1_y) * np.cos((20*perc/nb)*np.pi/180) + p1_y

                tss1,rss1 = cartesian2polar(p1a_x,p1a_y)
                tetas.append(tss1)
                rss.append(rss1)


            colors_new = np.ones((len([thetas,t1]+list(tetas)+[t2]),3)) * colors[int(p.year)-2007][:3]
          
            pol = ax.fill([thetas,t1]+list(tetas)+[t2],[rs,r1]+list(rss)+[r2],fill=False,c=colors_new[int(p.year)-2007],linestyle="dashed")


            
def add_fts_to_name():
    paths  = natsorted(glob.glob("/Volumes/Data_drive/Reduced/Test/*/*/*"))
    for p in paths:
        if(p[-3:]!="fts"):
            os.system('mv '+p +' '+ p+".fts")



def plot_dataset():
    events_selected = [ "03/09/2009",
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
                    "21/04/2023"]
    

    for i in range(0,len(events_selected)):
        events_selected[i] = datetime.strptime(events_selected[i],'%d/%m/%Y')



    poses = np.load("positions.npy")
    dates = np.load("dates.npy")


    fig = plt.figure(figsize=(100, 50), dpi=100)
    gs = GridSpec(nrows=1, ncols=2)
    years = np.zeros((17))

    for d in dates:
        years[int(d.split("T")[0][:4])-2007]+=1



    

    # pal = sns.color_palette("Greens_d", years.shape[0])
    pal = sns.color_palette("dark:#5A9_r", years.shape[0])

    rank = years.argsort().argsort()
    fake = pd.DataFrame({'year': list(range(2007,2024,1)), 'val': years})


    ax1 = fig.add_subplot(gs[0, 0])
    sns.barplot(ax=ax1,x = 'year', y = 'val', data = fake, palette=np.array(pal)[rank])
    ax1.set_xlabel("Years")
    ax1.set_ylabel("# Images",labelpad=30)
    ax1.set_title('Number of images per year within our dataset', pad=45)

    plt.setp(ax1.get_xticklabels()[::2], visible=False)
    # plt.setp(ax1.get_xticklabels(), visible=True)


    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    

    norm = matplotlib.colors.Normalize(vmin=years.min(), vmax=years.max()) 
    sm = plt.cm.ScalarMappable(cmap=sns.color_palette("dark:#5A9_r", years.shape[0],as_cmap=True), norm=norm) 
    plt.colorbar(sm,cax,shrink=0.5) 

    



    t = [mdates.date2num(datetime.strptime(i[:-13], "%Y-%m-%d")) for i in dates]
    loc = mdates.AutoDateLocator()





    AU = const.au.value
    R_sun = 696340000
    minr = (R_sun/AU)*12

    thetas = np.arctan2(poses[:,1],poses[:,0])
    radii  = np.sqrt(poses[:,1]**2 + poses[:,0]**2)/AU


    ax2 = fig.add_subplot(gs[:, 1],projection='polar')


   

    colors = []
    for i in range(0,18):
        colors.append(plt.cm.tab20(i))
    
    
    yeeears = np.arange(2007,2025,1,int)
    yeeears = [mdates.date2num(datetime.strptime(str(y), "%Y")) for y in yeeears ]
 

    norm = matplotlib.colors.BoundaryNorm(yeeears,len(colors))

    deltac = (yeeears[-1]-yeeears[0])/(2*(len(colors)-1))

    smap=matplotlib.colors.ListedColormap(colors)

    sc = plt.scatter(thetas,radii, marker='o', c=t,cmap=smap,norm=norm)
    
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=smap), ticks=yeeears-deltac,format=mdates.AutoDateFormatter(loc),shrink=0.8,ax=ax2,drawedges=False)

    # cbar.ax.tick_params(size=0)
    cbar.minorticks_off()

    coords_earth = astrospice.generate_coords('Earth', '2023-05-01T00:00:00.000')
    coords_earth_hee = coords_earth.transform_to(sunpy.coordinates.frames.HeliocentricEarthEcliptic()).data

    earth_x,eath_y = coords_earth_hee.x.value[0]*1000,coords_earth_hee.y.value[0]*1000
    theta_earth = np.arctan2(eath_y,earth_x)
    radii_earth = np.sqrt(earth_x**2 + eath_y**2)/AU

    backcolor='black'
    fsize=20
    lines,labels = plt.thetagrids(range(0,360,45),(u'0\u00b0',u'45\u00b0',u'90\u00b0',u'135\u00b0',u'+/- 180\u00b0',u'- 135\u00b0',u'- 90\u00b0',u'- 45\u00b0'), ha='center', fmt='%d',fontsize=fsize,color=backcolor, alpha=0.9,zorder=4)
    for l in labels:
        print(l.get_text())
        if(l.get_text()!='90°' and l.get_text()!='-90°' and l.get_text()!='0°'):
            l.set_y(l.get_position()[1]-0.1)
    plt.rgrids((0.1,0.3,0.5,0.7,1.0,2.5),('0.10','0.3','0.5','0.7','1.0 AU','2.5 AU'),angle=125, fontsize=fsize-3,alpha=0.5, color=backcolor)
    middles=np.arange(360/12/2 ,360, 360/12)*np.pi/180
    ax2.bar(0.0, 2.5, width= 360*np.pi/180, bottom=0.05, color='w', edgecolor='w',zorder=0)
    ax2.set_ylim(0, 1.2)
    ax2.set_facecolor('#c9c9c9')
    ax2.tick_params(axis='both',color='#c9c9c9')
    plt.grid(axis='y',color='#c9c9c9', linestyle=':', linewidth=1)    
    #Sun
    ax2.scatter(0,0,s=100,c='yellow',alpha=1, edgecolors='black', linewidth=0.3)
    #earth
    ax2.scatter(0.0,1.0,marker='o',s=50.0,c='blue')
    ax2.set_theta_zero_location('E')
    ax2.set_title("Training set STEREO A positions and Test CME events points of views", pad=20)


    plot_for_range(ax2,events_selected,dates,colors)


    plt.show()



unfolder_images()


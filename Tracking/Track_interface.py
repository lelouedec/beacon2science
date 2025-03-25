
# self.images = natsorted(glob.glob("./L2_pngs/beacon/2024-.7-28/*"))

import sys
from PyQt5.QtWidgets import QApplication,QCheckBox, QMainWindow,QSlider, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import Qt, pyqtSignal
import PyQt5
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['backend'] = 'Qt5Agg' 
from skimage import exposure
import pickle
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import glob
from natsort import natsorted
from datetime import datetime,timedelta,timezone
import skimage
from skimage.draw import line
from skimage.transform import  resize,rotate
import cv2
import os 
import matplotlib.dates as mdates

from astropy.wcs import WCS
from scipy.ndimage import shift
import matplotlib.pyplot as plt 
from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from sunpy.coordinates import Helioprojective
from sunpy.coordinates.ephemeris import get_horizons_coord
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from PIL import Image
from skimage import color, morphology
import csv
import pandas as pd
import time 
matplotlib.rcParams.update({'font.size': 20})


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
                        "30/12/2009",
                        "01/03/2010",
                        "03/04/2010",
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



def draw_ecliptic(dtime,diff,header,size,earth,stereo):
    wcoord = WCS(header)
    # earth = get_body_heliographic_stonyhurst('earth', dtime)
    # stereo = get_horizons_coord('STEREO-A', dtime)
    e_hpc = SkyCoord(earth).transform_to(Helioprojective(observer=stereo))
    
    e_x = e_hpc.Tx.to(u.deg).value *np.pi/180
    e_y = e_hpc.Ty.to(u.deg).value *np.pi/180


    e_pa = np.arctan2(-np.cos(e_y)*np.sin(e_x), np.sin(e_y))
    delta_pa = e_pa#(-90.0*np.pi/180)#e_pa
    e_val = [(delta_pa)-2*np.pi/180, (delta_pa)+2*np.pi/180]


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

           
    #### displaying lines for the ecliptic in green
    rr, cc = line(maxid,0,maxid2,size-1)
    diff_begin[rr,cc] = [255.0,0.0,0.0]
    diff_begin[rr+1,cc] = [255.0,0.0,0.0]

    rr, cc = line(minid,0,minid2,size-1)
    diff_begin[rr,cc] = [255.0,0.0,0.0]
    diff_begin[rr-1,cc] = [255.0,0.0,0.0]

    return diff_begin, elon_reg

# new_events_list[i][3]
def load_enhanced_beacon(event):
    # with open("./enhanced_beacon/"+event[3].strftime("%Y-%m-%d")+".p", 'rb') as f:
    #     jplot_dict = pickle.load(f)
    #     imgs = jplot_dict["data"]
    #     headers = jplot_dict["dates"]
    #     return imgs,headers
    imgs = []
    headers = []
    for e in event:
        files_paths = natsorted(glob.glob("./enhanced_beacon/"+e.strftime("%Y-%m-%d")+"*.p"))
        for p in files_paths:
            with open(p, 'rb') as f:
                jplot_dict = pickle.load(f)
                imgs.append(jplot_dict["data"])
                headers.append(jplot_dict["header"])
    return imgs,headers

def load_enhanced_beacon_normal(event):
    imgs = []
    headers = []
    for e in event:
        files_paths = natsorted(glob.glob("./enhancedbeacon0_nonmedian/"+e.strftime("%Y-%m-%d")+"*.p"))
        for p in files_paths:
            with open(p, 'rb') as f:
                jplot_dict = pickle.load(f)
                imgs.append(jplot_dict["data"])
                headers.append(jplot_dict["header"])
    return imgs,headers

def load_beacon(event):
     with open("./beacon/"+event[3].strftime("%Y-%m-%d")+".p", 'rb') as f:
        jplot_dict = pickle.load(f)
        imgs = jplot_dict["data"]
        headers = jplot_dict["dates"]
        return imgs,headers
     
def load_science(event):
     with open("./science/"+event[3].strftime("%Y-%m-%d")+".p", 'rb') as f:
        jplot_dict = pickle.load(f)
        imgs = jplot_dict["data"]
        headers = jplot_dict["dates"]
        return imgs,headers
     


def load_l2_beacon(dates):
    datas   = []
    headers = []
    for d in dates:
        prefix = str(d.strftime('%Y'))+"-"+str(d.strftime('%m'))+"-"+str(d.strftime('%d'))
        paths = natsorted(glob.glob('/Volumes/Data_drive/L2_data/Test/beacon/'+prefix+"/*"))
        print(len(paths),'/Volumes/Data_drive/L2_data/Test/beacon/'+prefix+"/*")
        for p in paths:
            filea = fits.open(p)
            datas.append(filea[0].data)
            headers.append(filea[0].header)
            filea.close()
    return datas,headers



class HoverableLabel(QLabel):
    hover = pyqtSignal(int, float,float)
    click = pyqtSignal(int, float,float,bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event):
        x = event.pos().x()
        relative_x = x / self.width()
        relative_y = event.pos().y() / self.height()
        self.hover.emit(self.index, relative_x,relative_y)

    def mousePressEvent(self, event):
        x = event.pos().x()
        relative_x = x / self.width()
        relative_y = event.pos().y() / self.height()
        if (event.button()== PyQt5.QtCore.Qt.RightButton):
            right = True
        else:
            right = False
        self.click.emit(self.index, relative_x,relative_y,right)

class ImagePlotViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image and Plot Viewer")
        self.size = 600
        self.setGeometry(50, 50, self.size*3, self.size*2)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.main_layout = QVBoxLayout()
        self.image_layout = QHBoxLayout()
        self.main_layout.addLayout(self.image_layout)
        
        events_dates,new_events_list = create_dates_test()

        self.event = new_events_list[int(sys.argv[1])]
        
        
        self.type =sys.argv[2]
        if (self.type=="enhancedbeacon"):
            self.images,self.headers = load_enhanced_beacon(self.event)
            with open("./jplots_final_interpolated/"+self.event[3].strftime("%Y-%m-%d")+".p", 'rb') as f:
                jplot_dict = pickle.load(f)
            self.plot_data = jplot_dict["data"]
        elif(self.type=="beacon"):
            print(("loading beacon"))
            self.images,self.headers = load_beacon(self.event)
            with open("./jplots_final_beacon/"+self.event[3].strftime("%Y-%m-%d")+".p", 'rb') as f:
                jplot_dict = pickle.load(f)
            self.plot_data = jplot_dict["data"]

        elif(self.type=="science"):
            print(("loading science"))
            self.images,self.headers = load_science(self.event)
            with open("./jplots_final_science/"+self.event[3].strftime("%Y-%m-%d")+".p", 'rb') as f:
                jplot_dict = pickle.load(f)
            self.plot_data = jplot_dict["data"]
        elif(self.type =="enhancedbeacon0"):
            self.images,self.headers = load_enhanced_beacon_normal(self.event)
            with open("./jplots_final/"+self.event[3].strftime("%Y-%m-%d")+".p", 'rb') as f:
                jplot_dict = pickle.load(f)
            self.plot_data = jplot_dict["data"]
            print(self.plot_data.shape)
        else:
            print("wrong input type")
            exit()


        dates_headers = []
        for im in self.headers:
            dates_headers.append(datetime.strptime(im["DATE-END"], '%Y-%m-%dT%H:%M:%S.%f'))


        print(dates_headers)

        beacon_earth = []
        beacon_stereo = []
        for d in range(0,len(self.headers),50):
            beacon_earth  += get_body_heliographic_stonyhurst('earth', dates_headers[d:d+50])
            beacon_stereo += get_horizons_coord('STEREO-A', dates_headers[d:d+50])
            
        self.elongations_images = []
        self.final_images = []
        # ax.set_xlim(self.event[2],self.event[5])
        for i in range(0,len(self.images)):
            dif,elong1 = draw_ecliptic(datetime.strptime(self.headers[i]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f'),self.images[i],self.headers[i],self.images[0].shape[0],beacon_earth[i],beacon_stereo[i])
            # if(datetime.strptime(self.headers[i]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')>self.event[2] and datetime.strptime(self.headers[i]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')<self.event[5]):
            self.final_images.append(dif)
            self.elongations_images.append(elong1)

        self.vmin = np.nanmedian(self.plot_data) - 1 * np.nanstd(self.plot_data)
        self.vmax = np.nanmedian(self.plot_data) + 1 * np.nanstd(self.plot_data)
       
        self.tracked = []
        self.tracked_local = []

        if(len(sys.argv)>3):

            if(sys.argv[3]=="reload"):
                files = natsorted(glob.glob("./jplots_pickles/Tracks/"+self.event[3].strftime("%Y-%m-%d")+"/"+sys.argv[2]+"/*.csv"))
                for f in files:
                    with open(f, 'r') as f:
                        for line in csv.DictReader(f):
                            el = float(line['TRACK_DATE'])
                            dat = datetime.strptime(line['ELON'],'%Y-%m-%dT%H:%M:%S')
                            self.tracked.append([dat,el])
                           
                            for i in range(0,len(dates_headers)):
                                if(np.abs((dates_headers[i]-dat).total_seconds()/60.0)<2):
                                    index_x = (np.abs(self.elongations_images[i][self.elongations_images[i].shape[0]//2,:] - el*np.pi/180.0)).argmin()
                                    self.tracked_local.append([i, index_x/self.elongations_images[i].shape[0],(self.elongations_images[i].shape[0]//2)/self.elongations_images[i].shape[0]])
        

                
        self.dates = jplot_dict["dates"]
        self.elongations = jplot_dict["elongations"]

        if(self.dates[0].year<2015):
            self.origin='upper'
        else:
            self.origin='lower'


        self.current_start = len(self.final_images)//2
        self.visible_images = 4

        self.image_labels = []
        for i in range(self.visible_images):
            label = HoverableLabel(self)
            label.setAlignment(Qt.AlignCenter)
            label.hover.connect(self.handle_hover)
            label.click.connect(self.handle_click)
            self.image_labels.append(label)
            self.image_layout.addWidget(label)

        self.figure = plt.figure(figsize=(10, 4))
        self.figure.subplots_adjust(
            top=0.93,
            bottom=0.07,
            left=0.04,
            right=0.99,
            hspace=0.2,
            wspace=0.2
        )
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("button_press_event", self._on_left_click)
        self.main_layout.addWidget(self.canvas)
        self.layout.addLayout(self.main_layout)
        

        self.controls_layout = QVBoxLayout()

        self.button_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.undo_button = QPushButton("Undo")
        self.prev_button.clicked.connect(self.shift_images_backward)
        self.next_button.clicked.connect(self.shift_images_forward)
        self.undo_button.clicked.connect(self.undo_last_click)
        self.button_layout.addWidget(self.prev_button)
        self.button_layout.addWidget(self.next_button)
        self.button_layout.addWidget(self.undo_button)

        self.finish_button = QPushButton("End")
        self.finish_button.clicked.connect(self.finish)
        self.button_layout.addWidget(self.finish_button)

        self.sliders_layout = QHBoxLayout()
        self.slider = QSlider()
        self.slider.setMinimum(5)
        self.slider.setMaximum(250)
        self.slider.setValue(100)
        self.slider.sliderPosition = 100
        self.slider.setOrientation(1)  # 1 is horizontal, 0 is vertical
        self.slider.valueChanged.connect(self.show_value)
        self.contrast_std = 1.0
        self.labelj = QLabel("JPLOT Contrast")
        self.sliders_layout.addWidget(self.labelj)
        self.sliders_layout.addWidget(self.slider)


        self.slider_position = QSlider()
        self.slider_position.setMinimum(0)
        self.slider_position.setMaximum(len(self.final_images))
        self.slider_position.setValue(len(self.final_images)//2)
        self.slider_position.sliderPosition = len(self.final_images)//2
        self.slider_position.setOrientation(1)  # 1 is horizontal, 0 is vertical
        self.slider_position.valueChanged.connect(self.changeindex)
        self.button_layout.addWidget(self.slider_position)

        
        
        self.slider2 = QSlider()
        self.slider2.setMinimum(1)
        self.slider2.setMaximum(1000)
        self.slider2.setValue(100)
        self.slider2.sliderPosition = 100
        self.slider2.setOrientation(1)  # 1 is horizontal, 0 is vertical
        self.slider2.valueChanged.connect(self.show_value2)
        self.contrast_std2 = 100
        self.labeli = QLabel("Images Contrast")
        self.sliders_layout.addWidget(self.labeli)
        self.sliders_layout.addWidget(self.slider2)
        
        self.Hist = True
        self.Histb = QCheckBox("Hist eq.")
        self.Histb.setChecked(True)
        self.Histb.stateChanged.connect(self.hist_changed)
        self.button_layout.addWidget(self.Histb)



        self.medianed = True
        self.checkbox = QCheckBox("Medianed")
        self.checkbox.setChecked(True)
        self.checkbox.stateChanged.connect(self.checkbox_changed)
        self.button_layout.addWidget(self.checkbox)


        self.medianjplot = np.nanmedian(self.plot_data)
        self.stdjplot = np.nanstd(self.plot_data)


        self.controls_layout.addLayout(self.button_layout)
        self.controls_layout.addLayout(self.sliders_layout)
        self.layout.addLayout(self.controls_layout)
        self.current_image = len(self.final_images)//2
       
        self.update_display()



    def hist_changed(self,state):
        self.Hist = bool(state)
        self.update_display()

    def _on_left_click(self,event):
        print(matplotlib.dates.num2date(event.xdata), event.ydata,event)
        if(event.button == 3):
            to_remove=[]
            for i in range(0,len(self.tracked)):
                date = self.tracked[i][0].replace(tzinfo=timezone.utc)
                if(np.abs((date-matplotlib.dates.num2date(event.xdata).replace(tzinfo=timezone.utc)).total_seconds())/60.0 <3.0 and np.abs(self.tracked[i][1]- event.ydata)<0.5):
                    to_remove.append(i)
            self.tracked = [i for j, i in enumerate(self.tracked) if j not in to_remove]
            self.tracked_local = [i for j, i in enumerate(self.tracked_local) if j not in to_remove]
            self.update_display()

    def show_value(self,value):
        self.contrast_std = value/100
        self.update_display()

    def checkbox_changed(self,state):
        self.medianed = bool(state)
        self.update_display()

    def show_value2(self,value):
        self.contrast_std2 = value
        # if(self.type=="enhancedbeacon" or self.type=="enhancedbeacon0"):
        #     self.contrast_std2 = self.contrast_std2 /200000
        # else:
        self.update_display()

    def changeindex(self,value):
        self.current_start = value
        self.update_display()


    def fnc2min(self,params,x,data):
        pw   = params["pw"].value
        adj1 = params['adj1'].value
        adj2 = params['adj2'].value

        model = adj1 * np.power(x+adj2,pw)
        return model- data

    def update_display(self):
        
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.vmin = self.medianjplot - self.contrast_std * self.stdjplot
        self.vmax = self.medianjplot + self.contrast_std * self.stdjplot
        
        if(self.medianed):
            disp_Data  = np.median(self.plot_data,2)
        else:
            a,b,c = self.plot_data.shape
            disp_Data = self.plot_data.reshape((a,b*c))

        disp_Data = np.nan_to_num(disp_Data,0.0)

       
      
        ax.imshow(disp_Data, cmap='gray', aspect='auto',interpolation='none',origin=self.origin, extent=[self.dates[0], self.dates[-1],self.elongations[0] , self.elongations[1]],vmin=self.vmin,vmax=self.vmax)
        
       
        loc_ticks= int(np.ceil(((self.dates[-1]-self.dates[0]).total_seconds()/(60*60*24)*1/7)))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 12), interval=loc_ticks))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6), interval=loc_ticks))


        
        for i, label in enumerate(self.image_labels):
            image_index = self.current_start + i
            if image_index < len(self.final_images):
                w,h,c = self.final_images[image_index].shape
                # qimg  = QImage(self.final_images[image_index], w, h, w, QtGui.QImage.Format_Indexed8) 
                
                
                img = self.final_images[image_index].copy()

               
                vmin  = np.median(img)  - 2.5 * np.std(img)
                vmax  = np.median(img)  + 2.5 * np.std(img)
                img[img>vmax] = vmax
                img[img<vmin] = vmin
                img = (img-vmin)/(vmax-vmin)
                img[img>1.0] = 1.0


                intercept = -(0.5 * self.contrast_std2/100) + 0.5
                img   = self.contrast_std2/100 * img  + intercept
              


                img = np.where(img > 1,1,img)
                img = np.where(img < 0,0,img)

                #     img = exposure.equalize_adapthist(img,clip_limit=self.contrast_std2/1000,kernel_size=img.shape[0]//10)
                



                qImg = PyQt5.QtGui.QImage((img*255.0).astype(np.uint8), w, h,3*w, QImage.Format_RGB888)
                pixmap = QPixmap(qImg)
                label.setPixmap(pixmap.scaled(self.size, self.size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                label.index = image_index
                ax.axvline(x=datetime.strptime(self.headers[image_index]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f'), color='g')
                
                for t in self.tracked_local:
                    if (t[0]==image_index):
                        painter = PyQt5.QtGui.QPainter(label.pixmap())
                        pen = PyQt5.QtGui.QPen()
                        pen.setWidth(12)
                        pen.setColor(PyQt5.QtGui.QColor('yellow'))
                        painter.setPen(pen)
                        painter.drawPoint(int(t[1]*self.size), int(t[2]*self.size))
                        painter.end()

                
              
                label.show()
            else:
                label.hide()

        

        self.prev_button.setEnabled(self.current_start > 0)
        self.next_button.setEnabled(self.current_start + self.visible_images < len(self.final_images))


        if(len(self.tracked)>0):
            ax.scatter(np.asarray(self.tracked)[:,0],np.asarray(self.tracked)[:,1], color='b')
            
        
        starttime = time.time()
        self.canvas.draw()
        print("time plotting images",time.time()-starttime)

    def select_image(self, index):
        self.current_image = index
        self.update_display()

    def handle_click(self, index, relative_x,relative_y,button):
        if(button):
            to_remove = []
            print(self.tracked_local)
            for i in range(0,len(self.tracked_local)):
                distance = np.sqrt((relative_x-self.tracked_local[i][1])**2 + (relative_y-self.tracked_local[i][2])**2)
                if(distance<0.01 and index==self.tracked_local[i][0]):
                    to_remove.append(i)
            self.tracked = [i for j, i in enumerate(self.tracked) if j not in to_remove]
            self.tracked_local = [i for j, i in enumerate(self.tracked_local) if j not in to_remove]
            self.update_display()
        else:
            self.tracked.append([datetime.strptime(self.headers[index]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f')
                                ,self.elongations_images[index][int(relative_y*self.images[0].shape[0]),int(relative_x*self.images[0].shape[0])]*180/np.pi])
            self.tracked_local.append([index, relative_x,relative_y])
            self.update_display()

        
        
    def get_index(self,date):
        for d in self.dates:
            if(np.abs((d-date).total_seconds())<200):
                return d

    def handle_hover(self, index, relative_x,relative_y):
        

      
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        self.vmin = np.nanmedian(self.plot_data) - self.contrast_std * np.nanstd(self.plot_data)
        self.vmax = np.nanmedian(self.plot_data) + self.contrast_std * np.nanstd(self.plot_data)
        
        if(self.medianed):
            disp_Data  = np.median(self.plot_data,2)
        else:
            a,b,c = self.plot_data.shape
            disp_Data = self.plot_data.reshape((a,b*c))


       
        ax.imshow(disp_Data , cmap='gray', aspect='auto',interpolation='none',origin=self.origin, extent=[self.dates[0], self.dates[-1],self.elongations[0] , self.elongations[1]],vmin=self.vmin,vmax=self.vmax)
        loc_ticks= int(np.ceil(((self.dates[-1]-self.dates[0]).total_seconds()/(60*60*24)*1/7)))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6), interval=loc_ticks))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 2), interval=loc_ticks))
        
        
        # ax.pcolor(self.dates,np.linspace(self.elongations[0],self.elongations[1],self.plot_data.shape[0]),self.plot_data,cmap='gray')
        y = self.elongations_images[index][int(relative_y*self.images[0].shape[0]),int(relative_x*self.images[0].shape[0])]*180/np.pi
        # ax.scatter(self.dates_tracks_tracked,self.elongs_tracked,c="pink")

        ax.scatter(x=datetime.strptime(self.headers[index]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f'),y=y, color='r')
        ax.axvline(x=datetime.strptime(self.headers[index]["DATE-END"],'%Y-%m-%dT%H:%M:%S.%f'), color='g')
        # ax.set_yscale('log')

        for t in self.tracked:
            ax.scatter(x=t[0],y=t[1], color='b')
        


        self.canvas.draw()

    def shift_images_forward(self):
        if self.current_start + self.visible_images < len(self.images):
            self.current_start += 1
            self.slider_position.tracking = True
            self.slider_position.sliderPosition = int(self.current_start)
            self.slider_position.update()
            self.slider_position.repaint()

            self.update_display()

    def shift_images_backward(self):
        if self.current_start > 0:
            self.current_start -= 1
            self.slider_position.tracking = True
            self.slider_position.sliderPosition = int(self.current_start)
            self.slider_position.update()
            self.slider_position.repaint()
            self.update_display()

    def undo_last_click(self):
        self.tracked = self.tracked[:-1]
        self.tracked_local  = self.tracked_local[:-1]
        self.update_display()

    def finish(self):
        print("end")
        print(self.tracked)
        dates = []
        elongations = []
        for i,d in enumerate(self.tracked):
            dates.append(d[1])
            elongations.append(d[0])

        pd_data = {'TRACK_DATE':dates,'ELON':elongations,'ELON_STDD':np.zeros((len(dates))),'SC':['A' for x in range(len(elongations))]}

        newpath = "jplots_pickles/Tracks/"+self.event[3].strftime("%Y-%m-%d")+"/"+sys.argv[2]
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        df = pd.DataFrame(pd_data, columns=['TRACK_DATE', 'ELON', 'SC', 'ELON_STDD'])
        df.to_csv(newpath+"/"+ str(sys.argv[1])+ '.csv', index=False, date_format='%Y-%m-%dT%H:%M:%S')
        self.close() 

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet("QLabel{font-size: 20pt;}\
                      QPushButton{font-size: 20pt;}\
                      QSlider{font-size: 20pt;}\
                      QCheckBox{font-size: 20pt;}")


    viewer = ImagePlotViewer()
    viewer.show()
    sys.exit(app.exec_())






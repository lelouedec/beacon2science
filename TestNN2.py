import torch
import numpy as np 
import torchvision
torchvision.disable_beta_transforms_warning()
from natsort import natsorted 
import glob 
from datetime import datetime,timedelta

import matplotlib

matplotlib.rcParams.update({'font.size': 20})

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import mean_squared_error as MSE
from skimage.transform import resize
import pickle




from skimage.transform import resize
from skimage import exposure


import tqdm


import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)




import models.RIFE as RIFE

def normalize(img,rangev=2.5):
    
    vmax = np.median(img)+rangev*np.std(img)
    vmin = np.median(img)-rangev*np.std(img)

    img[img>vmax] = vmax
    img[img<vmin] = vmin

    img = (img-vmin)/( (vmax-vmin))

    img[img>1.0] = 1.0
    return img

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



def load_final_enhanced(dates,folder = "E-beacon"):
    imgs_list    = []
    imgs_headers = []
    imgs_times = []
    for e in dates:
        prefix=str(e.strftime('%Y'))+"-"+str(e.strftime('%m'))+"-"+str(e.strftime('%d'))
        imgs = natsorted(glob.glob("/media/lelouedec/DATA/"+folder+"/"+prefix+"*"))
        for im in imgs:
            with open(im, 'rb') as f:
                jplot_dict = pickle.load(f)
                imgs_list.append(jplot_dict['data'])
                imgs_headers.append(jplot_dict['header'])
                time = datetime.strptime(im.split("/")[-1][:-3], '%Y-%m-%dT%H-%M-%S')
                imgs_times.append(time)

    return imgs_list,imgs_headers,imgs_times



device = torch.device("cpu")
if(torch.backends.mps.is_available()):
    device = torch.device("mps")
elif(torch.cuda.is_available()):
    device = torch.device("cuda:1")

model = RIFE.Model()
model.flownet.to(device)

model.load_model("PAPER_NN2.pth")

_,new_events_list = create_dates_test()
for i in range(0,len(new_events_list)):
    e = new_events_list[i]
    if(e[0].year<2015):
        origin = 'upper'
    else:
        origin = 'lower'

    differences,headers,times = load_final_enhanced(e)
    differences2 = []
    headers2 = []

    with torch.no_grad():
        for p in tqdm.tqdm(range(0,len(differences)-1,1)):
            

            time1 = times[p]
            time2 = times[p+1]
            diff_time = (time2-time1).total_seconds()

            if(diff_time/3600<=4.0):
                
                data1 = differences[p]
                data2 = differences[p+1]
                

                if(time1.year<=2015):
                    data1 = np.fliplr(data1)
                    data2 = np.fliplr(data2)

                S1 = torch.tensor(data1.copy()).float().unsqueeze(0).unsqueeze(1).to(device)
                S2 = torch.tensor(data2.copy()).float().unsqueeze(0).unsqueeze(1).to(device)


                output1 = model.inference(S1,S2,timestep=torch.tensor([0.33]))
                output2 = model.inference(S1,S2,timestep=torch.tensor([0.66]))
        

                output1 = output1[0,0,:,:].cpu().numpy()
                output2 = output2[0,0,:,:].cpu().numpy()

                header1 = headers[p]
                header2 = headers[p+1]

                hdr3 = header2.copy()
                hdr4 = header2.copy()

                timeout1 = time1+timedelta(minutes=40)
                timeout2 = time1+timedelta(minutes=80)
                
                hdr3["DATE-END"] = timeout1.strftime('%Y-%m-%dT%H:%M:%S')+".000"
                hdr4["DATE-END"] = timeout2.strftime('%Y-%m-%dT%H:%M:%S')+".000"

                output1 = exposure.match_histograms(output1, data1)
                output2 = exposure.match_histograms(output2, data1)

                if(time1.year<=2015):
                    data1 = np.fliplr(data1)
                    data2 = np.fliplr(data2)
                    output1 = np.fliplr(output1)
                    output2 = np.fliplr(output2)

              


                to_save = {'data': data1,
                          'header':header1
                        }
                pickle.dump(to_save, open("/media/lelouedec/DATA/IE-beacon/"+time1.strftime("%Y-%m-%dT%H-%M-%S").replace(":","-")+".p", 'wb'))

                to_save = {'data': output1,
                          'header':hdr3
                        }
                pickle.dump(to_save, open("/media/lelouedec/DATA/IE-beacon/"+timeout1.strftime("%Y-%m-%dT%H-%M-%S").replace(":","-")+".p", 'wb'))

                to_save = {'data': output2,
                          'header':hdr4
                        }
                pickle.dump(to_save, open("/media/lelouedec/DATA/IE-beacon/"+timeout2.strftime("%Y-%m-%dT%H-%M-%S").replace(":","-")+".p", 'wb'))

                to_save = {'data': data2,
                          'header':header2
                        }
                pickle.dump(to_save, open("/media/lelouedec/DATA/IE-beacon/"+time2.strftime("%Y-%m-%dT%H-%M-%S").replace(":","-")+".p", 'wb'))
            


               
            

imgs_paths = natsorted(glob.glob("/media/lelouedec/DATA/IE-beacon/*"))

PSNR_enhanced = []
MSE_enhanced = []
SSIM_enhanced = []
magnitudes_enhanced= []
contrast = 5
with torch.no_grad():
    for p in tqdm.tqdm(range(0,len(imgs_paths),1)):
              
            with open(imgs_paths[p], 'rb') as f:
                jplot_dict = pickle.load(f)
                enhanced = jplot_dict['data']
                hdr = jplot_dict['header']

            if(len(glob.glob("/media/lelouedec/DATA/rdifs_science_120/"+imgs_paths[p].split("/")[-1][:-4]+"*")) ==1 and  len(glob.glob("/media/lelouedec/DATA/E-beacon/"+imgs_paths[p].split("/")[-1][:-4]+"*"))==0 ):
                science_path = glob.glob("/media/lelouedec/DATA/rdifs_science_120/"+imgs_paths[p].split("/")[-1][:-4]+"*")[0]
                with open(science_path, 'rb') as f:
                    jplot_dict = pickle.load(f)
                    sciencerdif = resize(normalize(jplot_dict['data']),(512,512),preserve_range=True)

                f_transform = np.fft.fft2(enhanced)
                f_transform_shifted = np.fft.fftshift(f_transform)  # Shift zero frequency to center
                magnitude_spectrum_enhanced = np.log1p(np.abs(f_transform_shifted))

                magnitudes_enhanced.append(magnitude_spectrum_enhanced)


                intercept    = -(0.5 * contrast) + 0.5
                enhanced           = contrast * enhanced  + intercept
                sciencerdif  = contrast * sciencerdif + intercept
           
                sciencerdif = np.where(sciencerdif > 1,1,sciencerdif)
                sciencerdif = np.where(sciencerdif < 0,0,sciencerdif)

                enhanced = np.where(enhanced > 1,1,enhanced)
                enhanced = np.where(enhanced < 0,0,enhanced)




                psnr_enhanced,mse_enhanced   = PSNR(sciencerdif,enhanced,data_range=enhanced.max()-enhanced.min()),MSE(sciencerdif,enhanced)    

                PSNR_enhanced.append(psnr_enhanced)
                MSE_enhanced.append(mse_enhanced)
                
                ssim_enhanced,imgssime   = ssim(sciencerdif,enhanced,full=True,gaussian_weights=True,data_range=enhanced.max()-enhanced.min())
                SSIM_enhanced.append(ssim_enhanced)

print(np.array(PSNR_enhanced).mean())
print(np.array(MSE_enhanced).mean())
print(np.array(SSIM_enhanced).mean())
np.save("/media/lelouedec/DATA/magnitude_interpolated.npy",np.array(magnitudes_enhanced))
            
              
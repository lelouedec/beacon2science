import torch
from ESRGAN import *
import torch
from torch import nn,optim
# from attention import * 
import b2s_dataset
import matplotlib.pyplot as plt 
import numpy as np 
import torchvision
torchvision.disable_beta_transforms_warning()
from kornia.geometry.transform import translate
import ESRGAN
from math import log10, sqrt
import cv2
# import Nextlevel
#https://wichtelmania.com/en/w/CcESsMijwlNn4ZCcEWiTLo6bL410
#from backbones_unet.model.unet import Unet
import kornia
from kornia.enhance.equalization import equalize_clahe
import matplotlib.cm as cm
import unet2
import json
import sys 


from torch.utils.tensorboard import SummaryWriter



def train_discriminator(discriminator,data,sr,adversarial_looser,device,target):
  

    gt_output = discriminator(target.detach().clone())
   
    sr_output = discriminator(sr.detach().clone())

    real_label = torch.full([target.shape[0], 1], 1.0, dtype=torch.float, device=device,requires_grad=False)
    fake_label = torch.full([target.shape[0], 1], 0.0, dtype=torch.float, device=device,requires_grad=False)


    adversarial_loss  = (adversarial_looser(gt_output-sr_output.mean(0,keepdim=True), real_label) +\
                        adversarial_looser(sr_output-gt_output.mean(0,keepdim=True), fake_label))/2 
    


    return adversarial_loss 




def train(config):
    device = torch.device("cpu")
    if(torch.backends.mps.is_available()):
        device = torch.device("mps")
    elif(torch.cuda.is_available()):
        device = torch.device("cuda")

    
    print("THIS WILL RUN ON DEVICE:", device)

    model = unet2.ResUnet(1,full_size=config["full_size"])
    discriminator = ESRGAN.Discriminator(1,1,64,full_size=config["full_size"])

    if(config["load_models"]==True):
        model.load_state_dict(torch.load(config["generator"],map_location=torch.device('cpu')))
        discriminator.load_state_dict(torch.load(config["discriminator"],map_location=torch.device('cpu')))


    if(torch.cuda.device_count() >1):
        model = torch.nn.DataParallel(model)
        discriminator = torch.nn.DataParallel(discriminator)

    model.to(device)
    discriminator.to(device)

    minibacth = config["minibacth"]
    full_size = config["full_size"]

    dataset = b2s_dataset.FinalDataset(256,full_size,config["data_path"],True,False)
    dataset_validation = b2s_dataset.FinalDataset(256,full_size,config["data_path"],False,True)

    # dataset = b2s_dataset.FinalDataset(256,full_size,"/Volumes/Data_drive/",True,False)
    # dataset_validation = b2s_dataset.FinalDataset(256,full_size,"/Volumes/Data_drive/",False,True)

    dataloader = torch.utils.data.DataLoader(
                                                dataset,
                                                batch_size=minibacth,
                                                shuffle=True,
                                                num_workers=int(minibacth/2),
                                                pin_memory=False
                                            )


    dataloader_validation = torch.utils.data.DataLoader(
                                                dataset_validation,
                                                batch_size=minibacth,
                                                shuffle=True,
                                                num_workers=int(minibacth/2),
                                                pin_memory=False
                                            )
    

    g_optimizer = optim.Adam(model.parameters(),config["lrgen"])
    d_optimizer = optim.Adam(discriminator.parameters(),config["lrdisc"])

    if(config["load_optim"]):
        g_optimizer.load_state_dict(torch.load(config["goptim"],map_location=torch.device('cpu')))
        d_optimizer.load_state_dict(torch.load(config["doptim"],map_location=torch.device('cpu')))
    
    g_scheduler = optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, patience=5)
    d_scheduler = optim.lr_scheduler.ReduceLROnPlateau(d_optimizer, patience=5)



    adversarial_looser = nn.BCEWithLogitsLoss()
    if(config["pxl_loss"]=="SSIM"):
        pixel_looser  = kornia.losses.MS_SSIMLoss(reduction='none').to(device)
    elif(config["pxl_loss"]=="MSE"):
        pixel_looser = nn.MSELoss(reduction="mean")
    elif(config["pxl_loss"]=="L1"):
        pixel_looser = nn.L1Loss(reduction="mean")
   

    # content_looser = ESRGAN.ContentLoss("vgg19",False,1,None,["features.34"],[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]).to(device)

    pixel_weight = config["pixel_weight"]
    adversarial_weight = config["adversarial_weight"]
    diff_weight = config["diff_weight"]
    if(config["warmupiter"]):
        warmupiter = int(dataset.__len__()*4/minibacth)


    g_losses  = []
    a_losses  = []
    p_losses  = []
    a_losses2 = []


    g_losses_validation  = []
    a_losses_validation  = []

    best_validation = config["best_validation"]



    cnt = 0
    writer = SummaryWriter(log_dir=config["logdir"])
    for i in range(config["startepoch"],config["endepoch"]):
        g_loss  = 0.0
        a_loss  = 0.0
        a_loss2 = 0.0
        p_loss  = 0.0
        for data in dataloader:

            ########################################
            ########TRAINING GENERATOR FIRST########
            ########################################
            
            LR1 = data["LR1"].to(device)
            LR2 = data["LR2"].to(device)
            HR1 = data["HR1"].to(device)
            HR2 = data["HR2"].to(device)


            D1  = data["diff1"].to(device)


            g_optimizer.zero_grad()
            sr1 = model(LR1,LR2)

            sr2 = model(LR2,LR1)


            real_label = torch.full([sr1.shape[0], 1], 1.0, dtype=torch.float, device=device,requires_grad=False)


            shift  = data["tr1"].float().to(device)
            if(config["full_size"]==512):
                shift = shift/2

            difference1 = sr2 - translate(sr1,shift,mode='bilinear',padding_mode='border')


            diff_loss = (difference1-D1)**2

            
            loss =  pixel_weight*((0.5*pixel_looser(sr1.float(),HR1)+0.5*pixel_looser(sr2.float(),HR2))).mean()
            if(config["diff_loss"]):
                loss = loss + diff_weight*(diff_loss).mean() 

             
            if(cnt>warmupiter):
                gt_output = discriminator(HR1)
                sr_output = discriminator(sr1)
                adversarial_loss=  adversarial_looser(sr_output-gt_output.mean(0,keepdim=True), real_label)
                
                loss+= adversarial_weight*adversarial_loss
                
                gt_output = discriminator(HR2)
                sr_output = discriminator(sr2)
                adversarial_loss2 =  adversarial_looser(sr_output-gt_output.mean(0,keepdim=True), real_label)
                
                loss+= adversarial_weight*adversarial_loss2
                

            loss.backward()
            g_optimizer.step()
     


            ########################################
            ######## TRAINING DISCRIMINATOR ########
            ########################################

            
            d_optimizer.zero_grad()
            d_loss = train_discriminator(discriminator,data,sr1,adversarial_looser,device,HR1)
            d_loss+= train_discriminator(discriminator,data,sr2,adversarial_looser,device,HR2)
            d_loss.backward()
            d_optimizer.step()


            g_loss += loss.item()
            p_loss += loss.item()
            a_loss2+= d_loss.item()
                
            cnt+=1

        g_losses.append(g_loss/dataset.__len__())
        a_losses.append(a_loss/dataset.__len__())
        p_losses.append(p_loss/dataset.__len__())
        a_losses2.append(a_loss2/dataset.__len__())


        g_loss_validation = 0.0
        a_loss_validation = 0.0
        with torch.no_grad():
            for data in dataloader_validation:
                LR1 = data["LR1"].to(device)
                LR2 = data["LR2"].to(device)
                HR1 = data["HR1"].to(device)
                HR2 = data["HR2"].to(device)


                D1  = data["diff1"].to(device)
                sr1 = model(LR1,LR2)
                sr2 = model(LR2,LR1)

                

                shift  = data["tr1"].float().to(device)
                if(config["full_size"]==512):
                    shift = shift/2
                difference1 = sr2 - translate(sr1,shift,mode='bilinear',padding_mode='border')
 
            
                
                        

                loss =  pixel_weight*((0.5*pixel_looser(sr1.float(),HR1)+0.5*pixel_looser(sr2.float(),HR2))).mean()
               
               
                g_loss_validation += loss.item()

            # scheduler.step(g_loss_validation/dataset_validation.__len__())
            g_losses_validation.append(g_loss_validation/dataset_validation.__len__())
            a_losses_validation.append(a_loss_validation/dataset_validation.__len__())

        #g_scheduler.step(g_loss_validation/dataset_validation.__len__())
        #d_scheduler.step(g_loss_validation/dataset_validation.__len__())
        


        writer.add_scalar('gLoss/train', np.array(g_losses)[-1], i)
        writer.add_scalar('gLoss/test',  np.array(g_losses_validation)[-1], i)
        writer.add_scalar('aLoss/train', np.array(a_losses2)[-1], i)
        writer.add_scalar('aLoss/test',  np.array(a_losses_validation)[-1], i)

        writer.add_image('fake/img1', sr1[0,0,:,:], i, dataformats='HW')
        writer.add_image('fake/img2', sr2[0,0,:,:], i, dataformats='HW')

        writer.add_image('HR/img1', HR1[0,0,:,:], i, dataformats='HW')
        writer.add_image('HR/img2', HR2[0,0,:,:], i, dataformats='HW')

        writer.add_image('LR/img1', LR1[0,0,:,:], i, dataformats='HW')
        writer.add_image('LR/img2', LR2[0,0,:,:], i, dataformats='HW')


        if(len(g_losses_validation)>1):
            if(g_losses_validation[-1]<best_validation):
                torch.save(model.module.state_dict(), config["generator"])
                torch.save(discriminator.module.state_dict(),config["discriminator"])
                torch.save(g_optimizer.state_dict(),config["goptim"])
                torch.save(d_optimizer.state_dict(),config["doptim"])
                best_validation = g_losses_validation[-1]
        else:
            torch.save(model.module.state_dict(), config["generator"])
            torch.save(discriminator.module.state_dict(),config["discriminator"])
            torch.save(g_optimizer.state_dict(),config["goptim"])
            torch.save(d_optimizer.state_dict(),config["doptim"])



if __name__ == "__main__":
    with open(sys.argv[1]) as handle:
        config = json.load(handle)
    train(config)



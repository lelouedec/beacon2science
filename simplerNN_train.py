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

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def train_discriminator(discriminator,data,sr,adversarial_looser,device,target):
  

    gt_output = discriminator(target.detach().clone())
   
    sr_output = discriminator(sr.detach().clone())

    real_label = torch.full([target.shape[0], 2], 1.0, dtype=torch.float, device=device,requires_grad=False)
    fake_label = torch.full([target.shape[0], 2], 0.0, dtype=torch.float, device=device,requires_grad=False)


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

    model = unet2.ResUnet(2)
    discriminator = ESRGAN.Discriminator(2,2,64)

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

    dataset = b2s_dataset.FinalDataset(256,full_size,"/gpfs/data/fs72241/lelouedecj/",True,False)
    dataset_validation = b2s_dataset.FinalDataset(256,full_size,"/gpfs/data/fs72241/lelouedecj/",False,True)

    # dataset = b2s_dataset.FinalDataset(256,full_size,"/Volumes/Data_drive/",True,False)
    # dataset_validation = b2s_dataset.FinalDataset(256,full_size,"/Volumes/Data_drive/",False,True)

    # dataset = b2s_dataset.CombinedDataloader3(256,512,"../finals_test",True,False)
    # dataset_validation = b2s_dataset.CombinedDataloader3(256,512,"../finals_test",False,True)

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
    writer = SummaryWriter()
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
            sr = model(LR1,LR2)

            sr1 = sr[:,0,:,:].unsqueeze(1)
            sr2 = sr[:,1,:,:].unsqueeze(1)


            real_label = torch.full([sr1.shape[0], 2], 1.0, dtype=torch.float, device=device,requires_grad=False)


            difference1 = sr2 - translate(sr1,data["tr1"].float().to(device),mode='bilinear',padding_mode='border')


            diff_loss = (difference1-D1)**2

            
            loss =  pixel_weight*((0.5*pixel_looser(sr1.float(),HR1)+0.5*pixel_looser(sr2.float(),HR2))).mean()
            if(config["diff_loss"]):
                loss = loss + diff_weight*(diff_loss).mean() 

             
            if(cnt>warmupiter):
                gt_output = discriminator(torch.cat([HR1,HR2],1))
                sr_output = discriminator(sr)
                
                adversarial_loss=  adversarial_looser(sr_output-gt_output.mean(0,keepdim=True), real_label) 
                loss+= adversarial_weight*adversarial_loss

            loss.backward()
            g_optimizer.step()
     


            ########################################
            ######## TRAINING DISCRIMINATOR ########
            ########################################

            
            d_optimizer.zero_grad()
            d_loss = train_discriminator(discriminator,data,sr,adversarial_looser,device,torch.cat([HR1,HR2],1))
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
                sr = model(LR1,LR2)

                sr1 = sr[:,0,:,:].unsqueeze(1)
                sr2 = sr[:,1,:,:].unsqueeze(1)

                difference1 = sr2 - translate(sr1,data["tr1"].float().to(device),mode='bilinear',padding_mode='border')
 
            
                gt_output = discriminator(torch.cat([HR1,HR2],1))
                sr_output = discriminator(sr)


                real_label = torch.full([sr1.shape[0], 2], 1.0, dtype=torch.float, device=device,requires_grad=False)
                
                adversarial_loss=  adversarial_looser(sr_output-gt_output.mean(0,keepdim=True), real_label) 

                diff_loss = (difference1-D1)**2
                        

                loss =  pixel_weight*((0.5*pixel_looser(sr1.float(),HR1)+0.5*pixel_looser(sr2.float(),HR2))).mean()
                if(config["diff_loss"]):
                    loss = loss + diff_weight*(diff_loss).mean() 

               
                g_loss_validation += loss.item()
                a_loss_validation += adversarial_loss.item()

            # scheduler.step(g_loss_validation/dataset_validation.__len__())
            g_losses_validation.append(g_loss_validation/dataset_validation.__len__())
            a_losses_validation.append(a_loss_validation/dataset_validation.__len__())

        g_scheduler.step(g_loss_validation/dataset_validation.__len__())
        d_scheduler.step(g_loss_validation/dataset_validation.__len__())
        


        writer.add_scalar('gLoss/train', np.array(g_losses)[-1], i)
        writer.add_scalar('gLoss/test',  np.array(g_losses_validation)[-1], i)
        writer.add_scalar('aLoss/train', np.array(a_losses)[-1], i)
        writer.add_scalar('aLoss/test',  np.array(a_losses_validation)[-1], i)

        writer.add_image('fake/img1', sr1[0,0,:,:], i, dataformats='HW')
        writer.add_image('fake/img2', sr1[0,0,:,:], i, dataformats='HW')

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
            torch.save(g_optimizer.state_dict(),config["gopti"])
            torch.save(d_optimizer.state_dict(),config["doptim"])



if __name__ == "__main__":
    with open(sys.argv[1]) as handle:
        config = json.load(handle)
    train(config)



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




def train():
    device = torch.device("cpu")
    if(torch.backends.mps.is_available()):
        device = torch.device("mps")
    elif(torch.cuda.is_available()):
        device = torch.device("cuda")

    
    print("THIS WILL RUN ON DEVICE:", device)

    # sudo rmmod nvidia_uvm
    # sudo modprobe nvidia_uvm


    model = unet2.ResUnet(2)
    discriminator = ESRGAN.Discriminator(2,2,64)

    model.load_state_dict(torch.load("gan_gen.pth"))
    discriminator.load_state_dict(torch.load("gan_disc.pth"))


    if(torch.cuda.device_count() >1):
        model = torch.nn.DataParallel(model)
        discriminator = torch.nn.DataParallel(discriminator)

    model.to(device)
    discriminator.to(device)

    minibacth = 8
    full_size = 1024

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
    

    g_optimizer = optim.Adam(model.parameters(),1e-5)
    d_optimizer = optim.Adam(discriminator.parameters(),1e-6)

    g_optimizer.load_state_dict(torch.load("gopti.pth"))
    d_optimizer.load_state_dict(torch.load("dopti.pth"))
    
    # g_scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=60, gamma=0.1)
    # d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=60, gamma=0.5)



    adversarial_looser = nn.BCEWithLogitsLoss()
    pixel_looser  = kornia.losses.MS_SSIMLoss(reduction='none').to(device) #nn.MSELoss(reduction="mean")
   

    content_looser = ESRGAN.ContentLoss("vgg19",False,1,None,["features.34"],[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]).to(device)

    content_weight = 1
    pixel_weight = 1
    adversarial_weight = 0.005
    diff_weight = 20
    tv_weight = 10
    warmupiter = int(dataset.__len__()*4/minibacth)


    g_losses  = []
    a_losses  = []
    p_losses  = []
    a_losses2 = []


    g_losses_validation  = []
    a_losses_validation  = []

    best_validation = 5.154

    cnt = 0
    writer = SummaryWriter()
    for i in range(50,200):
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
            distances = data["distances"].to(device)


            D1  = data["diff1"].to(device)


            g_optimizer.zero_grad()
            sr = model(LR1,LR2)

            sr1 = sr[:,0,:,:].unsqueeze(1)
            sr2 = sr[:,1,:,:].unsqueeze(1)


            real_label = torch.full([sr1.shape[0], 2], 1.0, dtype=torch.float, device=device,requires_grad=False)


            difference1 = sr2 - translate(sr1,data["tr1"].float().to(device),mode='bilinear',padding_mode='border')


            diff_loss = (difference1-D1)**2

                        
            loss = diff_weight*(distances* diff_loss).mean() + pixel_weight*(distances*(0.5*pixel_looser(sr1.float(),HR1)+0.5*pixel_looser(sr2.float(),HR2))).mean()

             
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
                distances = data["distances"].to(device)


                D1  = data["diff1"].to(device)
                sr = model(LR1,LR2)

                sr1 = sr[:,0,:,:].unsqueeze(1)
                sr2 = sr[:,1,:,:].unsqueeze(1)

                difference1 = sr2 - translate(sr1,data["tr1"].float().to(device),mode='bilinear',padding_mode='border')
                difference2 = sr1 - translate(sr2,-1*data["tr1"].float().to(device),mode='bilinear',padding_mode='border')
                
                                
                
            
                gt_output = discriminator(torch.cat([HR1,HR2],1))
                sr_output = discriminator(sr)


                real_label = torch.full([sr1.shape[0], 2], 1.0, dtype=torch.float, device=device,requires_grad=False)
                
                adversarial_loss=  adversarial_looser(sr_output-gt_output.mean(0,keepdim=True), real_label) 

                diff_loss = (difference1-D1)**2
                        
                loss = diff_weight*(distances* diff_loss).mean() + pixel_weight*(distances*(0.5*pixel_looser(sr1.float(),HR1)+0.5*pixel_looser(sr2.float(),HR2))).mean()

               
                g_loss_validation += loss.item()
                a_loss_validation += adversarial_loss.item()

            # scheduler.step(g_loss_validation/dataset_validation.__len__())
            g_losses_validation.append(g_loss_validation/dataset_validation.__len__())
            a_losses_validation.append(a_loss_validation/dataset_validation.__len__())

        # g_scheduler.step()
        # d_scheduler.step()
        # print(g_losses[-1],g_losses_validation[-1])
        
        outpt  = sr1[0,0,:,:].clone().detach().cpu().numpy()
        outpt2 = sr2[0,0,:,:].clone().detach().cpu().numpy()


        writer.add_scalar('gLoss/train', np.array(g_losses)[-1], i)
        writer.add_scalar('gLoss/test',  np.array(g_losses_validation)[-1], i)
        writer.add_scalar('aLoss/train', np.array(a_losses)[-1], i)
        writer.add_scalar('aLoss/test',  np.array(a_losses_validation)[-1], i)

        writer.add_image('fake img1', sr1[0,0,:,:], i, dataformats='HW')
        writer.add_image('fake img2', sr1[0,0,:,:], i, dataformats='HW')

        writer.add_image('HR img1', HR1[0,0,:,:], i, dataformats='HW')
        writer.add_image('HR img2', HR2[0,0,:,:], i, dataformats='HW')

        writer.add_image('LR img1', LR1[0,0,:,:], i, dataformats='HW')
        writer.add_image('LR img2', LR2[0,0,:,:], i, dataformats='HW')



        writer.add_image('diff  HR img1', data["diff2"][0][0], i, dataformats='HW')
        writer.add_image('diff  HR img2', data["diff1"][0][0], i, dataformats='HW')

        writer.add_image('diff fake img1', difference2[0][0], i, dataformats='HW')
        writer.add_image('diff fake img2', difference1[0][0], i, dataformats='HW')

        #writer.add_image('diff LR img1', data["diff2_b"][0][0], i, dataformats='HW')
        #writer.add_image('diff LR img2', data["diff1_b"][0][0], i, dataformats='HW')


        if(len(g_losses_validation)>1):
            if(g_losses_validation[-1]<best_validation):
                torch.save(model.module.state_dict(), "gan_gen.pth")
                torch.save(discriminator.module.state_dict(),"gan_disc.pth")
                torch.save(g_optimizer.state_dict(),"gopti.pth")
                torch.save(d_optimizer.state_dict(),"dopti.pth")
                best_validation = g_losses_validation[-1]
        else:
            torch.save(model.module.state_dict(), "gan_gen.pth")
            torch.save(discriminator.module.state_dict(),"gan_disc.pth")
            torch.save(g_optimizer.state_dict(),"gopti.pth")
            torch.save(d_optimizer.state_dict(),"dopti.pth")

def PSNR_RMSE(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr, np.sqrt(mse)




if __name__ == "__main__":
    train()



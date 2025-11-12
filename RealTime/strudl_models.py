import torch.nn as nn
import torch
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)




class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

  
        self.up = nn.ConvTranspose3d(in_channels, in_channels//2, kernel_size=2, stride=2)
        # self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # # input is CDHW
        # diffZ = x2.size()[2] - x1.size()[2]
        # diffY = x2.size()[3] - x1.size()[3]
        # diffX = x2.size()[4] - x1.size()[4]
        
        # x1 = F.pad(x1, [diffZ // 2, diffZ - diffZ // 2,
        #                 diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CNN3D_Justin(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNN3D_Justin, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
     

        self.maxp = nn.MaxPool3d(2)
     # Encoder layers


        self.inc = DoubleConv(1,64)
        self.encoder1 = Down(64,128)
        self.encoder2 = Down(128,256)
        self.encoder3 = Down(256,512)

        # Decoder layers

        self.decoder_1 = Up(512,256)
        self.decoder_2 = Up(256,128)
        self.decoder_3 = Up(128,64)

        self.dropout = nn.Dropout3d(0.8)

        self.sigmoid = nn.Sigmoid()
        self.outconv = nn.Conv3d(64, output_channels, kernel_size=1)


    def forward(self, input_img):

        x1 = self.inc(input_img)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)

        
        # x4 = self.dropout(x4)
        
        
        # Decoder Stage - 4
        x_3d = self.decoder_1(x4,x3)
        x_2d = self.decoder_2(x_3d,x2)
        x_1d = self.decoder_3(x_2d,x1)
 

       
        return self.sigmoid(self.outconv(x_1d))
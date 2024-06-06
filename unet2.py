import torch
import torch.nn as nn
from ESRGAN import *

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.GroupNorm(1,output_dim),
            nn.ReLU(),

            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1,output_dim),            
        )
        self.rel = nn.ReLU()
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(1,output_dim),
        )
 
    def forward(self, x):
        return self.rel(self.conv_block(x) + self.conv_skip(x))


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class ResUnet(nn.Module):
    def __init__(self, channel, filters=[64, 128, 256, 512],full_size=1024):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.GroupNorm(1,filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )
        self.full_size = full_size

        self.residual_conv_1   = ResidualConv(filters[0], filters[0], 1, 1) #input_dim, output_dim, stride, padding
        self.residual_conv_1_2 = ResidualConv(filters[0], filters[1], 1, 1)#(128,256,256)

        self.residual_conv_2   = ResidualConv(filters[1], filters[1], 2, 1)
        self.residual_conv_2_2 = ResidualConv(filters[1], filters[2], 1, 1)#(256,128,128)

        self.residual_conv_3   = ResidualConv(filters[2], filters[2], 2, 1)
        self.residual_conv_3_2 = ResidualConv(filters[2], filters[3], 1, 1)#(512,64,64)

        self.residual_conv_4   = ResidualConv(filters[3], filters[3], 2, 1)
        self.residual_conv_4_2 = ResidualConv(filters[3], filters[3], 1, 1)#(512,32,32)

        self.bottleneck = nn.Sequential( nn.Conv2d(filters[3], filters[3], 1), nn.ReLU(),)#(512,32,32)


        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[3], filters[3], 1, 1)

        self.upsample_2 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[3], filters[2], 1, 1)

        self.upsample_3 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[2], filters[1], 1, 1)

        self.upsample_4 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv4 = ResidualConv(filters[1] + filters[1], filters[0], 1, 1)
        
        if(self.full_size==1024):
            scaling_factor=4
        else:
            scaling_factor = 2
        #  Upscaling is done by sub-pixel convolution, with each such block upscaling by a factor of 2
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        self.subpixel_convolutional_blocks = nn.Sequential(
        *[SubPixelConvolutionalBlock(kernel_size=3, n_channels=filters[0], scaling_factor=2) for i
            in range(n_subpixel_convolution_blocks)])


        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], channel, 1, 1),
            nn.Sigmoid(),
        )

    def scale_to_depth(self,x):
        n ,c , in_h, in_w = x.size()
        out_h, out_w = in_h //4, in_w//4

        x = x.reshape(n,c,256,4, 256,4)
        x = x.permute(0,3,5,1,2,4)
        x = x.reshape(n,4*4*c,256,256)
        return x


    def forward(self, x,x2):

        x = torch.cat([x,x2],1)
        # Encoder
        x1   = self.input_layer(x) + self.input_skip(x)
        # print("x1: ",x1.shape)
        
        x2   = self.residual_conv_1(x1)
        x2_2 = self.residual_conv_1_2(x2)
        # print("x2_2: ",x2_2.shape)

        x3   = self.residual_conv_2(x2_2)
        x3_2 = self.residual_conv_2_2(x3)
        # print("x3_2: ",x3_2.shape)

        x4   = self.residual_conv_3(x3_2)
        x4_2 = self.residual_conv_3_2(x4)
        # print("x4_2: ",x4_2.shape)

        x5   = self.residual_conv_4(x4_2)
        x5_2 = self.residual_conv_4_2(x5)
        # print("x5_2: ",x5_2.shape)




        # Bridge
        x6_2 = self.bottleneck(x5_2)
        # print("x6_2: ",x6_2.shape)




        # Decoder
        x5 = torch.cat([x6_2, x5_2], dim=1)
        x6 = self.up_residual_conv1(x5)## filters [3]
        # print("x6: ",x6.shape)



        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x4_2], dim=1) 
        x8 = self.up_residual_conv2(x7) ## filters [2]
        # print("x8: ",x8.shape)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x3_2], dim=1)
        x10 = self.up_residual_conv3(x9) ## filters [1]
        # print("x10: ",x10.shape)


        x10 = self.upsample_4(x10)
        x11 = torch.cat([x10, x2_2], dim=1)
        x11 = self.up_residual_conv4(x11) ## filters [0]
        # print("x11: ",x11.shape)

        x11 = self.subpixel_convolutional_blocks(x11)
       
        output = self.output_layer(x11)

        return output
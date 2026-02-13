import torch
import torch.nn as nn
import numpy as np


#this is the definiton of the Autoencoder block this could be used in both the encoder and the decoder 

class AutoEncoderBlock(nn.Module):
    def __init__(self, in_c, out_c,encode=False):

        super(AutoEncoderBlock, self).__init__()

        self.encode = encode
        self.in_channel = in_c
        self.out_channel = out_c
        # Padding keeps spatial sizes from collapsing too fast in the decoder.
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1)
        if encode:
            self.maxPooling = nn.MaxPool2d(kernel_size=2,stride=2 )
            self.up_conv = None
        else:
            self.maxPooling = None
            self.up_conv = nn.ConvTranspose2d(out_c, out_c,4,2,1)
    

        self.relu = nn.ReLU()
        self.bnorm = nn.BatchNorm2d(out_c)

    def forward(self,x):
        x_p = self.conv(x)
        x_p = self.relu(self.bnorm(x_p))
        if self.encode:
            x_p = self.maxPooling(x_p)
        else:
            x_p = self.up_conv(x_p)
        return x_p
    

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            AutoEncoderBlock(1, 32, encode=True),
            AutoEncoderBlock(32, 64, encode=True),
            AutoEncoderBlock(64, 128, encode=True)
        )   

        self.decoder = nn.Sequential(
            AutoEncoderBlock(encode=False,in_c=128,out_c=64),
            AutoEncoderBlock(encode=False,in_c=64,out_c=32),
            AutoEncoderBlock(encode=False,in_c=32,out_c=1),
        )
        # Upsample decoder output back to original spatial dimensions
        self.final_upsample = nn.Upsample(size=(128, 6000), mode='bilinear', align_corners=False)

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_upsample(x)
        return x
    
    def encode_latent(self,x):
        x = self.encoder(x)
        return x
    
    def decode_latent(self,x):
        x = self.decoder(x)
        return x
    


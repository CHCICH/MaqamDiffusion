import torch
import torch.nn as nn
import numpy as np


#this is the definiton of the Autoencoder block this could be used in both the encoder and the decder 

class AutoEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mode="encode"):
        super().__init__()

        assert mode in ["encode", "decode"]
        self.mode = mode

        if mode == "encode":
            self.op = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            )
        else:
            self.op = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.op(x)
        x = self.norm(x)
        x = self.act(x)
        return x
    

class AutoEncoder(nn.Module):
    def __init__(self,width=128, height=1024):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            AutoEncoderBlock(1, 32, mode="encode"),
            AutoEncoderBlock(32, 64, mode="encode"),
            AutoEncoderBlock(64, 128, mode="encode"),
            AutoEncoderBlock(128,256,mode="encode"),
            AutoEncoderBlock(256,128,mode="encode")
        )   
        
        self.decoder = nn.Sequential(
            AutoEncoderBlock(128,256,mode="decode"),
            AutoEncoderBlock(256,128,mode="decode"),
            AutoEncoderBlock(128, 64, mode="decode"),
            AutoEncoderBlock(64, 32, mode="decode"),
            AutoEncoderBlock(32, 1, mode="decode"),
        )

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode_latent(self,x):
        x = self.encoder(x)
        return x
    
    def decode_latent(self,x):
        x = self.decoder(x)
        return x
    

class Classifier(nn.Module):
    def __init__(self,input_size, output_size):
        super(Classifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.classify = nn.Sequential(
            nn.Linear(input_size,input_size//4),
            nn.ReLU(),
            nn.Linear(input_size//4, input_size//8),
            nn.ReLU(),
            nn.Linear(input_size//8,output_size*16),
            nn.ReLU(),
            nn.Linear(output_size*16, output_size)
        )
    
    def forward(self,x):
        return self.classify(x)

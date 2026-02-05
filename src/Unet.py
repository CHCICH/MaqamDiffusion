import torch
import torch.nn as nn
import numpy as np

class U_net_Block(nn.Module):
    def __init__(self,in_ch, out_ch, diffusion_time_emb_dim, up=False):
        super.__init__()
        self.time_mlp = nn.Linear(diffusion_time_emb_dim,out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch,out_ch,3,padding=1)
            self.transform = nn.ConvTranspose2d(out_ch,out_ch, 4,2,1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch,3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch,4,2,1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2] 
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class SinosoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        exponents = torch.arange(half_dim, dtype=torch.float32, device=device)
        inv_freq = torch.exp(-np.log(10000) * exponents / half_dim) 
        if x.dim() == 1:
             x = x.unsqueeze(-1)
        sinusoidal_arg = x * inv_freq
        emb = torch.cat([sinusoidal_arg.sin(), sinusoidal_arg.cos()], dim=-1)
        return emb


class DenoiserNetwork_Unet(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 1
        down_channels = (32,64, 128, 256, 512)
        up_channels = (512, 256, 128, 64,32)
        out_dim = 1
        diffusion_time_emd_dim = 32 

        self.time_mlp = nn.Sequential(
                SinosoidalPosEmb(diffusion_time_emd_dim),
                nn.Linear(diffusion_time_emd_dim, diffusion_time_emd_dim),
                nn.ReLU()
        )
        
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
        self.downs = nn.ModuleList([
            U_net_Block(down_channels[i], down_channels[i+1], diffusion_time_emd_dim)
            for i in range(len(down_channels) - 1)
        ])
        self.ups = nn.ModuleList([
            U_net_Block(up_channels[i], up_channels[i+1], diffusion_time_emd_dim, up=True)
            for i in range(len(up_channels) - 1)
        ])
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        residual_inputs = []
        for down in self.downs:
            x = down.forward(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up.forward(x, t)
        
        return self.output(x)
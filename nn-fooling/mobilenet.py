import torch
import torch.nn as nn
import torch.nn.functional as F

def h_swish(x):
    return x * F.relu6(x+3)/6

class MobileNetV3Block(nn.Module):
    def __init__(
        self, 
        in_channels, out_channels, kernel_size, stride=1, padding=0,
        se=True, 
        expansion=2, fcreduction=4, nl=nn.LeakyReLU(0.1, True)):
        """
        Require:
            - in_channels:
            - out_channels:
            - kernel_size:
            - se(bool): where to use se
        """
        
        super(self.__class__, self).__init__()
        begin_out_channels = in_channels*expansion
        
        self.beginconv = nn.Conv2d(
            in_channels, 
            begin_out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding
        )
        self.beginbn = nn.BatchNorm2d(begin_out_channels)

        self.nl = nl
        
        self.dwconv = nn.Conv2d(
            begin_out_channels,
            begin_out_channels,
            (3,3),
            stride=1,
            padding=1,
            groups=begin_out_channels
        )
        self.dwbn = nn.BatchNorm2d(begin_out_channels)

        if se:
            self.pfc1 = nn.Linear(begin_out_channels, begin_out_channels//fcreduction)
            self.pfc2 = nn.Linear(begin_out_channels//fcreduction, begin_out_channels)
        
        self.se = se
        
        self.endconv = nn.Conv2d(
            begin_out_channels,
            out_channels,
            (1,1)
        )

        if in_channels != out_channels or stride != 1:
            self.outadj = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        else:
            self.outadj = lambda x: x

    def forward(self, x: torch.Tensor):
        
        out = self.beginconv(x)
        out = self.beginbn(out)
        out = F.leaky_relu(out, 0.1, inplace=True)
        
        out = self.dwconv(out)
        out = self.dwbn(out)
        out = self.nl(out)
    
        if self.se:
            # N C H W
            meano = F.adaptive_avg_pool2d(out, 1)[:,:,0,0]
            meano = self.pfc1(meano)
            meano = F.relu(meano, inplace=True)
            meano = self.pfc2(meano)
            meano = h_swish(meano)
            
            out = out * meano[:,:,None,None]

        out = self.endconv(out)
        out = self.nl(out)
        
        return self.outadj(x) + out

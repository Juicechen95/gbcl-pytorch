import torch
import numpy as np
from torch.nn import Module, Conv2d, Parameter
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
import math
#from torch.nn.parameter import Parameter

class GCM(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, mode = 'tig', dilation=1, groups=1, bias=False):
        super(GCM, self).__init__()
        self.nInputPlane = in_channels
        self.nOutputPlane = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = _pair(dilation)
        self.groups = groups
        self.kW = kernel_size
        self.mode = mode
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        ## init weight
        weight = torch.zeros(self.nOutputPlane, self.nInputPlane, self.kW, self.kW)
        if self.mode == 'tig':
            print('in tig\n')
            n = 0
            for mm in list(range(int(self.nOutputPlane/4))):
                j = mm*4 + 1
                la = 2+(3/(self.nOutputPlane/4))*n
                ga = (1/(self.nOutputPlane/4))*n
                theta = (180/(self.nOutputPlane/4))*n
                print('j')
                print(j)
                print(theta)
                
                for i in list(range(self.nInputPlane)):
                    weight[j-1][i] = self.gabor2(self.kW, 3,((theta+i)*0.0174), 0, 1.68,0.5,1)
                    weight[j][i] = self.gabor2(self.kW, 3,((theta+i)*0.0174), 0, 1.68,0.5, 0)
                    weight[j+1][i] = self.gabor2(self.kW, 3,((theta+i)*0.0174), 0, 1.68,1, 1)
                    weight[j+2][i] = self.gabor2(self.kW, 3,((theta+i)*0.0174), 0, 1.68,1, 0)
                n += 1
        if self.mode == 't1':
            print('in t1\n')
            n = 0
            for mm in list(range(int(self.nOutputPlane/4))):
                j = mm*4 + 1
                la = 2+(3/(self.nOutputPlane/4))*n
                ga = (1/(self.nOutputPlane/4))*n
                theta = (180/(self.nOutputPlane/4))*n
                print('j')
                print(j)
                print(theta)
                
                for i in list(range(self.nInputPlane)):
                    weight[j-1][i] = self.gabor2(self.kW, 3,((theta)*0.0174), 0, 1.68,0.5,1)
                    weight[j][i] = self.gabor2(self.kW, 3,((theta)*0.0174), 0, 1.68,0.5, 0)
                    weight[j+1][i] = self.gabor2(self.kW, 3,((theta)*0.0174), 0, 1.68,1, 1)
                    weight[j+2][i] = self.gabor2(self.kW, 3,((theta)*0.0174), 0, 1.68,1, 0)
                n += 1
        elif self.mode == 'tig1':
            print('in tig1\n')
            n = 0
            for mm in list(range(int(self.nOutputPlane/4))):
                j = mm*4 + 1
                la = 2+(3/(self.nOutputPlane/4))*n
                ga = (1/(self.nOutputPlane/4))*n
                theta = (180/(self.nOutputPlane/4))*n
                print('j')
                print(j)
                print(theta)
                
                for i in list(range(self.nInputPlane)):
                    weight[j-1][i] = self.gabor2(self.kW, 3,((theta+i*3)*0.0174), 0, 1.68,0.5,1)
                    weight[j][i] = self.gabor2(self.kW, 3,((theta+i*3)*0.0174), 0, 1.68,0.5, 0)
                    weight[j+1][i] = self.gabor2(self.kW, 3,((theta+i*3)*0.0174), 0, 1.68,1, 1)
                    weight[j+2][i] = self.gabor2(self.kW, 3,((theta+i*3)*0.0174), 0, 1.68,1, 0)
                n += 1
        elif self.mode == 'tig2':
            print('in tig2\n')
            n = 0
            for mm in list(range(int(self.nOutputPlane/4))):
                j = mm*4 + 1
                la = 2+(3/(self.nOutputPlane/4))*n
                ga = (1/(self.nOutputPlane/4))*n
                theta = (180/(self.nOutputPlane/4))*n
                print('j')
                print(j)
                print(theta)
                
                for i in list(range(self.nInputPlane)):
                    weight[j-1][i] = self.gabor2(self.kW, 3+i,((theta+i)*0.0174), 0, 1.68,0.5,1)
                    weight[j][i] = self.gabor2(self.kW, 3+i,((theta+i)*0.0174), 0, 1.68,0.5, 0)
                    weight[j+1][i] = self.gabor2(self.kW, 3+i,((theta+i)*0.0174), 0, 1.68,1, 1)
                    weight[j+2][i] = self.gabor2(self.kW, 3+i,((theta+i)*0.0174), 0, 1.68,1, 0)
                n += 1
        elif self.mode == 'tig3':
            print('in tig3\n')
            n = 0
            for mm in list(range(int(self.nOutputPlane/4))):
                j = mm*4 + 1
                la = 2+(3/(self.nOutputPlane/4))*n
                ga = (1/(self.nOutputPlane/4))*n
                theta = (180/(self.nOutputPlane/4))*n
                print('j')
                print(j)
                print(theta)
                
                for i in list(range(self.nInputPlane)):
                    weight[j-1][i] = self.gabor2(self.kW, 5,((theta+i)*0.0174), 0, 1.68,0.5,1)
                    weight[j][i] = self.gabor2(self.kW, 5,((theta+i)*0.0174), 0, 1.68,0.5, 0)
                    weight[j+1][i] = self.gabor2(self.kW, 5,((theta+i)*0.0174), 0, 1.68,1, 1)
                    weight[j+2][i] = self.gabor2(self.kW, 5,((theta+i)*0.0174), 0, 1.68,1, 0)
                n += 1
        elif self.mode == 'tig4':
            print('in tig4\n')
            n = 0
            for mm in list(range(int(self.nOutputPlane/8))):
                j = mm*8 + 1
                la = 2+(3/(self.nOutputPlane/8))*n
                ga = (1/(self.nOutputPlane/8))*n
                theta = (180/(self.nOutputPlane/8))*n
                print('j')
                print(j)
                print(theta)
                
                for i in list(range(self.nInputPlane)):
                    weight[j-1][i] = self.gabor2(self.kW, 2+i,((theta+i*3)*0.0174), 0, 1.68,0.5,1)
                    weight[j][i] = self.gabor2(self.kW, 2+i,((theta+i*3)*0.0174), 0, 1.68,0.5, 0)
                    weight[j+1][i] = self.gabor2(self.kW, 2+i,((theta+i*3)*0.0174), 0, 1.68,1, 1)
                    weight[j+2][i] = self.gabor2(self.kW, 2+i,((theta+i*3)*0.0174), 0, 1.68,1, 0)
                    weight[j+3][i] = self.gabor2(self.kW, 3+i,((theta+i*3)*0.0174), 0, 1.68,1, 1)
                    weight[j+4][i] = self.gabor2(self.kW, 3+i,((theta+i*3)*0.0174), 0, 1.68,1, 0)
                    weight[j+5][i] = self.gabor2(self.kW, 3+i,((theta+i*3)*0.0174), 0, 1.68,1, 1)
                    weight[j+6][i] = self.gabor2(self.kW, 3+i,((theta+i*3)*0.0174), 0, 1.68,1, 0)
                n += 1
        elif self.mode == 'tig5':
            print('in tig3\n')
            n = 0
            for mm in list(range(int(self.nOutputPlane/4))):
                j = mm*4 + 1
                la = 2+(3/(self.nOutputPlane/4))*n
                ga = (1/(self.nOutputPlane/4))*n
                theta = (180/(self.nOutputPlane/4))*n
                print('j')
                print(j)
                print(theta)
                
                for i in list(range(self.nInputPlane)):
                    weight[j-1][i] = self.gabor2(self.kW, 3+i,((theta+i*3)*0.0174), 0, 1.68,0.5,1)
                    weight[j][i] = self.gabor2(self.kW, 3+i,((theta+i*3)*0.0174), 0, 1.68,0.5, 0)
                    weight[j+1][i] = self.gabor2(self.kW, 3+i,((theta+i*3)*0.0174), 0, 1.68,1, 1)
                    weight[j+2][i] = self.gabor2(self.kW, 3+i,((theta+i*3)*0.0174), 0, 1.68,1, 0)
                n += 1
        elif self.mode == 'tig6':
            print('in tig6\n')
            n = 0
            for mm in list(range(int(self.nOutputPlane/4))):
                j = mm*4 + 1
                la = 2+(3/(self.nOutputPlane/4))*n
                ga = (1/(self.nOutputPlane/4))*n
                theta = (180/(self.nOutputPlane/4))*n
                print('j')
                print(j)
                print(theta)
                
                for i in list(range(self.nInputPlane)):
                    weight[j-1][i] = self.gabor2(self.kW, 3+i,((theta)*0.0174), 0, 1.68,0.5,1)
                    weight[j][i] = self.gabor2(self.kW, 3+i,((theta)*0.0174), 0, 1.68,0.5, 0)
                    weight[j+1][i] = self.gabor2(self.kW, 4-i,((theta)*0.0174), 0, 1.68,1, 1)
                    weight[j+2][i] = self.gabor2(self.kW, 4-i,((theta)*0.0174), 0, 1.68,1, 0)
                n += 1
        elif self.mode == 'tig7':
            print('in tig7\n')
            n = 0
            for mm in list(range(int(self.nOutputPlane/8))):
                j = mm*8 + 1
                la = 2+(3/(self.nOutputPlane/8))*n
                ga = (1/(self.nOutputPlane/8))*n
                theta = (180/(self.nOutputPlane/8))*n
                print('j')
                print(j)
                print(theta)
                
                for i in list(range(self.nInputPlane)):
                    weight[j-1][i] = self.gabor2(self.kW, 2+i,((theta+i)*0.0174), 0, 1.68,0.5,1)
                    weight[j][i] = self.gabor2(self.kW, 2+i,((theta+i)*0.0174), 0, 1.68,0.5, 0)
                    weight[j+1][i] = self.gabor2(self.kW, 2+i,((theta+i)*0.0174), 0, 1.68,1, 1)
                    weight[j+2][i] = self.gabor2(self.kW, 2+i,((theta+i)*0.0174), 0, 1.68,1, 0)
                    weight[j+3][i] = self.gabor2(self.kW, 3+i,((theta+i)*0.0174), 0, 1.68,1, 1)
                    weight[j+4][i] = self.gabor2(self.kW, 3+i,((theta+i)*0.0174), 0, 1.68,1, 0)
                    weight[j+5][i] = self.gabor2(self.kW, 3+i,((theta+i)*0.0174), 0, 1.68,1, 1)
                    weight[j+6][i] = self.gabor2(self.kW, 3+i,((theta+i)*0.0174), 0, 1.68,1, 0)
                n += 1
        self.weight = Parameter(weight)
        self.weight.requires_grad=False

        
    def gabor2(self, Sx, lam, theta, shi, sigma,gamma, R):
        pi = 3.14
        Sy = Sx
        #print('Sx\n', Sx)
        sigma = 0.56*lam
        Gabor = torch.zeros(Sx,Sy)
        for x in list(range(Sx)):
            for y in list(range(Sy)):
                xPrime =  (x+1-Sx/2-1)*math.cos(theta) + (y+1-Sy/2-1)*math.sin(theta) #equation 1
                yPrime = -(x+1-Sx/2-1)*math.sin(theta)  + (y+1-Sy/2-1)*math.cos(theta) #equation 2
                #r=1 means real part of gabor
                if R == 1:
                    Gabor[x][y] = math.exp(-1/(sigma*3)*((xPrime*xPrime)+(yPrime*yPrime * gamma*gamma )))*math.cos(2*pi*xPrime/lam  + shi) #equation 3
                else:
                    Gabor[x][y] = math.exp(-1/(sigma*3)*((xPrime*xPrime)+(yPrime*yPrime * gamma*gamma )))*math.sin(2*pi*xPrime/lam  + shi) #equation 3
                    
        return Gabor
            
        
    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

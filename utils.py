import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import piio
import random


CHECKPOINTING = ('CHECKPOINTING' in os.environ and os.environ['CHECKPOINTING']) or False
REAL_FFT = False
EPS = 1e-8

Tensor = torch.FloatTensor
if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

if CHECKPOINTING:
    import torch.utils
    import torch.utils.checkpoint as ck
    # XXX: it appears that there is a bug in checkpoint.py of pytorch
    # can't set preserve_rng_state to False, unfortunatly
    # torch.utils.checkpoint.preserve_rng_state = False

    def checkpoint(fun, *args):
        x = Tensor([0])
        x.requires_grad_(True)
        def f(x, *args):
            return fun(*args)
        return ck.checkpoint(f, x, *args)
else:
    def checkpoint(fun, *args):
        return fun(*args)

def fcheckpoint(f):
    def g(*args):
        return checkpoint(f, *args)
    return g

def read_tensor(path, tobatch=True, cpu=False):
    v = piio.read(path)
    v = torch.FloatTensor(v, device='cpu')
    if not cpu and torch.cuda.is_available():
        v = v.cuda()
    v = v.permute((2,0,1))
    if tobatch:
        v = torch.stack([v], dim=0)
    return v

def write_tensor(path, tensor):
    tensor = tensor.permute((0, 2, 3, 1)).squeeze()
    piio.write(path, tensor.cpu().detach().numpy())

def csplit(t):
    assert(t.size()[-1] == 2)
    return t[...,0], t[...,1]

def ri2c(r, i):
    return torch.stack([r, i], dim=-1)

def r2c(r):
    return ri2c(r, torch.zeros(r.size(), device=r.device))

def conj(c):
    assert(c.size()[-1] == 2)
    real, imag = csplit(c)
    return ri2c(real, -imag)

def cabs(c):
    assert(c.size()[-1] == 2)
    return r2c(c.norm(p=2, dim=-1))

def fft(r):
    if REAL_FFT:
        return torch.rfft(r, 2, onesided=True)
    return torch.fft(r2c(r), 2)

def ifft(c):
    assert(c.size()[-1] == 2)
    if REAL_FFT:
        return torch.irfft(c, 2, onesided=True)
    return torch.ifft(c, 2)[...,0]

@fcheckpoint
def cmul(t1, t2):
    assert(t1.size()[-1] == 2)
    assert(t2.size()[-1] == 2)
    a, b = csplit(t1)
    c, d = csplit(t2)
    P1 = a * c
    P2 = b * d
    P3 = (a + b) * (c + d)
    return ri2c(P1 - P2, P3 - P2 - P1)

@fcheckpoint
def cdiv(t1, t2):
    assert(t1.size()[-1] == 2)
    assert(t2.size()[-1] == 2)
    real1, imag1 = csplit(t1)
    real2, imag2 = csplit(t2)
    den = real2*real2 + imag2*imag2 + EPS
    return torch.stack([(real1*real2 + imag1*imag2)/den, (imag1*real2 - real1*imag2)/den], dim=-1)

def psf2otf(k, size):
    shape = k.size()
    w = size[-2] - shape[-2]
    h = size[-1] - shape[-1]
    k = F.pad(k, (0, w, 0, h))
    k = torch.roll(k, (-shape[-2]//2+1, -shape[-1]//2+1), dims=(-2, -1))
    return fft(k)

def otf2psf(fk, size):
    k = ifft(fk)
    k = torch.roll(k, (size[-2]//2, size[-1]//2), dims=(-2, -1))
    k = k[...,0:size[-2],0:size[-1]]
    return k

def pad_circular(x, pad):
    x = torch.cat([x[...,-pad:,:], x, x[...,0:pad,:]], dim=-2)
    x = torch.cat([x[...,:,-pad:], x, x[...,:,0:pad]], dim=-1)
    return x

def edgetaper(x, ks):
    ks = ks.item()
    ks //= 2
    # build the hann window
    w = torch.hann_window(ks*2)
    wt = torch.stack([w], dim=1)
    wt = wt.repeat((1,x.size()[-1]))
    W = torch.ones(x.size())

    W[:,:,:,0:ks] *= w[0:ks]
    W[:,:,:,-ks:] *= w[-ks:]
    W[:,:,0:ks,:] *= wt[0:ks,:]
    W[:,:,-ks:,:] *= wt[-ks:,:]

    # blur x
    k = torch.ones((x.size()[0], 1, ks, ks))/(ks*ks)
    fk = psf2otf(k, x.size())
    fx = fft(x)
    fx = cmul(fx, fk)
    b = ifft(fx)

    # blend the two images
    return b * (1 - W) + x * W


# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def conv3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class Conv3x3pad(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(Conv3x3pad, self).__init__()
        self.pad = nn.ReflectionPad2d(padding=1)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=0, bias=False)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x

class Conv3x3circ(nn.Module):

    def __init__(self, in_planes, out_planes, bias=False):
        super(Conv3x3circ, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=0, bias=bias)

    def forward(self, x):
        x = pad_circular(x, 1)
        x = self.conv(x)
        return x

class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = Conv3x3pad(inplanes, planes)
        # self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.bn1 = nn.GroupNorm(16, 64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3pad(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.bn2 = nn.GroupNorm(16, 64)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Upsample2(nn.Module):

    def __init__(self):
        super(Upsample2, self).__init__()

    def forward(self, x, size):
        x = F.pad(x, (2, 2, 2, 2), mode='replicate')
        x = F.interpolate(x, mode='bilinear', scale_factor=2, align_corners=False)
        if x.size() != size:
            w = size[-2] - x.size()[-2]
            h = size[-1] - x.size()[-1]
            x = x[...,4:4+size[-2],4:4+size[-1]]
        return x


class Roll2d(nn.Module):

    def __init__(self, shifts):
        super(Roll2d, self).__init__()
        self.shifts = shifts

    def forward(self, x):
        return x.roll(self.shifts, dims=(-2, -1))


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


def MSE_shift_crop(S, C, p=2):
    if p == 2:
        L = nn.MSELoss()
    elif p == 1:
        L = nn.L1Loss()
    else:
        assert(False)
    L2 = nn.MSELoss()
    # TODO: cleanup
    def loss(a, b):
        loss = 0
        mse = 0
        batch_size = a.size()[0]
        for i in range(batch_size):
            aa = a[i,...]
            bb = b[i,...]
            bbb = bb[...,C:-C-1,C:-C-1]
            v = None
            v2 = None
            for dx in range(-S, S+1):
                for dy in range(-S, S+1):
                    aaa = aa.roll((dx, dy), dims=(-2, -1))
                    aaa = aaa[...,C:-C-1,C:-C-1]
                    if v is None:
                        v = L(aaa, bbb)
                        v2 = L(aaa, bbb)
                    else:
                        v = torch.min(L(aaa, bbb), v)
                        v2 = torch.min(L2(aaa, bbb), v2)
            loss += v
            mse += v2
        return loss / batch_size, mse / batch_size
    return loss

def mse_to_psnr(mse):
    import math
    return 10 * math.log10(1 / mse)



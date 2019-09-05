#!/usr/bin/env python3
# coding: utf-8

import os
import math
import numpy as np
import numpy.random
from PIL import Image
import scipy.ndimage
import scipy.signal

def center_kernel(x):
    dx = np.sum(range(1, x.shape[1]+1) * np.sum(x, axis=1))
    dy = np.sum(range(1, x.shape[0]+1) * np.sum(x, axis=0))
    dx = round(x.shape[1]//2+1 - dx)
    dy = round(x.shape[0]//2+1 - dy)
    y = scipy.ndimage.shift(x, [dx, dy], order=0)
    return y

def generate_kernel(s):
    s2 = s * 8
    k = np.zeros((s2, s2)).astype('float32')

    import random

    dx, dy = 0, 0
    x = s2 // 2
    y = s2 // 2
    v = 1
    dt = 0.06
    m = numpy.random.normal() + 4

    for j in range(100):
        dx = min(m, max(-m, dx + numpy.random.normal()*2))
        dy = min(m, max(-m, dy + numpy.random.normal()*2))
        x += dx * dt
        y += dy * dt
        v = max(1e-4, v + numpy.random.normal())

        xL = math.floor(x)
        xH = xL + 1
        yL = math.floor(y)
        yH = yL + 1

        ax = x-xL
        ay = y-yL
        bx = xH-x
        by = yH-y

        if xL >= 0 and yL >= 0 and xH < s2 and yH < s2:
            k[xL,yL] += bx*by*dt
            k[xL,yH] += bx*ay*dt
            k[xH,yL] += ax*by*dt
            k[xH,yH] += ax*ay*dt
        else:
            break

    k = k[(s2-s)//2:(s2+s)//2, (s2-s)//2:(s2+s)//2]
    if np.sum(k) > 0:
        k = k / np.sum(k)
    k = center_kernel(k)
    return k

if __name__ == '__main__':
    import argparse
    from libtiff import TIFF

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str)
    parser.add_argument("-n", type=int, default='10000')
    parser.add_argument("-s", type=int, default=31)
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    import progressbar
    bar = progressbar.ProgressBar()
    for i in bar(range(args.n)):
        tif = TIFF.open('{}/{}.tiff'.format(args.out, i), mode='w')
        tif.write_image(generate_kernel(args.s).astype('float32'))


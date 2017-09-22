import mrcfile
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
#you need cp /usr/lib/python2.7/dist-packages/cv2.x86_64-linux-gnu.so to
# ~/venv/venv1/lib/python2.7/site-packages/
import cv2

#trackpy
import pandas as pd
from pandas import DataFrame,Series
import pims
import trackpy as tp
#foundconter

#%matplotlib inline

#cmap = matplotlib.cm.gray_r
mpl.rc('image',cmap='gray')
mpl.rc('figure',figsize=(10,6))
def postprocess(n):
    mrc=mrcfile.mmap('/home/qjq/data/A02-FixMask-8tp-192x160-down128.mrcs.mrcs',mode="r")
    x=mrc.data[n]
#    plt.imshow(x,cmap=cmap)
    plt.imshow(x)
    g_b=cv2.GaussianBlur(x,(43,43),7)
    nm = cv2.normalize(g_b,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)
    ret, thresh = cv2.threshold(nm, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(nm, contours, -1, (0,255,0), 1)
    plt.imshow(nm)
    mrc.close()
def preprocess(n):
    mrc=mrcfile.mmap('/home/qjq/data/A02-FixMask-8tp-192x160-down128.mrcs.mrcs',mode="r")
    x=mrc.data[n]
    plt.imshow(x)
    mrc.close()
def gauss_norm(n):
    mrc=mrcfile.mmap('/home/qjq/data/A02-FixMask-8tp-192x160-down128.mrcs.mrcs',mode="r")
    x=mrc.data[n]
    g_b=cv2.GaussianBlur(x,(43,43),7)
    nm = cv2.normalize(g_b,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)
    plt.imshow(nm)
    mrc.close()
#def main():
#    test()

#if __name__ == '__main__':
#    main()

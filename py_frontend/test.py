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
def mynorm(mrc_x):
    mrc_y=np.zeros(mrc_x.shape,mrc_x.dtype)
    mrc_x_mean=mrc_x.mean()
    mrc_x_stdvar=np.sqrt(mrc_x.var())
    mrc_y=(mrc_x-mrc_x_mean)/mrc_x_stdvar
    #remove >3 and <-3
    np.where(mrc_y>3.0,3.0,mrc_y)
    np.where(mrc_y<-3.0,-3.0,mrc_y)
    #plt.imshow(mrc_y)
    return mrc_y

def postprocess(n):
    mrc=mrcfile.mmap('/home/qjq/data/A02-FixMask-8tp-192x160-down128.mrcs.mrcs',mode="r")
    x_origin=mrc.data[n]
#    plt.imshow(x,cmap=cmap)
#    plt.imshow(x)
    x=mynorm(x_origin)
    g_b=cv2.GaussianBlur(x,(43,43),7)
    nm = cv2.normalize(g_b,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)
    ret, thresh = cv2.threshold(nm, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    nm_copy = nm.copy()
    cv2.drawContours(nm, contours, -1, (0,255,0), 1)
    plt.imshow(nm)
    mrc.close()

def adaptive(n):
    mrc=mrcfile.mmap('/home/qjq/data/A02-FixMask-8tp-192x160-down128.mrcs.mrcs',mode="r")
    x_origin=mrc.data[n]
#    plt.imshow(x,cmap=cmap)
#    plt.imshow(x)
    x=mynorm(x_origin)
    g_b=cv2.GaussianBlur(x,(43,43),7)
    nm = cv2.normalize(g_b,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)
#    ret, thresh = cv2.threshold(nm, 127, 255, 0)
    thresh = cv2.adaptiveThreshold(nm,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
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

def cnt_area(contours):
    len_Contours = len(contours)
    list_Moment=[]
    list_Area=[]
    list_Centroid=[]
    list_CenterBias=[]
    projection_center=(64.0,64.0)
    for item in contours:
        list_Moment.append(cv2.moments(item))
    for item in contours:
        list_Area.append(cv2.contourArea(item))
    for item in list_Moment:
        cx = int (item['m10']/item['m00'])
        cy = int (item['m01']/item['m00'])
        list_Centroid.append((cx,cy))
    for item in list_Centroid:
        dist = np.sqrt(np.power(item[0]-projection_center[0],2)+np.power(item[1]-projection_center[1],2))
        list_CenterBias.append(dist)

def check_convex_particle(contours):
    #check convex
    convex_if=[]
    for item in contours:
        convex_if.append(cv2.isContourConvex(item))
    return convex_if
def convex_con
#convex,the best particle
class MrcProjection:
    def __init__ (self,mrc_array)
#
#def main():
#    test()

#if __name__ == '__main__':
#    main()

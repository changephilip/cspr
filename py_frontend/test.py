import mrcfile
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import pickle
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
    cv2.drawContours(nm_copy, contours, -1, (0,255,0), 1)
    plt.imshow(nm_copy)
    mrc.close()
    return contours

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
    return contours

def ad_mean_c(n):
    mrc=mrcfile.mmap('/home/qjq/data/A02-FixMask-8tp-192x160-down128.mrcs.mrcs',mode="r")
    AngstromPerPixel=1.3
    x_origin=mrc.data[n]
    x_fft=hpf_lpf_fft(x_origin,x_origin.shape[0],AngstromPerPixel)
    x_real = x_fft [:,:,0]
    x=mynorm(x_real)
    nm = cv2.normalize(x,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)
    ret, thresh = cv2.threshold(nm, 127, 255, 0)
    thresh = cv2.adaptiveThreshold(nm, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    nm_copy = nm.copy()
    cv2.drawContours(nm, contours, -1, (0,255,0), 1)
    plt.imshow(nm)
    mrc.close()
    return contours

def ad_gaussian_c(n):
    mrc=mrcfile.mmap('/home/qjq/data/A02-FixMask-8tp-192x160-down128.mrcs.mrcs',mode="r")
    AngstromPerPixel=1.3
    x_origin=mrc.data[n]
    x_fft=hpf_lpf_fft(x_origin,x_origin.shape[0],AngstromPerPixel)
    x_real = x_fft [:,:,0]
    x=mynorm(x_real)
    nm = cv2.normalize(x,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)
    ret, thresh = cv2.threshold(nm, 127, 255, 0)
    thresh = cv2.adaptiveThreshold(nm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    nm_copy = nm.copy()
    cv2.drawContours(nm, contours, -1, (0,255,0), 1)
    plt.imshow(nm)
    mrc.close()
    return contours

def preprocess(n):
    mrc=mrcfile.mmap('/home/qjq/data/A02-FixMask-8tp-192x160-down128.mrcs.mrcs',mode="r")
    x=mrc.data[n]
    plt.imshow(x)
    mrc.close()
    return x

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
    #get area information list

def check_convex_particle(contours):
    #check convex,not needed
    convex_if=[]
    for item in contours:
        convex_if.append(cv2.isContourConvex(item))
    return convex_if

#high pass and low pass filter in DFT
def hpf_lpf_fft(particle,particle_size,AngstromPerPixel):
    dft = cv2.dft(np.float32(particle),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    mask_backgroud = np.ones((particle_size,particle_size),np.uint8)
    x,y = np.ogrid[0:particle_size,0:particle_size]
    hpf = 30
    lpf = 200
    hpf_size = int (AngstromPerPixel*particle_size/hpf)
    lpf_size = int (AngstromPerPixel*particle_size/lpf)
    #outside will be False
    mask_hpf = (x-(particle_size+1)/2)**2 + (y-(particle_size+1)/2)**2 <= hpf_size**2
    #inside will be False
    mask_lpf = (x-(particle_size+1)/2)**2 + (y-(particle_size+1)/2)**2 >= lpf_size**2
    mask_hyper = mask_hpf * mask_lpf
    mask_backgroud = np.multiply(mask_backgroud , mask_hyper )
    dft_shift[:,:,0] = np.multiply(dft_shift[:,:,0] , mask_backgroud)
    dft_shift[:,:,1] = np.multiply(dft_shift[:,:,1] , mask_backgroud)
    particle_back_ishift = np.fft.ifftshift(dft_shift)
    particle_back = cv2.idft(particle_back_ishift)

    return particle_back

def new_post(n):
    mrc=mrcfile.mmap('/home/qjq/data/A02-FixMask-8tp-192x160-down128.mrcs.mrcs',mode="r")
    AngstromPerPixel=1.3
    x_origin=mrc.data[n]
    x_fft=hpf_lpf_fft(x_origin,x_origin.shape[0],AngstromPerPixel)
    x_real = x_fft [:,:,0]
    x=mynorm(x_real)
    nm = cv2.normalize(x,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)
    ret, thresh = cv2.threshold(nm, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    nm_copy = nm.copy()
    cv2.drawContours(nm, contours, -1, (0,255,0), 1)
    plt.imshow(nm)
    mrc.close()
    return contours

def nnew_post(x_origin):
    AngstromPerPixel = 1.3
    x_fft = hpf_lpf_fft(x_origin, x_origin.shape[0], AngstromPerPixel)
    x_real = x_fft[:,:,0]
    x = mynorm(x_real)
    nm = cv2.normalize(x, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC1)
    ret , thresh = cv2.threshold(nm, 127, 255, 0)
    contours, hierarchy = cv2.findContours( thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    List_area  = cnt_area_test( contours)
    max_area = max(List_area)
    max_area_index = List_area.index(max(List_area))
    #max_area_bias = List_Center[max_area_index]
    return max_area, max_area_index
#adjust cv2.threshold to get the best area size
#find the largest two or three particles in 128x128 image
#if the area is over the threshold we defined before,then we modify cv2.threshold a little higher
#if the area is too small,should we need to lower the threshold for increasing the area?

def step_post(n):
    mrc = mrcfile.mmap('/home/qjq/data/A02-FixMask-8tp-192x160-down128.mrcs.mrcs',mode = 'r')
    AngstromPerPixel = 1.3
    x_orgin = mrc.data[n]
    x_fft = hpl_lpf_fft(x_origin, x_origin.shape[0],AngstromPerPixel)
    x_real = x_fft[:,:,0]
    x = mynorm(x_real)
    nm = cv2.normalize(x, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    

def step_contours(nm_pic):
    step=5
    start_step=127
    ret , thresh = cv2.threshold(nm_pic, start_step,255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max(cnt_area_test( contours))
    #if max_area > nm_pic:

def ero_dial(n):
    mrc = mrcfile.mmap("/home/qjq/data/A02-FixMask-8tp-192x160-down128.mrcs.mrcs",mode ='r')


def writeout(x_t,low_area,high_area,writeoutfilename):
    filewriteout = open(writeoutfilename,"w+")
    count_xt=0
    index_xt=0
    list_ret=[]
    for item in x_t:
        if item>low_area and item < high_area:
            list_ret.append(index_xt)
        index_xt = index_xt+1
    for item in list_ret:
        filewriteout.write(str(item))
        filewriteout.write("\n")
    filewriteout.close()

def cnt_area_test(contours):
    len_Contours = len(contours)
    list_Moment=[]
    list_Area = []
    list_Centroid = []
    list_CenterBias=[]
    projection_center=(64.0,64.0)
    for item in contours:
        list_Moment.append(cv2.moments(item))
    for item in contours:
        list_Area.append(cv2.contourArea(item))
    #for item in list_Moment:
     #   cx = int (item['m10']/item['m00'])
     #   cy = int (item['m01']/item['m00'])
     #   list_Centroid.append((cx,cy))
    #for item in list_Centroid:
     #   dist = np.sqrt(np.power(item[0]-projection_center[0],2)+np.power(item[1]-projection_center[1],2))
     #   list_CenterBias.append(dist)
    return list_Area
def read_from_pickle(picklefilename):
    pkl_file = open(picklefilename,'rb')
    data1 = pickle.load(pkl_file)
    x_t = np.array(data1).reshape(-1,1)
    pkl_file.close()
    return x_t

def dbscan_particle(x_t, eps, min_samples):
    db = DBSCAN(eps = 100, min_samples = 500)
    return db

def hist_fig(x_t):
    fig,(ax0,ax1) = plt.subplots(nrows = 2,figsize=(16,9))
    ax0.hist(x_t,100,40,normed =1,histtype='bar',facecolor='yellowgreen',alpha=0.75)
    ax0.set_title('pdf')
    ax1.hist(x_t,20,normed = 1,histtype='bar',facecolor='pink',alpha=0.75,cumulative=True,rwidth=0.8)
    ax1.set_title("cdf")
    fig.subplots_adjust(hspace=0.4)
    plt.show()
def test_main():
    mrc = mrcfile.mmap('/home/qjq/data/A02-FixMask-8tp-192x160-down128.mrcs.mrcs' , mode = "r")
    NumOfParticle = len(mrc.data)
    GlobalListArea=[]
    GlobalListAreaIndex=[]
    #GlobalListBias=[]
    for n in range(0,NumOfParticle):
        area, area_index  = nnew_post(mrc.data[n])
        GlobalListArea.append(area)
        GlobalListAreaIndex.append(area_index)
    return GlobalListArea,GlobalListAreaIndex
def return_pic(n):
    mrc = mrcfile.mmap('/home/qjq/data/A02-FixMask-8tp-192x160-down128.mrcs.mrcs',mode = 'r')
    AngstromPerPixel = 1.3
    x_origin = mrc.data[n]
    x_fft = hpf_lpf_fft(x_origin, x_origin.shape[0],AngstromPerPixel)
    x_real = x_fft[:,:,0]
    x = mynorm(x_real)
    nm = cv2.normalize(x, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return nm
def nothing(x):
    pass
def drawc(n):
    mrc = mrcfile.mmap('/home/qjq/data/A02-FixMask-8tp-192x160-down128.mrcs.mrcs',mode = 'r')
    AngstromPerPixel = 1.3

    x_origin = mrc.data[n]
    x_fft = hpf_lpf_fft(x_origin, x_origin.shape[0],AngstromPerPixel)
    x_real = x_fft[:,:,0]
    x = mynorm(x_real)
    nm = cv2.normalize(x, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    cv2.namedWindow('test',cv2.WINDOW_NORMAL)
    cv2.createTrackbar('th','test',0,255,nothing)
   # nm_gray=cv2.cvtColor(nm,cv2.COLOR_BGR2GRAY)
    while True:
        out_image = nm.copy()
        th_val=cv2.getTrackbarPos('th','test')
        ret , thresh = cv2.threshold(nm,th_val,255,cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out_image,contours,-1,(0,255,0),1)
        cv2.imshow("test",out_image)
        k=cv2.waitKey(1) & 0xFF
        if k==27:
            break
    cv2.destroyAllWindows()
#def cnt_area_wrapper(particle_real):
#convex,the best particle
#class MrcProjection:
#    def __init__ (self,mrc_array)
#
#def main():
#    test()

#if __name__ == '__main__':
#    main()

import numpy as np
import matplotlib.pyplot as plt
import cv2
from load_data import *
from transformation import *
from pr2_utils import *
from map_utils import *
from motion_utils import *
from particle import *

# dis_fn = 'disparity20_1.png'
# rgb_fn = 'rgb20_1.png'
# disp_path = "../data/dataRGBD/Disparity20/" + dis_fn
# rgb_path = "../data/dataRGBD/RGB20/" + rgb_fn

def normalize(img):
   max_ = img.max()
   min_ = img.min()
   return (img - min_)/(max_-min_)

def getRGB(disp_path, rgb_path):
   # load RGBD image
   imd = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED) # (480 x 640)
   imc = cv2.imread(rgb_path)[...,::-1] # (480 x 640 x 3)

   # convert from disparity from uint16 to double
   disparity = imd.astype(np.float32)

   # get depth
   dd = (-0.00304 * disparity + 3.31)
   z = 1.03 / dd

   # calculate u and v coordinates 
   v,u = np.mgrid[0:disparity.shape[0],0:disparity.shape[1]]
   #u,v = np.meshgrid(np.arange(disparity.shape[1]),np.arange(disparity.shape[0]))

   # get 3D coordinates 
   fx = 585.05108211
   fy = 585.05108211
   cx = 315.83800193
   cy = 242.94140713
   x = (u-cx) / fx * z
   y = (v-cy) / fy * z

   # calculate the location of each pixel in the RGB image
   rgbu = np.round((u * 526.37 + dd*(-4.5*1750.46) + 19276.0)/fx)
   rgbv = np.round((v * 526.37 + 16662.0)/fy)
   valid = (rgbu>= 0)&(rgbu < disparity.shape[1])&(rgbv>=0)&(rgbv<disparity.shape[0])

   # coordinates, rgb color
   return z[valid],x[valid],y[valid], imc[rgbv[valid].astype(int),rgbu[valid].astype(int)]/255.0


def getOpital(x,y,z):
    a = np.vstack((x,y))
    a = np.vstack((a, np.ones(x.shape[0])))
    # rgbi, rgbj, 1
    b = np.vstack((z,z))
    b = np.vstack((b,z))
    c = a * b

    K = np.array([[585.05, 0, 242.94],
              [0, 585.05, 315.84],
              [0, 0, 1]])
    K_inv = np.linalg.inv(K)
    return np.matmul(K_inv, c)

def opticalToWorldCoordinate(optical_point, pose):
    # create points transformation matrix
    T_opt = np.vstack([optical_point, np.ones((optical_point.shape[1]))])

        # use transform tree
    T = Transform(**pose)

        # transform laser points to world frame
    optical_point_w = T.chain('wTb', 'bTc', 'cTo') @ T_opt

        # get the x-y-z coordinates
    optical_point_w = optical_point_w[:3,:].T

    return optical_point_w[:,:2]
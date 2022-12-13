import cv2
import numpy as np

from skimage.metrics import structural_similarity as skt_ssim 
import re


def read_image(path: str):
    return cv2.imread(path)
    

##### sum of squared differences (SSD)
def ssd(img1, img2):
    return np.sum((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32))**2)

##### structural similarity index measure (SSIM)
def ssim(img1, img2) -> float:
    return skt_ssim(img1, img2, channel_axis=2)
    
#####  Normalized Cross Corelation (NCC)
def ncc(img1, img2) -> float:
    return cv2.matchTemplate(img1, img2, cv2.TM_CCORR_NORMED)
    

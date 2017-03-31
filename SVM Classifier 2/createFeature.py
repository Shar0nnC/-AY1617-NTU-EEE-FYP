import numpy as np
from color16 import *
import cv2
import scipy.stats as s
from matplotlib import pyplot as plt

def createFeature(image):

  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  featureVector = gray_image.flatten()

  featureVector = featureVector.reshape(1,-1)


  return(featureVector)

def createFeature4(image):

  red = image[:,:,2]
  green = image[:,:,1]
  blue = image[:,:,0]

  red = red.astype(float)
  green = green.astype(float)
  blue = blue.astype(float)
  
  red_st = red.flatten()
  green_st = green.flatten()
  blue_st = blue.flatten()
  
  green_blue = np.mean(green) - np.mean(blue)
  green_blue = green_blue.astype(float)
  green_blue_st = green_blue.flatten()

  edges = cv2.Canny(image,0,240)
  edges_st = edges.flatten()

  featureVector1 = np.array([np.std(red_st),s.skew(red_st),np.std(green_st),np.std(blue_st),s.skew(blue_st),green_blue_st])

  featureVector2 = edges_st

  featureVector = np.concatenate([featureVector1, featureVector2])

  featureVector = featureVector.reshape(1,-1)

  return(featureVector)

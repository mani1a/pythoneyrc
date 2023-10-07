import cv2 as cv
import sys 
import numpy

image = cv.imread('led.jpg')
cv.imshow('image',image)

def rescaleframe(frame, scale = 0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0]* scale)
    
    dimension = (width,height)
    
    return cv.resize(frame, dimension, interpolation= cv.INTER_AREA)

resized_img = rescaleframe(image)
    

cv.imshow('LED',resized_img)

cv.waitKey(5000)

cv.destroyAllWindows()
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2, time
from jetbot import Camera
from jetbot import bgr8_to_jpeg
import ipywidgets.widgets as widgets
import traitlets


# In[2]:


def preprocessing(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (7,7),0)
    return gray

def thresholding(img_gray):
    _, img_th = cv2.threshold(img_gray,np.average(img_gray)-40,255,cv2.THRESH_BINARY)
    img_th2 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,15)
    img_th3 = np.bitwise_and(img_th, img_th2)
    img_th4 = cv2.subtract(img_th2, img_th3)
    for i in range(5):
        img_th4 = cv2.medianBlur(img_th4, 5)
    return img_th4

def mask_roi(img_th, roi):
    mask = np.zeros_like(img_th)
    cv2.fillPoly(mask, np.array([roi], np.int32), 255)
    masked_image = cv2.bitwise_and(img_th, mask)
    return masked_image

def drawContours(img_rgb, contours):
    for cnt in contours:
        area = cv2.contourArea(cnt)
        cv2.drawContours(img_rgb, [cnt], 0, (255,0,0), 1)
    return img_rgb

def approximationContour(img, contours, e=0.02):
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        epsilon = e*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(img, [approx], 0, (0,255,255), 2)
    return img

def rectwithname(img, contours, e=0.02):
    result = img.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        epsilon = e*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.rectangle(result,(x,y),(x+w,y+h),(255,0,255),2)
    return result

def find_midptr(contours):
    center_ptrs = []
    e=0.01
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        center_ptr = [y, x + 0.5*w,]
        center_ptrs.append(center_ptr)
    center_ptrs = np.array(center_ptrs)
    return center_ptrs

def find_midlane(center_ptrs, center_image_point):
    L2_norm = np.linalg.norm((center_ptrs - center_image_point), axis=1, ord=2)
    loc = np.where(L2_norm==L2_norm.min())[0][0]
    midlane = center_ptrs[loc]
    return midlane

def find_degree(center_image_point, midlane):
    return 57.2958*np.arctan((midlane[1] - center_image_point[1])/(center_image_point[0] - midlane[0]))


# In[3]:


width = 224
height = 224
camera = Camera.instance()
input_image = widgets.Image(format='jpeg', width=width, height=height)
result1 = widgets.Image(format='jpeg', width=width, height=height)
result2 = widgets.Image(format='jpeg', width=width, height=height)
result3 = widgets.Image(format='jpeg', width=width, height=height)
result4 = widgets.Image(format='jpeg', width=width, height=height)
image_box = widgets.HBox([input_image, result1, result2, result3, result4], layout=widgets.Layout(align_self='center'))
display(image_box)
# display(result)


# In[4]:


count = 0
while True:
    img = camera.value
    img_gray = preprocessing(img)
    img_th = thresholding(img_gray)
    roi = [(0, height),(0, height/2-30), (width, height/2-30),(width, height),]
    img_roi = mask_roi(img_th, roi)
    
    kernel = np.ones((5,3),np.uint8)
    img_cl = cv2.morphologyEx(img_roi,cv2.MORPH_CLOSE, np.ones((5,5),np.uint8),iterations=4)
    img_op = cv2.morphologyEx(img_cl,cv2.MORPH_OPEN, np.ones((5,5),np.uint8),iterations=3)
    
    cannyed_image = cv2.Canny(img_op, 300, 500)
    contours, _ = cv2.findContours(cannyed_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    img_approx = approximationContour(img, contours, e=0.02)
    img_approx_rect = rectwithname(img, contours, e=0.01)  
    
    center_ptrs = find_midptr(contours)
    
    center_image_point = [height-1, width/2-1]
        
    midlane = find_midlane(center_ptrs, center_image_point)
    seta = find_degree(center_image_point, midlane)
    
    cv2.line(img,(int(center_image_point[1]), int(center_image_point[0])),(int(midlane[1]),int(midlane[0])),(0,0,255),3)
    cv2.putText(img, f'{seta}', (int(midlane[1]), int(midlane[0])-5), cv2.FONT_HERSHEY_COMPLEX, 0.5,(255, 0, 0), 1)
    
    result_img1 = img_th
    result_img2 = img_cl
    result_img3 = img_op
    result_img4 = img
    
    #show results
    result_imgs = [result_img1, result_img2, result_img3, result_img4]
    result_values = [result1, result2, result3, result4]
    for result_img, result_value in zip(result_imgs, result_values):
#         if len(result_img.shape)==2:
#             result_img = np.stack((result_img,)*3,2)
        result_value.value = bgr8_to_jpeg(result_img)
    input_image.value = bgr8_to_jpeg(img_gray)
    
    if count ==1000:
        break
    else:
        count = count +1
#         print(count, end='  ')
        time.sleep(0.1)


# In[5]:


def search_road(img):
    img_gray = preprocessing(img)
    img_th = thresholding(img_gray)
    roi = [(0, height),(0, height/2-30), (width, height/2-30),(width, height),]
    img_roi = mask_roi(img_th, roi)
    
    kernel = np.ones((5,3),np.uint8)
    img_cl = cv2.morphologyEx(img_roi,cv2.MORPH_CLOSE, np.ones((5,5),np.uint8),iterations=4)
    img_op = cv2.morphologyEx(img_cl,cv2.MORPH_OPEN, np.ones((5,5),np.uint8),iterations=3)
    
    cannyed_image = cv2.Canny(img_op, 300, 500)
    contours, _ = cv2.findContours(cannyed_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    center_ptrs = find_midptr(contours)

    center_image_point = [height-1, width/2-1]
    midlane = find_midlane(center_ptrs, center_image_point)
    seta = find_degree(center_image_point, midlane)
    
    cv2.line(img,(int(center_image_point[1]), int(center_image_point[0])),(int(midlane[1]),int(midlane[0])),(0,0,255),3)
    cv2.putText(img, f'{seta}', (int(midlane[1]), int(midlane[0])-5), cv2.FONT_HERSHEY_COMPLEX, 0.5,(255, 0, 0), 1)
    return img, seta


# In[6]:


count = 0
while True:
    img = camera.value
    img_result, seta = search_road(img)
    
    input_image.value = bgr8_to_jpeg(img_gray)
    result1.value = bgr8_to_jpeg(img_result)
    if count ==20:
        break
    else:
        count = count +1
#         print(count, end='  ')
        time.sleep(0.1)


# In[ ]:





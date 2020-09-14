# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:17:02 2019

@author: Lapcom Store
"""

#import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import image
from PIL import Image
import cv2
import numpy as np
import os
from os import walk
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skimage import color
from skimage import io
from skimage.transform import resize


letter_dict = { }
predicted_text=""
classifier = DecisionTreeClassifier()

for key in range(1,27):
    letter_dict[key] = chr(key+64)
    
#-------------------------
def prepare_model():
    train_dataset = pd.read_csv('D:\My Projects\AI Project\dataset\emnist/emnist-letters-train.csv')

    X = train_dataset.iloc[:, 1:].values       #.values : get a numpy array from dataset
    Y = train_dataset.iloc[:, 0].values        #.iloc : selecting rows & columns from dataset 
    classifier.fit(X, Y)                   # builds decisionTreeClassifier from training sets X,Y

def test_model():
    test_dataset = pd.read_csv('D:\My Projects\AI Project\dataset\emnist/emnist-letters-test.csv')
    X_test =  test_dataset.iloc[:, 1:].values 
    Y_test =  test_dataset.iloc[:, 0].values 
    for i in range(1,26):
        img2 = X_test[i]
        img2 = img2.reshape(1,784)
        y_pred = classifier.predict(img2) 
       # print(type(X_test[i]))
        img2.shape= (28,28)
        plt.imshow(255-img2, cmap='gray')
        plt.show()
       # y_pred = classifier.predict(img4)    #predict classfor X
        print(print_text(y_pred[0]))         #print(print_text(y_pred[9]))
        
    y_pred = classifier.predict(X_test)
    accuracy_score(y_pred, Y_test)    

def prepare_img_for_predection(): 
    for dirpath, dirnames, filenames in walk("croped"):
       image = color.rgb2gray(io.imread("croped/"+filenames))     
       #plt.imshow(255-img4, cmap='gray')
       # plt.show()
       image = resize(image, (28, 28),anti_aliasing=True)
       image = np.asarray(image)
       #print(img4.shape)
       image = image.reshape(1,784)
       #print(img4.shape)
       predict_new_img(image)
   
def predict_new_img(img):
    y_pred = classifier.predict(img) 
    predicted_text += print_text(y_pred)
    
def print_text(key):
     return letter_dict[key]
    # code to return text in gui
    
def delete_all():
    list1 = list()
    for dirpath, dirnames, filenames in walk("croped"):
         list1 = filenames
    for bath in list1:
        os.remove("croped/" + bath)
    
def croped(path):
    delete_all()
    image = cv2.imread(path)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
    _,thresh = cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV) 
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilated = cv2.dilate(thresh,kernel,iterations = 0) 
    _,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
    i=5
    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)
        cv2.imwrite("croped/"+"ss"+str(i)+".jpg",image[y:y+h,x:x+w])
        i=i+1
        
def delete_all():
    list1 = list()
    for dirpath, dirnames, filenames in walk("croped"):
         list1 = filenames

    for bath in list1:
        os.remove("croped/" + bath)
        





















#------------------------------------------
def print_text(key):
       return letter_dict[key]
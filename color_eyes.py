import numpy as np
import cv2 as cv
import pandas as pd
from tensorflow.keras.models import load_model

LEFT_X1 = (37 * 2)  -1
LEFT_X2 = (40 * 2) - 1
LEFT_Y = 37 * 2
RIGHT_X1 = (43 * 2) - 1
RIGHT_X2 = (46 * 2) - 1
RIGHT_Y = 43 * 2
PADDING = 30
IMAGE_SIZE = 224

def get_left_eye(img,row):

    l_x1 = row[LEFT_X1] - PADDING
    l_x2 = row[LEFT_X2] + PADDING
    l_width = l_x2-l_x1
    l_y1 = row[LEFT_Y] - round(l_width/2)
    l_y2 = l_y1+l_width #To be sure to have a square

    eye = img[l_y1:l_y2,l_x1:l_x2]
    eye = cv.resize(eye,(IMAGE_SIZE,IMAGE_SIZE))

    return eye

def get_right_eye(img,row):

    r_x1 = row[RIGHT_X1] - PADDING
    r_x2 = row[RIGHT_X2] + PADDING
    r_width = r_x2-r_x1
    r_y1 = row[RIGHT_Y] - round(r_width/2)
    r_y2 = r_y1+r_width #To be sure to have a square

    eye = img[r_y1:r_y2,r_x1:r_x2]
    eye = cv.resize(eye,(IMAGE_SIZE,IMAGE_SIZE))

    return eye

def predict(eye):
    eye = eye/255.0
    
    y_pred = model_iris_edge.predict(np.expand_dims(eye, axis=0))[0]
    normalized = (y_pred[:, :, 1] - np.min(y_pred[:, :, 1])) / (np.max(y_pred[:, :, 1]) - np.min(y_pred[:, :, 1]))
    normalized = normalized * 255
    normalized = normalized > 150
    normalized = normalized * 255
    normalized = normalized.astype(np.uint8)

    return normalized
    
def mean_rgb(img,mask):
    mask = np.array(mask).flatten()
    red = np.array(img[:,:,2]).flatten()
    green = np.array(img[:,:,1]).flatten()
    blue = np.array(img[:,:,0]).flatten()

    r = round(np.mean(red[np.nonzero(mask)]))
    g = round(np.mean(green[np.nonzero(mask)]))
    b = round(np.mean(blue[np.nonzero(mask)]))

    return r,g,b

def median_rgb(img,mask):
    mask = np.array(mask).flatten()
    red = np.array(img[:,:,2]).flatten()
    green = np.array(img[:,:,1]).flatten()
    blue = np.array(img[:,:,0]).flatten()

    r = round(np.median(red[np.nonzero(mask)]))
    g = round(np.median(green[np.nonzero(mask)]))
    b = round(np.median(blue[np.nonzero(mask)]))

    return r,g,b

    

df = pd.read_excel("data/landmarks68.xlsx")
model_iris_edge = load_model('models/MobileNetV2_Iris_Seg_10May.h5')

df_eyes = pd.DataFrame()

for index, row in df.iterrows():
    filename = row["Filename"]
    path = "output/" + filename
    img = cv.imread(path)

    left_eye = get_left_eye(img,row)
    left_mask = predict(left_eye)

    right_eye = get_right_eye(img,row)
    right_mask = predict(right_eye)

    l_median_r,l_median_g,l_median_b = median_rgb(left_eye,left_mask)
    l_mean_r,l_mean_g,l_mean_b = mean_rgb(left_eye,left_mask)

    r_median_r,r_median_g,r_median_b = median_rgb(right_eye,right_mask)
    r_mean_r,r_mean_g,r_mean_b = mean_rgb(right_eye,right_mask)

    dict = {"Filename":filename,
            "Left Median Red":l_median_r,
            "Left Median Green":l_median_g,
            "Left Median Blue":l_median_b,            
            "Right Median Red":r_median_r,
            "Right Median Green":r_median_g,
            "Right Median Blue":r_median_b,            
            "Left Mean Red":l_mean_r,
            "Left Mean Green":l_mean_g,
            "Left Mean Blue":l_mean_b,            
            "Right Mean Red":r_mean_r,
            "Right Mean Green":r_mean_g,
            "Right Mean Blue":r_mean_b}

    df_eyes = df_eyes.append(dict, ignore_index = True)
    df_eyes.to_excel("eyes.xlsx")

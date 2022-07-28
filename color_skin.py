import numpy as np
import cv2 as cv
import pandas as pd
import os
import shutil

POINTS_L = [1, 32, 49, 4]
POINTS_R = [17, 36, 55, 14]
#POINTS_L = [37, 32, 49]
#POINTS_R = [46, 36, 55]
id_points_l = "_".join(map(lambda x: str(x),POINTS_L)) # [1,2,3] -> 1_2_3
id_points_r = "_".join(map(lambda x: str(x),POINTS_R)) # [1,2,3] -> 1_2_3
id_points = id_points_l + "__" + id_points_r

cols_x_l = list(map(lambda p: (p*2)-1,POINTS_L))
cols_y_l = list(map(lambda p: (p*2),POINTS_L))
cols_x_r = list(map(lambda p: (p*2)-1,POINTS_R))
cols_y_r = list(map(lambda p: (p*2),POINTS_R))


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

def get_corners(row,cols_x,cols_y):
    x = list(row[cols_x])
    y = list(row[cols_y])
    corners = list(zip(x,y))
    corners = np.array([corners], dtype=np.int32)
    return corners
    

#Remove and recreate the the segmentation folder
segmentation_folder = "segmentation__" + id_points
shutil.rmtree(segmentation_folder,ignore_errors=True)
os.mkdir(segmentation_folder)

#Read the landmarks files
df = pd.read_excel("data/landmarks68.xlsx")
df_skin = pd.DataFrame()


for index, row in df.iterrows():
    filename = row["Filename"]
    path = "output/" + filename
    img = cv.imread(path)

    mask = np.ones(img.shape[:2], dtype=np.uint8)
    mask.fill(0)

    left_corners = get_corners(row,cols_x_l,cols_y_l)
    right_corners = get_corners(row,cols_x_r,cols_y_r)    
    cv.fillPoly(mask, left_corners, 255)
    cv.fillPoly(mask, right_corners, 255)    

    masked_image = cv.bitwise_or(img, np.dstack([mask]*3))
    cv.imwrite(segmentation_folder + "/" + filename + ".png",masked_image)

    median_r,median_g,median_b = median_rgb(img,mask)
    mean_r,mean_g,mean_b = mean_rgb(img,mask)

    dict = {"Filename":filename,
            "Median Red":median_r,
            "Median Green":median_g,
            "Median Blue":median_b,            
            "Mean Red":mean_r,
            "Mean Green":mean_g,
            "Mean Blue":mean_b}

    df_skin = df_skin.append(dict, ignore_index = True)

df_skin.to_excel("data/skin__"+id_points+".xlsx")

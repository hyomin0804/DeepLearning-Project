# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:10:20 2019

@author: user
"""

import glob
import os 
import pydicom
import cv2
import numpy as np
from read_roi import read_roi_zip


def get_LUT_value(data, window, level):
    try:
        window = window[0]
    except TypeError:
        pass
    try:
        level = level[0]
    except TypeError:
        pass

    return np.piecewise(data,
                        [data <= (level - 0.5 - (window - 1) / 2),
                         data > (level - 0.5 + (window - 1) / 2)],
                        [0, 255, lambda data: ((data - (level - 0.5)) /
                         (window - 1) + 0.5) * (255 - 0)])
    


group_number_list=['1','2','3']

for group_number in group_number_list:
    INPUT_FOLDER = 'D:/Project/spine/rawdata/group_%s/'%group_number
    #INPUT_FOLDER = 'D:/Project/spine/rawdata/group_2/'
    #INPUT_FOLDER = 'D:/Project/spine/rawdata/group_3/'
    patients = os.listdir(INPUT_FOLDER)
    patients.sort()
    
    for patient in patients:
        dcm_list=glob.glob(INPUT_FOLDER+patient+"/*.dcm")
        roi_zip_list=glob.glob(INPUT_FOLDER+patient+"/*.zip")
        
        for dcm in dcm_list:
            ds=pydicom.read_file(dcm)
            ArrayDicom = ds.pixel_array
    #        if not ds.RescaleSlope == '':
    #            ArrayDicom=ds.RescaleSlope*ArrayDicom+ds.RescaleIntercept
            ArrayDicom=get_LUT_value(ArrayDicom, ds.WindowWidth, ds.WindowCenter)
            diff=ArrayDicom.shape[0]-ArrayDicom.shape[1]
            diff_2=int(diff/2)
            if diff_2+diff_2 < diff:
                sum_numpy=np.zeros((ArrayDicom.shape[0],int(diff_2)))
                sum_numpy_=np.zeros((ArrayDicom.shape[0],int(diff_2)+1))
                tmp=np.append(sum_numpy, ArrayDicom, axis=-1)
                add_ArrayDicom=np.append(tmp, sum_numpy_, axis=-1)
            else:
                sum_numpy=np.zeros((ArrayDicom.shape[0],int(diff_2)))
                tmp=np.append(sum_numpy, ArrayDicom, axis=-1)
                add_ArrayDicom=np.append(tmp, sum_numpy, axis=-1)
            
            file_name=dcm.split("/")[-1].split("\\")[0]+"_"+dcm.split("\\")[-1].split(".")[0].replace("0000","")
            cv2.imwrite("D:/Project/spine/data/img_%s/%s_%s.jpg"%(group_number,group_number,file_name),add_ArrayDicom)
        
        
        for roi in roi_zip_list:
            if group_number == 1:
                if "0000.zip" in roi:
                    ds=pydicom.read_file(roi.replace(".zip",".dcm"))
                else:
                    ds=pydicom.read_file(roi.replace(".zip","0000.dcm"))
            elif group_number == 2:            
                ds=pydicom.read_file(roi.replace(".zip",".dcm"))
            
            elif group_number == 3:            
                ds=pydicom.read_file(roi.replace(".zip","0000.dcm"))
            
            ArrayDicom = ds.pixel_array
            rois = read_roi_zip(roi)
            roi_key=list(rois.keys())
            mask_tmp = np.zeros((ArrayDicom.shape[0], ArrayDicom.shape[1]), np.uint8)
            for j,nam in enumerate(roi_key):
                if rois[nam]['type']=='polygon' or rois[roi]['type'] == 'freehand':
                    roi_xy=list(zip(rois[nam]['x'],rois[nam]['y']))
                    roi_xy=np.array(roi_xy)
                    mask = np.zeros((ArrayDicom.shape[0], ArrayDicom.shape[1]), np.uint8)
                    mask2 = np.zeros((ArrayDicom.shape[0]+2, ArrayDicom.shape[1]+2), np.uint8)               
                    roi_xy = roi_xy.reshape((-1,1,2))
                    img = cv2.polylines(mask[:,:],[roi_xy],True,(255,255,255))
                    cv2.floodFill(img, mask2, (0,0), 255)
                    img = cv2.bitwise_not(img)
                    mask[:,:]=img
                    mask_tmp[mask==255]=255
                else:
                    print(rois[nam]['type'])
                    print(roi)
                    print("error!!!!!!!!!!!!!!!!!!!")
                    
                    
            diff=ArrayDicom.shape[0]-ArrayDicom.shape[1]
            diff_2=int(diff/2)
            if diff_2+diff_2 < diff:
                sum_numpy=np.zeros((ArrayDicom.shape[0],int(diff_2)))
                sum_numpy_=np.zeros((ArrayDicom.shape[0],int(diff_2)+1))
                tmp=np.append(sum_numpy, mask_tmp, axis=-1)
                add_mask=np.append(tmp, sum_numpy_, axis=-1)
            else:
                sum_numpy=np.zeros((ArrayDicom.shape[0],int(diff_2)))
                tmp=np.append(sum_numpy, mask_tmp, axis=-1)
                add_mask=np.append(tmp, sum_numpy, axis=-1)
            file_name=roi.split("/")[-1].split("\\")[0]+"_"+roi.split("\\")[-1].split(".")[0].replace("0000","")
            cv2.imwrite("D:/Project/spine/data/mask_%s/%s_%s.jpg"%(group_number,group_number,file_name),add_mask)

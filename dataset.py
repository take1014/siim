#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import os

from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
import ast
import glob
import pydicom
import cv2

import torch
import torch.utils.data as data

# dicom_obj:
# tag           name                value
#(0008, 0100) Code Value          SH: '113109'
def process_dicom(dicom_obj):
    pixel_data_tag=(0x7fe0, 0x0010) #ignore the pixel data
    data_dict={}
    for x in dicom_obj:
        if x.tag==pixel_data_tag:
            continue
        data_dict[x.name] = x.value
        assert x.value == dicom_obj[x.tag].value
    return data_dict

def get_train_data(dataset_dir):

    train_data = None

    if os.path.exists('./train_data.csv'):
        train_data =  pd.read_csv('./train_data.csv')
    else:
        # read csv
        train_image_level = pd.read_csv(dataset_dir + '/train_image_level.csv')
        train_study_level = pd.read_csv(dataset_dir + '/train_study_level.csv')

        # StudyInstanceUID$B$NNs$rDI2C!#!J(Bid$B$+$i(B_study$B$r%9%W%j%C%H$7$?$b$N!K(B
        train_study_level['StudyInstanceUID'] = train_study_level['id'].apply(lambda x: x.replace('_study', ''))

        # train_image_level$B$K%^!<%8$9$k(B
        # $B%^!<%8$7$?$H$-$KF1$8NsL>$O%j%M!<%`$5$l$k(B
        train_image_level = train_image_level.merge(train_study_level, on='StudyInstanceUID')

        # id_x:image_name, id_y:StudyInstanceUID+_study$B$OITMW$J$N$G:o=|$9$k(B
        train_data = train_image_level.drop(['id_x','id_y'], axis=1)

        # $B2hA|$X$N%U%k%Q%9:n@.(B
        train_paths = []
        for sid in tqdm(train_data['StudyInstanceUID']):
            train_paths.append(glob.glob(dataset_dir + '/train/' + sid + '/*/*')[0])

        # $B%^!<%88e$N%G!<%?$K%Q%9$rDI2C(B
        train_data['path'] = train_paths

        train_data.to_csv('./train_data.csv', index=False)

        # png$B2hA|$N@8@.(B
        for _, row in tqdm(train_data.iterrows()):
            axis_list = []
            if 'non' not in row.label:
                # $BJ8;zNs$rCM$KJQ49$9$k(B
                dict_list = ast.literal_eval(row.boxes)
                for d in dict_list:
                    axis_list.append((d['x'], d['y'], d['width'], d['height']))
            else:
                non_count+=1

            ds = pydicom.dcmread(row.path)
            img = ds.pixel_array
            cv2.imwrite('./image/{}.png'.format(os.path.split(row.path)[-1][:-4]), img)

    return train_data


def create_annolist(anno_box, anno_label, w_ratio, h_ratio):

    splited_labels_list = anno_label.split(' ')

    # output
    boxes_list = []
    labels_list = []
    if splited_labels_list[0] == 'none':
        boxes_list.append([0, 0, 1, 1])
        labels_list.append([0, 0, 1, 1])
    else:
        # create boxes list
        boxes_dict_list = ast.literal_eval(anno_box)
        for anno_dict in boxes_dict_list:
            boxes_list.append((anno_dict['x']*w_ratio, anno_dict['y']*h_ratio,
                               anno_dict['width']*w_ratio, anno_dict['height']*h_ratio))

        # label$B$,B8:_$9$k>l9g!"(B
        # opacity 1 x y width height opacity 1 x y width height...$B$,O"B3$9$k$N$G!"(B
        # x, y, width, height$B$N$_<h$j=P$9(B
        np_labels_list = np.array([splited_labels_list[i] for i in range(len(splited_labels_list)) if i%6 > 1], dtype=np.float32)
        # box$B$O(Bx, y, width, height$B$N7+$jJV$7$J$N$G!"(B(row, col) = (n, 4)$B$N9TNs$KJQ49$9$k(B
        np_labels_list = np_labels_list.reshape(np_labels_list.size//4, 4)
        # $B%j%5%$%:8e$N:BI87O$KE}0l(B
        print(np_labels_list.shape)
        for np_label in np_labels_list:
            np_label[0] *= w_ratio
            np_label[1] *= h_ratio
            np_label[2] *= w_ratio
            np_label[3] *= h_ratio
        # ndarray$B$r(Blist$B$KJQ49(B
        labels_list = np_labels_list.tolist()

    return boxes_list, labels_list


class SIIMDataset(data.Dataset):
    def __init__(self):
        super(SIIMDataset, self).__init__()
        self.resize_sz   = (512, 512)
        self.dataset_dir = './data/'
        self.train_data  = get_train_data('/home/take/fun/dataset/covid-19')

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, ndx):
        # self.train_data's columns
        # yout can show print(self.train_data.columns)
        # [
        # 'boxes' : (x,y, width, height) dictionary,
        # 'label',
        # 'StudyInstanceUID': Instance name,
        # 'Negative for Pneumonia : $BGY1j$O1"@-$G$"$k(B',
        # 'Typical Appearance : $BE57?E*$J304Q(B',
        # 'Indeterminate Appearance : $BIT3NDj$J304Q(B',
        # 'Atypical Appearance : $BHsDj7A$N304Q(B',
        # 'path']
        dcm_image_name = os.path.split(self.train_data['path'][ndx])[-1]
        png_image_path = self.dataset_dir + dcm_image_name.replace('.dcm', '.png')

        # read image
        img = Image.open(png_image_path)
        w_ratio, h_ratio = self.resize_sz[0]/img.width, self.resize_sz[1]/img.height
        #print('width:{}, height{}, width_ratio{}, height_ratio{}'.format(img.width, img.height, w_ratio, h_ratio))
        img = img.resize(self.resize_sz)
        # convert to numpy array (dtype=np.float32)
        img = np.array(img, dtype=np.float32)
        img = img / 255   # normalize
        # convert to tensor and unsqueeze to add channel
        img = torch.from_numpy(img).unsqueeze(0)

        # annotation
        boxes_list, labels_list = create_annolist(self.train_data['boxes'][ndx],
                                                  self.train_data['label'][ndx],
                                                  w_ratio, h_ratio)

        return img, torch.FloatTensor(boxes_list), torch.FloatTensor(labels_list)

if __name__ == '__main__':
    #dataset_dir = '/home/take/fun/dataset/covid-19'
    #train_data = get_train_data(dataset_dir)
    dataset = SIIMDataset()
    image, boxes, label = dataset.__getitem__(121)
    image, boxes, label = dataset.__getitem__(0)
    print(image, boxes, label)

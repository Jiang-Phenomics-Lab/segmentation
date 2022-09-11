import re
import cv2
import os
import numpy as np
import shutil
import glob

def dataReg(img_dir, label_dir, out_dir):

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_img_dir = os.path.join(out_dir, 'image')
    out_label_dir = os.path.join(out_dir, 'label')
    if not os.path.isdir(out_img_dir):
        os.makedirs(out_img_dir)

    if not os.path.isdir(out_label_dir):
        os.makedirs(out_label_dir)

    list_files = os.listdir(img_dir)
    for f in list_files:
        if not f.endswith('.png'):
            continue
        img_path = os.path.join(img_dir, f)
        label_path = os.path.join(label_dir, f)
        
        if not os.path.isfile(img_path) or not os.path.isfile(label_path):
            continue

        label_img = cv2.imread(label_path)
        max_label = np.max(label_img)
        if max_label == 0:
            continue

        # copy files
        out_img_file = os.path.join(out_img_dir, f)
        out_label_file = os.path.join(out_label_dir, f)
        shutil.copyfile(img_path, out_img_file)
        shutil.copyfile(label_path, out_label_file)



    return

def splitData(in_dir, out_dir):

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    img_dir = os.path.join(in_dir, 'image')
    label_dir = os.path.join(in_dir, 'label')

    file_list = os.listdir(img_dir)

    dataset_size = len(file_list)
    indices = list(range(dataset_size))
    test_split = int(np.floor(0.2*dataset_size))
    val_split = int(np.floor(0.36*dataset_size))
    np.random.seed(0)
    np.random.shuffle(indices)
    train_indices = indices[val_split:]
    val_indices = indices[test_split:val_split]
    test_indices = indices[:test_split]

    train_path_list = []
    test_path_list = []
    val_path_list = []
    for idx in train_indices:
        if file_list[idx].endswith('.png'):
            train_path_list.append(file_list[idx])

    for idx in val_indices:
        if file_list[idx].endswith('.png'):
            val_path_list.append(file_list[idx])

    for idx in test_indices:
        if file_list[idx].endswith('.png'):
            test_path_list.append(file_list[idx])
    


    out_sub_dir = os.path.join(out_dir, 'test')
    out_img_dir = os.path.join(out_sub_dir, 'image')
    out_label_dir = os.path.join(out_sub_dir, 'label')
    if not os.path.isdir(out_img_dir):
        os.makedirs(out_img_dir)
    if not os.path.isdir(out_label_dir):
        os.makedirs(out_label_dir)
    for f in test_path_list:
        img_path = os.path.join(img_dir, f)
        label_path = os.path.join(label_dir, f)
        out_img_file = os.path.join(out_img_dir, f)
        out_label_file = os.path.join(out_label_dir, f)
        shutil.copyfile(img_path, out_img_file)
        shutil.copyfile(label_path, out_label_file)

    out_sub_dir = os.path.join(out_dir, 'validation')
    out_img_dir = os.path.join(out_sub_dir, 'image')
    out_label_dir = os.path.join(out_sub_dir, 'label')
    if not os.path.isdir(out_img_dir):
        os.makedirs(out_img_dir)
    if not os.path.isdir(out_label_dir):
        os.makedirs(out_label_dir)
    for f in val_path_list:
        img_path = os.path.join(img_dir, f)
        label_path = os.path.join(label_dir, f)
        out_img_file = os.path.join(out_img_dir, f)
        out_label_file = os.path.join(out_label_dir, f)
        shutil.copyfile(img_path, out_img_file)
        shutil.copyfile(label_path, out_label_file)

    out_sub_dir = os.path.join(out_dir, 'train')
    out_img_dir = os.path.join(out_sub_dir, 'image')
    out_label_dir = os.path.join(out_sub_dir, 'label')
    if not os.path.isdir(out_img_dir):
        os.makedirs(out_img_dir)
    if not os.path.isdir(out_label_dir):
        os.makedirs(out_label_dir)
    for f in train_path_list:
        img_path = os.path.join(img_dir, f)
        label_path = os.path.join(label_dir, f)
        out_img_file = os.path.join(out_img_dir, f)
        out_label_file = os.path.join(out_label_dir, f)
        shutil.copyfile(img_path, out_img_file)
        shutil.copyfile(label_path, out_label_file)

    return

def main():

    img_dir = '/media/baker/C05A8B528B5B5A2D/data/ZhangBC/EM/trainData/grayscale'
    label_dir = '/media/baker/C05A8B528B5B5A2D/data/ZhangBC/EM/trainData/vesselLabel'
    level_1_dir = '/media/baker/C05A8B528B5B5A2D/data/ZhangBC/EM/Level_1/V1'
    level_2_dir = '/media/baker/C05A8B528B5B5A2D/data/ZhangBC/EM/Level_2/V1'

    # regular data
    #dataReg(img_dir, label_dir, level_1_dir)

    # split img to train test val
    splitData(level_1_dir, level_2_dir)

    return

main()

import torch
import os
import os, sys, argparse, cv2
import torch
import time
sys.path.insert(0, '/home/zy/Work/Codes/PytorchWorkSpace/segmentation2d')
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp
from dataset import vesselDataset
import dataAugmentation

def segmentation_predict(in_dir, out_dir, module_path):
    
    best_model = torch.load(module_path)
    # if trained by multiple gpu
    if isinstance(best_model, torch.nn.DataParallel):
        best_model = best_model.module
        
    ENCODER = 'resnet18'
    ENCODER_WEIGHTS = 'imagenet'
    DEVICE = 'cuda'
    CLASSES = ['target']
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    aug_val = dataAugmentation.get_validation_augmentation()
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    preprocessing = dataAugmentation.get_preprocessing(preprocessing_fn)

    list_files = os.listdir(in_dir)
    for f in list_files:
        if not f.endswith('.png'):
            continue
        image = cv2.imread(os.path.join(in_dir, f))
        sample = aug_val(image=image)
        image = sample['image']
    
        sample = preprocessing(image=image)
        image= sample['image']
        
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        out_img = np.zeros_like(pr_mask)
        fg_mask = pr_mask > 0
        out_img[fg_mask] = 255
        out_file_path = os.path.join(out_dir, f)
        cv2.imwrite(out_file_path, out_img)

    return

def test():

    in_dir = '/media/baker/C05A8B528B5B5A2D/data/ZhangBC/EM/testData/369/crop512'
    out_dir = '/media/baker/C05A8B528B5B5A2D/data/ZhangBC/EM/testData/369/testResults'
    model_path = '/home/baker/Work/Codes/Python/segmentation/unetPP_v5_best.pth'

    segmentation_predict(in_dir, out_dir, model_path)

    return

test()
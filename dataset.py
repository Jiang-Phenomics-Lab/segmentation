import os
import cv2, torch
import numpy as np
from torch.utils.data import  Dataset

class vesselDataset(Dataset):
    ''''''
    CLASSES = ['background', 'target']
    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None, transform=None):
        self.transform = transform
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.ids = os.listdir(self.images_dir)
        self.image_paths, self.mask_paths = self._make_dataset()
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def _make_dataset(self):
        image_paths = []
        mask_paths =  []
        for f in self.ids:
            img_path = os.path.join(self.images_dir, f)
            mask_path = os.path.join(self.masks_dir, f)
            if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
                continue
            image_paths.append(img_path)
            mask_paths.append(mask_path)
            
        return image_paths, mask_paths
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        
        image = cv2.imread(image_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return image, mask
    
    def __len__(self):
        return len(self.ids)
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
sys.path.insert(0, '/home/baker/Work/Codes/PytorchWorkSpace/segmentation2d')
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import BoneSeg2DDataset
import dataAugmentation

train_image_dir = '/home/baker/Work/data/DL/Segmentation/Level_2/V5/train/image'
train_mask_dir = '/home/baker/Work/data/DL/Segmentation/Level_2/V5/train/label'

val_image_dir = '/home/baker/Work/data/DL/Segmentation/Level_2/V5/validation/image'
val_mask_dir = '/home/baker/Work/data/DL/Segmentation/Level_2/V5/validation/label'

test_image_dir = '/home/baker/Work/data/DL/Segmentation/Level_2/V5/test/image'
test_mask_dir = '/home/baker/Work/data/DL/Segmentation/Level_2/V5/test/label'

CLASSES = ['femur', 'tibia']

aug_train = dataAugmentation.get_training_augmentation()
aug_val = dataAugmentation.get_validation_augmentation()

ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'

#aux_params = dict(pooling='avg', dropout=0.5, activation='sigmoid', classes=4,)

# create segmentation model with pretrained encoder
#model = smp.Unet(ENCODER, classes=4, aux_params=aux_params)
model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights=ENCODER_WEIGHTS, classes=len(CLASSES), activation=ACTIVATION)
#best_model_path = '/home/baker/Work/Codes/PytorchWorkSpace/segmentation2d/tasks/unetPP_v4__model.pth'
#state_dict = torch.load(best_model_path)
#model.load_state_dict(state_dict.state_dict())
model = torch.nn.DataParallel(model)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
preprocessing = dataAugmentation.get_preprocessing(preprocessing_fn)

train_dataset = BoneSeg2DDataset(
    train_image_dir, 
    train_mask_dir, 
    augmentation=aug_train, 
    preprocessing=preprocessing,
    classes=CLASSES,
)

valid_dataset = BoneSeg2DDataset(
    val_image_dir, 
    val_mask_dir, 
    augmentation=aug_val, 
    preprocessing=preprocessing,
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=16)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[1,]),
    smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0,]),
    smp.utils.metrics.IoU(threshold=0.5, ),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

# create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

train_epoch.metrics[0].__name__ = "IoU_0"
train_epoch.metrics[1].__name__ = "IoU_1"
train_epoch.metrics[2].__name__ = "iou_score"

# train model for 40 epochs

max_score = 0
total_epochs = 35

for i in range(0, total_epochs):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './unetPP_v5_best.pth')
        print('Model saved!')
    
    if i % 5 == 0:
        torch.save(model, './unetPP_v5_epoch{}.pth'.format(i))
        print('Model saved!')
        
    if i == 10:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')
    if i == 20:
        optimizer.param_groups[0]['lr'] = 1e-6
        print('Decrease decoder learning rate to 1e-6!')

# ranzcr_clip

## Models

#### Model 0 (baseline): efficientnet_b0, VAL_AUC: 0.939, LB: 0.940, EPOCHS: 24

###### Params
input_shape=(512x512)  
dropout_rate=0.5  
drop_connect_rate=0.5  

###### Augmentations
A.RandomResizedCrop(scale=(0.85, 1.0))  
A.HorizontalFlip()  

#### Model 1: efficientnet_b0, VAL_AUC: 0.944, LB: 0.945, EPOCHS: 23

###### Params
input_shape=(600x600)  
dropout_rate=0.5  
drop_connect_rate=0.5  

###### Augmentations
A.RandomResizedCrop(scale=(0.85, 1.0))  
A.HorizontalFlip()  

#### Model 2 : efficientnet_b3, VAL_AUC: 0.949, LB: 0.950, EPOCHS: 18

###### Params
input_shape=(512x512)  
dropout_rate=0.6 
drop_connect_rate=0.6

###### Augmentations
A.RandomResizedCrop(scale=(0.85, 1.0))  
A.HorizontalFlip()  

## Bugs and features
- [x] validation and test TTA
- [ ] NVIDIA DALI for dataloading
- [x] infinite increase val_roc_auc
- [x] tensorboard names
- [x] test synchronization for multi-gpu
- [x] stats by classes
- [x] multithreaded dataset loading with images caching

## Ideas to improve the model:
- [x] hflip TTA;  :white_check_mark:VAL_AUC: +0.001, :white_check_mark:LB: +0.002
- [ ] progressive input size
- [ ] label smoothing
  - [x] :x:VAL_AUC: -0.1 for eps=0.05, may be better for smaller eps and together with harder augmentations
- [ ] more augmentations
  - [x] :white_check_mark:VAL_AUC: +0.002, :x:LB: -0.001
- [x] loss weights or sampling
  - [x] weights ~ sqrt(neg / pos) clamped to [0.5, 2.0] :x:VAL_AUC: -0.001, :white_check_mark:LB: +0.004
- [ ] different heads
- [ ] rank_average
  - [x] :x:VAL_AUC: -0.42, didn't work for two predictions, may be better for ensemble
- [ ] 5 folds ensemble
- [ ] bigger efficientnet: 
  - [x] efficientnet_b3 512x512 :white_check_mark:VAL_AUC: +0.01, :white_check_mark:LB: +0.01

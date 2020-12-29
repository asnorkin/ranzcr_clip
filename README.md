# ranzcr_clip

## Models

#### Model 0 (baseline): efficientnet_b0 512x512, dropout=0.5, VAL_AUC: 0.939, LB: 0.940, EPOCHS: 23

###### Params
dropout_rate=0.5  
drop_connect_rate=0.5  

###### Augmentations
A.RandomResizedCrop(height=512, width=512, scale=(0.85, 1.0))  
A.HorizontalFlip()  

## Bugs and features
- [x] validation and test TTA
- [ ] infinite increase val_roc_auc
- [x] tensorboard names
- [ ] test synchronization for multi-gpu

## Ideas to improve the model:
- [x] hflip TTA;  :white_check_mark:VAL_AUC: +0.001, :white_check_mark:LB: +0.002
- [x] label smoothing;   :x:VAL_AUC: -0.1 for eps=0.05, may be better for eps=0.001
- [ ] more augmentations
- [ ] different heads
- [x] rank_average;   :x:VAL_AUC: -0.42, didn't work for two predictions, may be better for ensemble
- [ ] 5 folds ensemble
- [ ] bigger efficientnet

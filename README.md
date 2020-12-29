# ranzcr_clip

## Models

#### Model 0 (baseline): efficientnet_b0 512x512, dropout=0.5, VAL_AUC: 0.965, LB: 0.938

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
- [ ] more augmentations
- [ ] rank_average
- [ ] 5 folds ensemble
- [ ] bigger efficientnet

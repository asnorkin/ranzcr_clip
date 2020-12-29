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
- [ ] TTA for validation and test
- [ ] test synchronization for multi-gpu
- [ ] folds data leak

## Ideas to improve the model:
- [ ] hflip TTA
- [ ] more augmentations
- [ ] rank_average
- [ ] 5 folds ensemble
- [ ] bigger efficientnet

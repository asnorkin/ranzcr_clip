# ranzcr_clip

#### [Interactive Demo](http://3.123.65.222:8501/)

This repo contains solution code for
[RANZCR CLiP](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification)
Competition on Kaggle.


## Get Started
#### 1. Prepare
Clone the repo and install requirements
```
git clone https://github.com/asnorkin/ranzcr_clip.git
python -m pip install -r requirements.txt
```
Install DVC and load data and pretrained models
```
python -m pip install dvc dvc[gdrive]
dvc pull
```

#### 2. Training
Run training
```
cd ranzcr_clip
./train_segmentation.sh
```
Check train output log
```
tail -F -n 10000 segmentation/train_outputs/<experiment>.log
```
Visualize train logs on tensorboard
```
tensorboard --logdir=segmentation/logs
```

#### 3. Run Demo with custom model
Copy trained model into the service
```
cp segmentation/checkpoints/<experiment>/<model_name>.ckpt models/<model>/<model_name>.ckpt
```
Run the demo with new model
```
PYTHONPATH=ranzcr_clip:${PYTHONPATH} streamlit run streamlit_app.py
```

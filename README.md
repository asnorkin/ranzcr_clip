# ranzcr_clip

This repo contains the code for
[RANZCR CLiP](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification)
Competition on Kaggle.

## Interactive Demo
You can play with already trained models [here](http://3.123.65.222:8501/).

## Usage
#### 1. Prepare
Clone the repo and install requirements
```
git clone https://github.com/asnorkin/ranzcr_clip.git
python -m pip install -r requirements.txt
```
Install DVC, load the dataset and pretrained models
```
python -m pip install dvc dvc[gdrive]
dvc pull
```
Or just models
```
dvc pull ranzcr_clip/models
```

#### 2. Training
Run training
```
cd ranzcr_clip
./train_segmentation.sh
```
Check training output log
```
tail -F -n 10000 segmentation/train_outputs/<experiment>.log
```
Visualize training logs on tensorboard
```
tensorboard --logdir=segmentation/logs
```

#### 3. Run Demo with custom model
Copy trained model into the service
```
cp segmentation/checkpoints/<experiment>/<model_name>.ckpt models/<model>/<model_name>.ckpt
```
Run the demo with new model locally
```
PYTHONPATH=ranzcr_clip:${PYTHONPATH} streamlit run app/app.py
```

#### 4. Deploy demo
```
docker-compose up --build -d
```

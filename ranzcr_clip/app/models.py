import os.path as osp
from typing import Dict, Tuple

import numpy as np
import streamlit as st
from pytorch_lightning import LightningModule

import common
from ranzcr_clip.classification.module import XRayClassificationModule
from ranzcr_clip.predictors.predictor import TorchModelPredictor
from ranzcr_clip.segmentation.module import XRaySegmentationModule


LABELS = [
    'ETT - Abnormal',
    'ETT - Borderline',
    'ETT - Normal',
    'NGT - Abnormal',
    'NGT - Borderline',
    'NGT - Incompletely Imaged',
    'NGT - Normal',
    'CVC - Abnormal',
    'CVC - Borderline',
    'CVC - Normal',
    'SGC - Present',
]


# Turn off model hashing because only confidence threshold will be change
@st.cache(hash_funcs={common.model_utils.ModelConfig: lambda _: None}, allow_output_mutation=True)
def load_model(model_dir: str, model_class: LightningModule) -> TorchModelPredictor:
    return TorchModelPredictor.create_from_checkpoints(model_dir, model_class)


@st.cache
def classification(image: np.ndarray, models_dir: str, threshold: float) -> Dict[str, str]:

    # Load classifier and setup confidence
    classifier = load_model(osp.join(models_dir, 'efficientnet_b5'), XRayClassificationModule)
    classifier.config.confidence_threshold = threshold

    # Classify
    predictions = classifier.predict(image, preprocess=True, output='binary', tta=False)
    class_labels = [LABELS[class_id] for class_id, prediction in enumerate(predictions) if prediction == 1]

    return {label[:3]: label[6:] for label in class_labels}


@st.cache
def segmentation(
    image: np.ndarray,
    models_dir: str,
    catheter_mask_threshold: float,
    lung_mask_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:

    # Catheter
    catheter_segmenter = load_model(osp.join(models_dir, 'unet_catheter'), XRaySegmentationModule)
    catheter_segmenter.config.confidence_threshold = catheter_mask_threshold
    catheter_mask = catheter_segmenter.predict(image, preprocess=True, output='binary', tta=False)

    # Lung
    lung_segmenter = load_model(osp.join(models_dir, 'unet_lung'), XRaySegmentationModule)
    lung_segmenter.config.confidence_threshold = lung_mask_threshold
    lung_mask = lung_segmenter.predict(image, preprocess=True, output='binary', tta=False)

    return catheter_mask, lung_mask

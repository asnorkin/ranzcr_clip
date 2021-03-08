from argparse import Namespace
from typing import Dict, Tuple

import cv2 as cv
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from pytorch_lightning import LightningModule

import common
from ranzcr_clip.classification.module import XRayClassificationModule
from ranzcr_clip.predictors.predictor import TorchModelPredictor
from ranzcr_clip.segmentation.module import XRaySegmentationModule
from ranzcr_clip import streamlit_state


# Models
CLASSIFIER_MODEL_DIR = {'EfficientNet-b5': 'ranzcr_clip/models/efficientnet_b5'}
CATHETER_SEGMENTATION_DIR = 'ranzcr_clip/models/unet_catheter'
LUNG_SEGMENTATION_DIR = 'ranzcr_clip/models/unet_lung'


# Labels
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
CATHETHERS = ['ETT', 'NGT', 'CVC', 'SGC']
CATHETER_COLORS = [(255, 204, 255), (255, 255, 204), (204, 255, 255), (224, 224, 224)]


INSTRUCTIONS = """
This project demonstrates the solution of Kaggle
[RANZCR CLiP Competition](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification)
into a simple interactive app.

👈 **Please upload your Chest X-Ray scan in the sidebar to start or use demo scan**.
"""


EXPLANATION = """
The algorithm processes Chest X-Ray Scan into 3 stages:
- Calculates lung masks
- Calculates catheters masks
- Checks catheters positions

You can read more about catheters and their positions in the
[Competition Description](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/data) on Kaggle.
"""


def instructions():
    # Session state for demo image
    # state = streamlit_state.get(use_demo=False)

    # Show instructions
    st.markdown(INSTRUCTIONS)
    # state.use_demo = state.use_demo or st.button('Use demo scan')


# Turn off model hashing because only confidence threshold will be change
@st.cache(hash_funcs={common.model_utils.ModelConfig: lambda _: None}, allow_output_mutation=True)
def load_model(model_dir: str, model_class: LightningModule) -> TorchModelPredictor:
    return TorchModelPredictor.create_from_checkpoints(model_dir, model_class)


@st.cache
def classification(image: np.ndarray, model_type: str, threshold: float) -> Dict[str, str]:

    # Load classifier and setup confidence
    classifier = load_model(CLASSIFIER_MODEL_DIR[model_type], XRayClassificationModule)
    classifier.config.confidence_threshold = threshold

    # Classify
    predictions = classifier.predict(image, preprocess=True, output='binary', tta=False)
    class_labels = [LABELS[class_id] for class_id, prediction in enumerate(predictions) if prediction == 1]

    return {label[:3]: label[6:] for label in class_labels}


@st.cache
def segmentation(
    image: np.ndarray,
    catheter_mask_threshold: float,
    lung_mask_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:

    # Catheter
    catheter_segmenter = load_model(CATHETER_SEGMENTATION_DIR, XRaySegmentationModule)
    catheter_segmenter.config.confidence_threshold = catheter_mask_threshold
    catheter_mask = catheter_segmenter.predict(image, preprocess=True, output='binary', tta=False)

    # Lung
    lung_segmenter = load_model(LUNG_SEGMENTATION_DIR, XRaySegmentationModule)
    lung_segmenter.config.confidence_threshold = lung_mask_threshold
    lung_mask = lung_segmenter.predict(image, preprocess=True, output='binary', tta=False)

    return catheter_mask, lung_mask


def scan_uploader_ui() -> np.ndarray:
    # Session state for demo image
    state = streamlit_state.get(use_demo=False)

    # Left panel title, use demo button and scan upload form
    st.sidebar.header('Chest X-Ray Scan')
    state.use_demo = state.use_demo or st.sidebar.button('Use demo scan')
    uploaded_scan = st.sidebar.file_uploader('Upload a chest x-ray scan', type=['png', 'jpg', 'jpeg'])

    # If no scan was loaded but use demo button clicked
    if uploaded_scan is None and state.use_demo:
        state.use_demo = True
        uploaded_scan = 'demo_scan.jpg'

    # Load scan and convert to grayscale
    if uploaded_scan is not None:
        uploaded_scan = np.array(Image.open(uploaded_scan).convert('L'))

    return uploaded_scan


def model_ui() -> Namespace:
    st.sidebar.header('Model')

    model_params = Namespace()

    # Classification model
    model_params.model_type = st.sidebar.selectbox('Choose the model', list(CLASSIFIER_MODEL_DIR.keys()))
    model_params.classification_threshold = st.sidebar.slider('Confidence', 0.0, 1.0, 0.5, step=0.01)

    # Catheter segmentation
    model_params.show_catheter_mask = st.sidebar.checkbox('Show catheter mask', value=True)
    if model_params.show_catheter_mask:
        model_params.catheter_mask_opacity = st.sidebar.slider('Catheter mask opacity', 0.0, 1.0, 0.3, step=0.01)
        model_params.catheter_mask_threshold = st.sidebar.slider('Catheter mask threshold', 0.0, 1.0, 0.5, step=0.01)
    else:
        model_params.catheter_mask_threshold = 0.5

    # Lung segmentation
    model_params.show_lung_mask = st.sidebar.checkbox('Show lung mask', value=True)
    if model_params.show_lung_mask:
        model_params.lung_mask_opacity = st.sidebar.slider('Mask opacity', 0.0, 1.0, 0.3, step=0.01)
        model_params.lung_mask_threshold = st.sidebar.slider('Mask threshold', 0.0, 1.0, 0.5, step=0.01)
    else:
        model_params.lung_mask_threshold = 0.5

    return model_params


def add_mask(
    image: np.ndarray, mask: np.ndarray, opacity: float, color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    color = np.asarray(color)

    # Prepare mask
    mask = cv.GaussianBlur(mask, ksize=(25, 25), sigmaX=0)

    image[mask > 0] = (opacity * color + (1 - opacity) * image)[mask > 0].astype(np.uint8)
    return image


def draw_classification_result(image, classification_result):
    # Set up constants
    offset = int(0.04 * image.shape[0])
    x1, y1 = offset, int(0.5 * offset)

    # Load font
    font = ImageFont.truetype('./OpenSans-Regular.ttf', int(0.7 * offset))

    # Calculate dynamic values
    dx_key = max([font.getsize(key + ' ' * 5)[0] for key in classification_result])
    dx_full = max([font.getsize(key + ' ' * 5 + val)[0] for key, val in classification_result.items()])
    x2 = x1 + int(0.4 * offset) + dx_full
    y2 = y1 + offset * len(classification_result)

    # Blur place for classification result
    image[y1:y2, x1:x2] = cv.GaussianBlur(image[y1:y2, x1:x2], (521, 521), 0)

    # Create PIL objects Image and Draw
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    # Draw
    for i, (catheter, state) in enumerate(classification_result.items()):
        color = CATHETER_COLORS[CATHETHERS.index(catheter)]
        draw.text((x1 + int(0.2 * offset), y1 + offset * i), catheter, color, font=font)
        draw.text((x1 + int(0.2 * offset) + dx_key, y1 + offset * i), state, color, font=font)

    return np.asarray(image)


def process_image(
    uploaded_scan: np.ndarray,
    classification_result: Dict[str, str],
    catheter_mask: np.ndarray,
    lung_mask: np.ndarray,
    params: Namespace,
) -> np.ndarray:
    result_image = np.copy(uploaded_scan)
    result_image = cv.cvtColor(result_image, cv.COLOR_GRAY2RGB)

    # Catheters
    if params.show_catheter_mask:
        for i, _catheter in enumerate(CATHETHERS):
            opacity, color = params.catheter_mask_opacity, CATHETER_COLORS[i]
            result_image = add_mask(result_image, catheter_mask[..., i], opacity, color)

    # Lungs
    if params.show_lung_mask:
        lung_color = (204, 204, 255)
        result_image = add_mask(result_image, lung_mask, params.lung_mask_opacity, lung_color)

    # Classification report
    result_image = draw_classification_result(result_image, classification_result)

    return result_image


def draw_result_ui(
    uploaded_scan: np.ndarray,
    classification_result: Dict[str, str],
    catheter_mask: np.ndarray,
    lung_mask: np.ndarray,
    params: Namespace,
) -> None:
    result_image = process_image(uploaded_scan, classification_result, catheter_mask, lung_mask, params)
    st.markdown(EXPLANATION)
    st.image(result_image, caption='Chest X-Ray Scan Processing Result', use_column_width=True)

    # st.subheader('Catheter Classification Report')
    # st.write('  \n'.join(classification_result.values()))


def main():
    st.title('RANZCR CLiP Solution Demo')

    # Load scan
    uploaded_scan = scan_uploader_ui()
    if uploaded_scan is None:
        instructions()
        return

    # Load params from UI
    params = model_ui()

    # Predict
    classification_result = classification(uploaded_scan, params.model_type, params.classification_threshold)
    catheter_mask, lung_mask = segmentation(uploaded_scan, params.catheter_mask_threshold, params.lung_mask_threshold)

    # Draw result
    draw_result_ui(uploaded_scan, classification_result, catheter_mask, lung_mask, params)


if __name__ == '__main__':
    main()

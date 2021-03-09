from argparse import Namespace
from enum import Enum
from typing import Dict, Optional, Tuple

import cv2 as cv
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from utils import get_images_list, load_image, upload_image


# Labels
CATHETHERS = ['ETT', 'NGT', 'CVC', 'SGC']
CATHETER_COLORS = [(255, 204, 255), (255, 255, 204), (204, 255, 255), (224, 224, 224)]


INSTRUCTIONS = """
This is an interactive demo of the solution of Kaggle
[RANZCR CLiP Competition](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification).

The goal of the competition is to check if all the catheters on chest x-ray scan are in normal positions
as abnormal catheters positions can hurt the patient.

👈 **Please select Chest X-Ray scan in the sidebar to start**.
"""


EXPLANATION = """
The algorithm processes Chest X-Ray Scan into 3 stages:
- Calculates lung masks
- Calculates catheter masks
- Checks catheters positions

You can read more about catheters and their positions in the
[Competition Description](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/data) on Kaggle.
"""


class ImageStatus(Enum):
    OK = 0
    LOADING_ERROR = 1
    NOT_UPLOADED = 2
    NOT_SELECTED = 3

    def is_ok(self) -> bool:
        return self.value == 0  # pylint: disable=W0143

    def hint(self) -> str:
        if self.value == 1:  # pylint: disable=W0143
            return 'Can\'t load image'
        if self.value == 2:  # pylint: disable=W0143
            return 'Please, upload the image'
        if self.value == 3:  # pylint: disable=W0143
            return 'Please, select the image'

        return 'Image successfully loaded'


class InterfaceType(Enum):
    SIMPLE = 0
    ADVANCED = 1

    @classmethod
    def from_string(cls, string):
        string = string.lower()
        if string == 'simple':
            return cls.SIMPLE

        if string == 'advanced':
            return cls.ADVANCED

        raise ValueError(f'Unexpected interface type string: {string}')


def select_image(path_to_images: str) -> (ImageStatus, Optional[np.ndarray]):
    """Show interface to choose the image, and load it
    Args:
        path_to_images (dict): path ot folder with images
    Returns:
        (status, image)
    """
    image_names_list = get_images_list(path_to_images)
    if len(image_names_list) < 1:
        return ImageStatus.LOADING_ERROR, None

    st.sidebar.subheader('Scan')
    image_options = ['Select X-Ray Scan'] + sorted(image_names_list) + ['Upload my image']
    image_name = st.sidebar.selectbox('', image_options)

    if image_name == 'Select X-Ray Scan':
        return ImageStatus.NOT_SELECTED, None

    try:
        if image_name != 'Upload my image':
            image = load_image(image_name, path_to_images)
        else:
            image = upload_image()

        return ImageStatus.OK, image

    except cv.error:
        return ImageStatus.LOADING_ERROR, None
    except AttributeError:
        return ImageStatus.NOT_UPLOADED, None


def select_interface() -> InterfaceType:
    st.sidebar.subheader('Interface')
    interface_type = st.sidebar.radio('', ['Simple', 'Advanced'])
    return InterfaceType.from_string(interface_type)


def select_model_params(interface_type: InterfaceType) -> Namespace:

    model_params = Namespace(
        classification_threshold=0.5,
        catheter_mask_opacity=0.3,
        catheter_mask_threshold=0.5,
        lung_mask_opacity=0.3,
        lung_mask_threshold=0.5,
        min_rel_blob_area=1e-4,
        tta=False,
    )

    sliders_count = 0

    def slider(label, default):
        nonlocal sliders_count
        key = f'{label}-{sliders_count}'
        sliders_count += 1
        return st.sidebar.slider(label, 0.0, 1.0, default, step=0.01, key=key)

    # Mask show checkboxes
    model_params.show_catheter_mask = st.sidebar.checkbox('Show catheter mask', value=True)
    model_params.show_lung_mask = st.sidebar.checkbox('Show lung mask', value=True)

    # Advanced interface params
    if interface_type == InterfaceType.ADVANCED:
        # Filter small blobs flag
        if not st.sidebar.checkbox('Filter small blobs', value=True):
            model_params.min_rel_blob_area = 0.0
        model_params.tta = st.sidebar.checkbox('HFlip TTA', value=False)

        st.sidebar.header('Parameters')

        # Classification
        st.sidebar.subheader('Catheters Position Classification')
        model_params.classification_threshold = slider('Confidence threshold', model_params.classification_threshold)

        # Catheters segmentation
        if model_params.show_catheter_mask:
            st.sidebar.subheader('Catheters Segmentation')
            model_params.catheter_mask_opacity = slider('Opacity', model_params.catheter_mask_opacity)
            model_params.catheter_mask_threshold = slider('Mask threshold', model_params.catheter_mask_threshold)

        # Lung segmentation
        if model_params.show_lung_mask:
            st.sidebar.subheader('Lungs Segmentation')
            model_params.lung_mask_opacity = slider('Opacity', model_params.lung_mask_opacity)
            model_params.lung_mask_threshold = slider('Mask threshold', model_params.lung_mask_threshold)

    return model_params


def instructions() -> None:
    st.markdown(INSTRUCTIONS)


def explanation() -> None:
    st.markdown(EXPLANATION)


def filter_small_blobs(mask: np.ndarray, min_rel_area: float = 3e-4) -> np.ndarray:
    total_area = mask.shape[0] * mask.shape[1]

    # Find blob contours
    _, thresh = cv.threshold(mask, 127, 255, 0)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Extract small contours and filter them
    small_contours = [contour for contour in contours if cv.contourArea(contour) / total_area < min_rel_area]
    if len(small_contours) > 0:
        # Convert to RGB and back because drawContours didn't work for grayscale
        mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
        cv.drawContours(mask, small_contours, -1, (0, 0, 0), -1)
        mask = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)

    return mask


def add_mask(
    image: np.ndarray,
    mask: np.ndarray,
    opacity: float,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    color = np.asarray(color)

    # Prepare mask
    mask = cv.GaussianBlur(mask, ksize=(25, 25), sigmaX=0)

    image[mask > 0] = (opacity * color + (1 - opacity) * image)[mask > 0].astype(np.uint8)
    return image


def draw_classification_result(image: np.ndarray, classification_result: Dict[str, str]) -> np.ndarray:
    if len(classification_result) == 0:
        return image

    # Set up constants
    offset = int(0.04 * image.shape[0])
    x1, y1 = offset, int(0.5 * offset)

    # Load font
    font = ImageFont.truetype('./app/artifacts/fonts/OpenSans-Regular.ttf', int(0.7 * offset))

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

    # Copy masks to prevent output of st.cache functions mutation
    catheter_mask, lung_mask = np.copy(catheter_mask), np.copy(lung_mask)

    # Remove small blobs from masks
    lung_mask = filter_small_blobs(lung_mask, params.min_rel_blob_area)
    for i, catheter in enumerate(CATHETHERS):
        catheter_mask[..., i] = filter_small_blobs(catheter_mask[..., i], params.min_rel_blob_area)

    # Filter masks that are not in classification result:
    for i, catheter in enumerate(CATHETHERS):
        if catheter not in classification_result:
            catheter_mask[..., i] = 0

    # Filter classification results that are not in catheter masks
    classification_result = {
        cat: res for cat, res in classification_result.items() if catheter_mask[..., CATHETHERS.index(cat)].sum() > 0
    }

    # Prepare image
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
    st.image(result_image, caption='Chest X-Ray Scan Processing Result', use_column_width=True)

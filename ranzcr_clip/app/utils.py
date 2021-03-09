import os
import os.path as osp
from argparse import ArgumentParser

import cv2 as cv
import numpy as np
import streamlit as st


@st.cache
def get_arguments():
    """Return the values of CLI params"""
    parser = ArgumentParser()
    parser.add_argument('--images_dir', default='artifacts/images')
    parser.add_argument('--models_dir', default='../models')
    args = parser.parse_args()
    return args.images_dir, args.models_dir


@st.cache
def load_image(image_name: str, path_to_folder: str) -> np.ndarray:
    """Load the image in grayscale
    Args:
        image_name (str): name of the image
        path_to_folder (str): path to the folder with image
    """
    image_file = osp.join(path_to_folder, image_name)
    image = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
    return image


def upload_image() -> np.ndarray:
    """Upload the image in grayscale"""
    file = st.sidebar.file_uploader('Upload your image (jpg, jpeg, or png)', ['jpg', 'jpeg', 'png'])
    image = cv.imdecode(np.fromstring(file.read(), np.uint8), 1)
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    return image


@st.cache
def get_images_list(path_to_folder: str) -> list:
    """Return the list of images from folder
    Args:
        path_to_folder (str): absolute or relative path to the folder with images
    """
    image_names_list = [x for x in os.listdir(path_to_folder) if x[-3:] in ['jpg', 'jpeg', 'png']]
    return image_names_list

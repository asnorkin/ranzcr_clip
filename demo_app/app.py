import os.path as osp

import streamlit as st

from models import classification, segmentation
from utils import get_arguments
from visuals import (
    draw_result_ui,
    explanation,
    instructions,
    select_image,
    select_interface,
    select_model_params,
    ImageStatus,
)


def main() -> None:
    path_to_images, models_dir = get_arguments()

    st.title('RANZCR CLiP Demo')

    # Check images dir
    if not osp.isdir(path_to_images):
        st.title(f'There is not images directory: {path_to_images}')
        return

    # Select interface type
    interface_type = select_interface()

    # Select image
    status, image = select_image(path_to_images)
    if not status.is_ok():
        if status == ImageStatus.NOT_SELECTED:
            instructions()
        else:
            st.header(status.hint())

        st.markdown('Source: https://github.com/asnorkin/ranzcr_clip')
        return

    # Select model_params
    params = select_model_params(interface_type)

    # Show explanation
    explanation()

    # Predict
    classification_result = classification(image, models_dir, params.classification_threshold)
    catheter_mask, lung_mask = segmentation(
        image, models_dir, params.catheter_mask_threshold, params.lung_mask_threshold
    )

    # Draw result
    draw_result_ui(image, classification_result, catheter_mask, lung_mask, params)

    st.markdown('Source: https://github.com/asnorkin/ranzcr_clip')


if __name__ == '__main__':
    main()

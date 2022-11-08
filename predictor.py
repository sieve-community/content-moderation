from typing import List
from sieve.types import FrameFetcher, SingleObject, BoundingBox, Temporal
from sieve.predictors import TemporalPredictor
from sieve.types.outputs import Detection
import tensorflow as tf
from sieve.types.constants import FRAME_NUMBER, BOUNDING_BOX, CLASS, TEMPORAL
import requests
import zipfile
from private_detector.utils.preprocess import preprocess_for_evaluation
import os 
import cv2 

class ContentModerator(TemporalPredictor):

    def setup(self):
        model_zip_url = 'https://storage.googleapis.com/private_detector/private_detector.zip'
        model_zip_path = 'private_detector.zip'
        save_path = 'private_detector_artifacts'
        model_path = f'{save_path}/private_detector/saved_model'
        # Download model
        r = requests.get(model_zip_url, stream=True)
        with open(model_zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        # Unzip model
        with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
            zip_ref.extractall(save_path)
        os.remove(model_zip_path)

        # Load model
        self.model = tf.saved_model.load(model_path)
    
    def predict(self, frame: FrameFetcher) -> SingleObject:
        # Get the frame array
        frame_data = frame.get_frame()
        # Convert to RGB
        frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
        height = frame_data.shape[0]
        width = frame_data.shape[1]
        frame_number = frame.get_current_frame_number()

        # Preprocess image
        image = tf.convert_to_tensor(frame_data)
        image = preprocess_for_evaluation(
            image,
            480,
            tf.float16
        )
        image = tf.reshape(image, -1)

        # Run the model
        preds = self.model([image])

        prob = tf.get_static_value(preds[0])[0]

        init_dict = {
            CLASS: "frame",
            TEMPORAL: Temporal(**{
                FRAME_NUMBER: frame_number,
                BOUNDING_BOX: BoundingBox.from_array([0, 0, width, height]),
                "lewdness_score": float(prob)
            })
        }

        output_object = SingleObject(**init_dict)

        return output_object
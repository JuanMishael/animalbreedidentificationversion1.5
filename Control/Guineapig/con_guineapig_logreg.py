import cv2
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf
import os

class gpLogReg:
    def __init__(self,url) -> None:
        self.image = url
    
    def predict_image(self):
        # Load the model
        load_extractor = tf.keras.models.load_model("././Model/Guineapig/resnetLogreg/resnet_EXTRACTOR.h5")

        modelpath = "././Model/Guineapig/resnetLogreg/dataSaved.pkl"

        with open(modelpath, 'rb') as file:
            saved_data = pickle.load(file)
            animal_breed = saved_data['class_name']
            model = saved_data['logreg_model']

        im = Image.open(self.image)
        img = im.convert("RGB")
        img= np.asarray(img)
        image_resized= cv2.resize(img, (224,224))
        features = load_extractor.predict(np.expand_dims(image_resized, axis=0))
        
        reshaped_features = features.reshape(features.shape[0],-1)
        predicted_class = model.predict(reshaped_features)
        pred_prob = model.predict_proba(reshaped_features)[:2]
        prediction_probability = pred_prob[0][predicted_class[0]]
        predicted_class

        output_class= animal_breed[predicted_class[0]]

        return [output_class, prediction_probability]

import cv2
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import decode_predictions
import io


class gpResNet:
    def __init__(self,url) -> None:
        self.image = url

    def predict_image(self):
        # Load the model
        loaded_model = tf.keras.models.load_model("././Model/Guineapig/resnet50/ResnetSavemodel.h5")

        picklefilepath = "././Model/Guineapig/resnet50/dataSaved.pkl"

        with open(picklefilepath, 'rb') as file:
            saved_data = pickle.load(file)
            self.animal_breed = saved_data['class_name']


        im = Image.open(self.image)
        img = im.convert("RGB")
        img= np.asarray(img)
        image_resized= cv2.resize(img, (256,256))
        image=np.expand_dims(image_resized,axis=0)
        print(image.shape)

        pred=loaded_model.predict(image)

        pred_proba = self.custom_decode_predictions(pred,top=1)
        rate = 0
        for _,confidence in pred_proba:
            rate = confidence

        output_class= self.animal_breed[np.argmax(pred)]

        return [output_class, rate]
    
    def custom_decode_predictions(self, prediction, top=3):
        # convert 2d Array of shape (1,10) to a 1D array of shape (10,)
        prediction = np.squeeze(prediction)

        #get the indices of the top "top" prediction
        top_indices = prediction.argsort()[-top:][::-1]

        class_labels = self.animal_breed

        top_prediction = [(class_labels[i], prediction[i]) for i in top_indices]

        return top_prediction

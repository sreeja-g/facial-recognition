from kivy.app import App

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics.texture import Texture

from kivy.clock import Clock
from kivy.logger import Logger

import os
import cv2
import tensorflow as tf
from layers import L1Dist
import numpy as np

class CamApp(App):

    def build(self):

        self.web_cam = Image(size_hint=(1,0.8))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,0.1))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,0.1))
        
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.verification_label)
        layout.add_widget(self.button)

        self.model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist':L1Dist})

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout
    
    def update(self, *args):
        
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]

        cam_input = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(cam_input, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def preprocess(self, file_path):
        
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        
        img = tf.image.resize(img, (100,100))
        img = img / 255.0

        return img

    def verify(self, *args):

        detection_threshold = 0.8
        verification_threshold = 0.85

        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)
        
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(SAVE_PATH)
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
            
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
        
        detection = np.sum(np.array(results) > detection_threshold)
        
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
        verified = verification > verification_threshold

        if verified:
            self.verification_label.text = 'Verified'
            self.verification_label.color = 'green'
        else:
            self.verification_label.text = 'Unverified'
            self.verification_label.color = 'red'
        
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(np.sum(np.array(results) > 0.7))
        Logger.info(np.sum(np.array(results) > 0.8))
        Logger.info(np.sum(np.array(results) > 0.9))
        Logger.info(np.sum(np.array(results) > 0.99))

        return results, verified

if __name__ == '__main__':
    CamApp().run()
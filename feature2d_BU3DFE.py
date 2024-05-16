import torch
from glob import glob
import os
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
from sklearn.preprocessing import normalize

basic_model = keras.models.load_model('GhostFaceNets/checkpoints/GN_W0.5_S2_ArcFace_epoch16.h5')

def create_2d_feature(input_name, output_name):
    # Load the face image
    face_image = cv2.imread(input_name)
    
    # Downscale
    downscaled_face = cv2.resize(face_image, (112,112), interpolation=cv2.INTER_CUBIC)
    
    # Save downscaled face image
    #cv2.imwrite('cropped_and_downscaled_face.jpg', downscaled_face)
    
    # Normalize values
    downscaled_face = (tf.cast(downscaled_face, "float32") - 127.5) * 0.0078125
    
    # Expand dimention
    downscaled_face = np.expand_dims(downscaled_face, axis=0)

    # Create feature embedding
    feature_embedding = basic_model(downscaled_face)
    
    torch.save(feature_embedding.numpy()[0], output_name)
    #print(f"Saved {output_name}\n")
    
input_folder = "/cluster/home/emmalei/Master_project_network/BU3DFE"
output_folder = "/cluster/home/emmalei/Master_project_network/feature_vectors/test/2d"

id_list = sorted(glob(os.path.join(input_folder, "*", "")))

expressions = {"AN":"ANG", "DI":"DIS", "FE":"FEA", "HA":"HAP", "NE":"NEU", "SA":"SAD", "SU":"SUR"}

for id in id_list:
    name_id = id[-6]+id[-3:-1]

    faces_2d = sorted(glob(os.path.join(input_folder, id, '*F2D.bmp')))
    for face_path in faces_2d: 
        input_name = face_path 
        output_name = f"{output_folder}/feat2d_{name_id}{expressions[face_path[-14:-12]]}_{face_path[-11]}.pt"
        
        create_2d_feature(input_name, output_name)
        
        print(f"In: {input_name}\nOut: {output_name}\n")
        
# Naming covention: feat2d_F01ANG_1.pt
# F01 means female identity number 01
# ANG is the current expression
# 1 is the current intensity of the frame (goes from 1 to 4)
      
    
            
        
    
    
    
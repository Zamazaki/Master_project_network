import torch
from glob import glob
import os
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2

basic_model = keras.models.load_model('GhostFaceNets/checkpoints/GN_W0.5_S2_ArcFace_epoch16.h5')
      
def create_2d_feature(input_name, output_name, save_sample):
    # Crop away non-face part of image
    cropped_face = cv2.imread(input_name)[252:1292, :]
    
    if save_sample:
        cv2.imwrite(f'/cluster/home/emmalei/Master_project_network/cropped_samples/{input_name.split("/")[6]}.jpg', cropped_face)
    
    # Downscale
    downscaled_face = cv2.resize(cropped_face, (112,112), interpolation=cv2.INTER_AREA)
    #cv2.imwrite('cropped_and_downscaled_face.jpg', downscaled_face)
    
    # Normalize values
    downscaled_face = (tf.cast(downscaled_face, "float32") - 127.5) * 0.0078125
    
    # Expand dimention
    downscaled_face = np.expand_dims(downscaled_face, axis=0)

    # Create feature embedding
    feature_embedding = basic_model(downscaled_face)[0]
    
    torch.save(feature_embedding.numpy(), output_name)
    #print(f"Saved {output_name}\n")
    
input_folder = "/cluster/home/emmalei/Master_project_network/BU4DFE"
output_folder = "/cluster/home/emmalei/Master_project_network/feature_vectors/train/2d"

id_list = sorted(glob(os.path.join(input_folder, "*", ""))) #/cluster/home/emmalei/Master_project_network/BU4DFE/M034/
id_list.remove("/cluster/home/emmalei/Master_project_network/BU4DFE/BU_WrlViewer_V2/")

for id in id_list:
    name_id = id[-5]+id[-3:-1]
    expression_folder = sorted(os.listdir(id))
    for expression in expression_folder:
        name_expression = expression[:3].upper()
        for i in range(5, 26):
            input_name = os.path.join(input_folder, id, expression, f'{i:03d}.jpg')
            output_name = f"{output_folder}/feat2d_{name_id}{name_expression}_{i-5}.pt"
            
            if expression == "Angry" and i == 5:
                create_2d_feature(input_name, output_name, True)
            else:
                create_2d_feature(input_name, output_name, False)
            
            print(f"In: {input_name}\nOut: {output_name}\n")
            
        
    
    
    
import torch
from glob import glob
import os
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
from sklearn.preprocessing import normalize

basic_model = keras.models.load_model('GhostFaceNets/checkpoints/GN_W0.5_S2_ArcFace_epoch16.h5')
      
def create_2d_feature(input_name, output_name, save_sample):
    # Crop away non-face part of image
    cropped_face = cv2.imread(input_name)[252:1292, :]
    
    if save_sample:
        cv2.imwrite(f'/cluster/home/emmalei/Master_project_network/cropped_samples/{input_name.split("/")[6]}.jpg', cropped_face)
    
    # Downscale
    downscaled_face = cv2.resize(cropped_face, (112,112), interpolation=cv2.INTER_CUBIC)
    
    # Save downscaled face image
    #cv2.imwrite('cropped_and_downscaled_face.jpg', downscaled_face)
    
    # Prepare image for the model (See:GhostFaceNets/eval_folder.py line 25)
    downscaled_face = (tf.cast(downscaled_face, "float32") - 127.5) * 0.0078125
    
    # Expand dimention
    downscaled_face = np.expand_dims(downscaled_face, axis=0)

    # Create feature embedding
    feature_embedding = basic_model(downscaled_face)
    
    # Normalize and save feature vector
    torch.save(normalize(feature_embedding.numpy())[0], output_name)
    
input_folder = "/cluster/home/emmalei/Master_project_network/BU4DFE"
output_folder = "/cluster/home/emmalei/Master_project_network/feature_vectors/train/2d-normalized"

id_list = sorted(glob(os.path.join(input_folder, "*", "")))
id_list.remove("/cluster/home/emmalei/Master_project_network/BU4DFE/BU_WrlViewer_V2/")

# Uncomment to get validation set
#chosen_frames = [3, 4, 26, 27]

for id in id_list:
    name_id = id[-5]+id[-3:-1]
    expression_folder = sorted(os.listdir(id))
    for expression in expression_folder:
        name_expression = expression[:3].upper()
        for i in range(5, 26): #range(1, 5) for validation
            #input_name = os.path.join(input_folder, id, expression, f'{chosen_frames[i-1]:03d}.jpg') for validation
            input_name = os.path.join(input_folder, id, expression, f'{i:03d}.jpg') #training
            #output_name = f"{output_folder}/feat2d_{name_id}{name_expression}_{i}.pt" for validation
            output_name = f"{output_folder}/feat2d_{name_id}{name_expression}_{i-5}.pt" #training
            
            # Save cropped version of fifth frame of every identity
            #if expression == "Angry" and i == 5:
            #    create_2d_feature(input_name, output_name, True)
            #else:
            create_2d_feature(input_name, output_name, False)
            
            print(f"In: {input_name}\nOut: {output_name}\n")

# Naming covention: feat2d_F01ANG_1.pt
# F01 means female identity number 01
# ANG is the current expression
# 1 is the current intensity of the frame (goes from 1 to 20,
# where intensity 1 maps to frame number 5 in each expression)
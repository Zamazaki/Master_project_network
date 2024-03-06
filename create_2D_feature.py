from tensorflow import keras
import tensorflow as tf
import torch
import numpy as np
import cv2
import os

# Load pretrained weighs and make the model work on a downscaled pictures
# Pretrained weights were downloaded from model GhostFaceNetV1-0.5-2 (A) https://github.com/HamadYA/GhostFaceNets/tree/main

mode = "train"
image_folder = "jpg_face_batch"

image_paths = os.listdir(image_folder)
feature_counter = 0 # To name the created feature vectors. Update after running a batch so we don't overwrite the next batch

basic_model = keras.models.load_model('GhostFaceNets/checkpoints/GN_W0.5_S2_ArcFace_epoch16.h5')

for image_path in image_paths:
    # Downscale
    downscaled_face = cv2.resize(cv2.imread(f"./{image_folder}/{image_path}"), (112,112), interpolation=cv2.INTER_AREA)

    # Normalize values
    downscaled_face = (tf.cast(downscaled_face, "float32") - 127.5) * 0.0078125
    
    # Expand dimention
    downscaled_face = np.expand_dims(downscaled_face, axis=0)

    # Create feature embedding
    feature_embedding = basic_model(downscaled_face)[0]
    
    torch.save(feature_embedding.numpy(), f"feature_vectors/{mode}/2d/feat2d_{feature_counter}.pt")
    print(f"Saved feature_vectors/{mode}/2d/feat2d_{feature_counter}.pt")
    feature_counter += 1

print(f"Images converted, feature counter is now {feature_counter}")



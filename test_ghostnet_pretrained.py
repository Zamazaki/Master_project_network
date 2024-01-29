#import GhostFaceNets.evals
from tensorflow import keras
#import tensorflow as tf
#import GhostFaceNets.IJB_evals
#import matplotlib.pyplot as plt
#import keras_cv_attention_models
import GhostFaceNets.GhostFaceNets, GhostFaceNets.GhostFaceNets_with_Bias
import numpy as np
import cv2

"""We load pretrained weighs and make the model work on a downscaled picture"""

image_path = "./test_data/F0001_AN01WH_F2D.bmp" #"./BU_3DFE/F0001/F0001_AN01WH_F2D.bmp"

basic_model = keras.models.load_model('GhostFaceNets/checkpoints/GN_W0.5_S2_ArcFace_epoch16.h5', compile=False)

downscaled_face = cv2.resize(cv2.imread(image_path), (112,112), interpolation=cv2.INTER_AREA)
#cv2.imwrite("scaled_down_face.jpg", resized_face)

feature_embedding = basic_model(np.expand_dims(downscaled_face, axis=0)) #cv2.imread(image_path)

print(f"Dimensions GhostNet feature vector {feature_embedding.shape}") 

# Note: Since the pretrained networks are already trained to give embeddings of a certain size, 
# I'm not sure if I can change them. That might be a problem...
from tensorflow import keras
import tensorflow as tf
import GhostFaceNets.GhostFaceNets_with_Bias as gfb
import GhostFaceNets.losses
import numpy as np
import cv2

"""We load pretrained weighs and make the model work on a downscaled picture"""

image_path = "./test_data/F0001_AN01WH_F2D.bmp" #"./BU_3DFE/F0001/F0001_AN01WH_F2D.bmp"

basic_model = keras.models.load_model('GhostFaceNets/checkpoints/GN_W0.5_S2_ArcFace_epoch16.h5')

downscaled_face = cv2.resize(cv2.imread(image_path), (112,112), interpolation=cv2.INTER_AREA)
#cv2.imwrite("scaled_down_face.jpg", resized_face)
#basic_model.summary()

downscaled_face = (tf.cast(downscaled_face, "float32") - 127.5) * 0.0078125 # Normalize values
downscaled_face = np.expand_dims(downscaled_face, axis=0)

feature_embedding = basic_model(downscaled_face) #cv2.imread(image_path)

np.savetxt("feature_vectors/2d/feat2d_1.csv", feature_embedding, delimiter=",")

#print(f"Dimensions GhostNet feature vector {feature_embedding.shape}") 

#print("Whole tensor:")
#np.set_printoptions(threshold=2000)
#print(feature_embedding)
#np.set_printoptions(threshold=1000)


import numpy as np
import tensorflow as tf
import numpy as np
import ldgcnn.models.ldgcnn
from utils.read_wrl import read_wrl
import os
import torch

# Some CUDA stuff seems to prevent this from working. I'll try to load the checkpoints in TF2 instead
# Never mind, seems to work now
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Create placeholders for the model to know what values to expect (input size in particular)
is_training_pl = tf.placeholder(tf.bool, shape=())
pointclouds_pl, labels_pl = ldgcnn.models.ldgcnn.placeholder_inputs(1, 7968)

# Gives out tensor of size (N, 3) where N is amount of points
point_cloud = torch.tensor(read_wrl("./test_data/F0001_AN01WH_F3D.wrl")) #read_wrl("./BU_3DFE/F0001/F0001_AN01WH_F3D.wrl") 
print(f"Dimensions of point cloud {point_cloud.shape}")

# Input point cloud needs to be size (B, N, 3) where B is batch size. The model also wants it as a ndarray
point_cloud = np.expand_dims(point_cloud.numpy(), 0)
print(f"New dimensions of point cloud {point_cloud.shape}")

# Create a session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = True


with tf.Session(config=config) as sess:
    with tf.device('/gpu:'+str(1)):
        # Getting feature extraction model ready for evaluation   
        model = ldgcnn.models.ldgcnn.calc_ldgcnn_feature(pointclouds_pl, is_training_pl)
        
        # Loading pretrained weights
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, "./ldgcnn/log/ldgcnn_model.ckpt")

        # Setting up the actual input arguments to be fed to the feature extractor 
        feed_dict_cnn = {pointclouds_pl: point_cloud, is_training_pl: False}

        # Getting the feature vector from the feature extractor
        global_feature = np.squeeze(model.eval(feed_dict=feed_dict_cnn))

        #np.savetxt("feature_vectors/3d/feat3d_1.csv", global_feature, delimiter=",")
        #torch.save(torch.from_numpy(global_feature), "feature_vectors/3d/feat3d_1.pt")
        torch.save(global_feature, "feature_vectors/3d/feat3d_1.pt")
import numpy as np
import tensorflow as tf
import numpy as np
import ldgcnn.models.ldgcnn
from utils.read_obj import read_obj
import os
import torch

# All .obj files in the dataset used have the same amount of vertices, so we do a little cheat >:^)

# Prefers to have gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Create placeholders for the model to know what values to expect (input size in particular)
is_training_pl = tf.placeholder(tf.bool, shape=())
pointclouds_pl, labels_pl = ldgcnn.models.ldgcnn.placeholder_inputs(1, 5904)

# Set up for batch creation of feature vectors
point_cloud_paths = os.listdir("obj_face_batch")
feature_counter = 0 # To name the created feature vectors. Update after running a batch so we don't overwrite the next batch

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
        
        for path in point_cloud_paths:
            # Gives out tensor of size (N, 3) where N is amount of points
            point_cloud = torch.tensor(read_obj("./obj_face_batch/"+path)) 
            
            # Input point cloud needs to be size (B, N, 3) where B is batch size. The model also wants it as a ndarray
            point_cloud = np.expand_dims(point_cloud.numpy(), 0)

            # Setting up the actual input arguments to be fed to the feature extractor 
            feed_dict_cnn = {pointclouds_pl: point_cloud, is_training_pl: False}

            # Getting the feature vector from the feature extractor
            global_feature = np.squeeze(model.eval(feed_dict=feed_dict_cnn))

            # Save feature vector
            torch.save(global_feature, f"feature_vectors/3d/feat3d_{feature_counter}.pt")
            print(f"Converted feature_vectors/3d/feat3d_{feature_counter}.pt")
            feature_counter += 1
print(f"3D files converted, feature counter is now {feature_counter}")

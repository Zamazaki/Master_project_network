import tensorflow as tf
import numpy as np
import ldgcnn.models.ldgcnn
from utils.read_wrl import read_wrl
import os
import torch

# Some CUDA stuff seems to prevent this from working. I'll try to load the checkpoints in TF2 instead
# Never mind, seems to work now
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

pointclouds_pl, labels_pl = ldgcnn.models.ldgcnn.placeholder_inputs(1, 7968)
_, layers = ldgcnn.models.ldgcnn.get_model(pointclouds_pl, tf.cast(False, tf.bool))

is_training_pl = tf.placeholder(tf.bool, shape=())

# Gives out tensor of size (N, 3) where N is amount of points
point_cloud = torch.tensor(read_wrl("./test_data/F0001_AN01WH_F3D.wrl")) #read_wrl("./BU_3DFE/F0001/F0001_AN01WH_F3D.wrl") 
print(f"Dimensions of point cloud {point_cloud.shape}")

#Convert from torch tensor to tensorflow tensor
#point_cloud = tf.convert_to_tensor(point_cloud)

point_cloud = np.expand_dims(point_cloud.numpy(), 0)

# Input point cloud needs to be size (B, N, 3) where B is batch size
#point_cloud = tf.expand_dims(point_cloud, 0)
#point_cloud = tf.make_ndarray(tf.make_tensor_proto(point_cloud))#.eval(session=tf.Session()) # Add extra dimension so it fits the expected input size
print(f"New dimensions of point cloud {point_cloud.shape}")

# Create a session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = True

#session = tf.Session()
with tf.Session(config=config) as sess:
    variable_names = [v.name for v in tf.global_variables()]
    variables = tf.global_variables()
    
    # Variables before #43 belong to the feature extractor.
    saver = tf.train.Saver(variables[0:44])
    saver.restore(sess, "./ldgcnn/log/ldgcnn_model.ckpt")

    feed_dict_cnn = {pointclouds_pl: point_cloud, is_training_pl: False}

    print(type(point_cloud))
    global_feature = np.squeeze(layers['global_feature'].eval(feed_dict=feed_dict_cnn))

    print(f"Dimensions of the global feature vector: {global_feature.shape}")

#global_feature = calc_ldgcnn_feature(point_cloud, tf.cast(False, tf.bool), None)
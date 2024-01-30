import tensorflow as tf
import numpy as np
import ldgcnn.models.ldgcnn
from utils.read_wrl import read_wrl
import os
import torch

# Some CUDA stuff seems to prevent this from working. I'll try to load the checkpoints in TF2 instead
# Never mind, seems to work now
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

is_training_pl = tf.placeholder(tf.bool, shape=())
pointclouds_pl, labels_pl = ldgcnn.models.ldgcnn.placeholder_inputs(1, 7968)
#_, layers = ldgcnn.models.ldgcnn.get_model(pointclouds_pl, tf.cast(False, tf.bool))

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
    with tf.device('/gpu:'+str(1)):
        #variable_names = [v.name for v in tf.global_variables()]
        #print("-----------------------------------------------------------------------------")
        #print(variable_names)
        #variables = tf.global_variables()
        #variables = ['fc3/weights/Adam', 'fc3/weights', 'fc2/weights/Adam_1', 'fc2/bn/gamma/Adam', 'fc3/biases', 'fc2/weights/Adam', 'fc2/bn/gamma', 'fc2/bn/fc2/bn/moments/Squeeze_1/ExponentialMovingAverage', 'fc2/bn/fc2/bn/moments/Squeeze/ExponentialMovingAverage', 'fc2/biases/Adam', 'fc2/biases', 'fc1/weights/Adam_1', 'fc1/weights/Adam', 'fc1/weights', 'fc1/bn/gamma/Adam_1', 'fc1/bn/gamma', 'fc1/bn/fc1/bn/moments/Squeeze_1/ExponentialMovingAverage', 'fc1/bn/fc1/bn/moments/Squeeze/ExponentialMovingAverage', 'fc1/bn/beta/Adam_1', 'fc1/biases/Adam_1', 'fc1/biases/Adam', 'fc3/weights/Adam_1', 'fc1/biases', 'dgcnn4/weights/Adam', 'dgcnn4/bn/gamma/Adam_1', 'dgcnn4/bn/gamma', 'dgcnn4/bn/dgcnn4/bn/moments/Squeeze_1/ExponentialMovingAverage', 'dgcnn4/bn/dgcnn4/bn/moments/Squeeze/ExponentialMovingAverage', 'dgcnn4/bn/beta/Adam_1', 'fc2/bn/beta/Adam_1', 'dgcnn4/bn/beta', 'dgcnn4/biases/Adam_1', 'dgcnn4/biases/Adam', 'fc2/bn/gamma/Adam_1', 'dgcnn4/biases', 'dgcnn1/weights', 'agg/bn/beta', 'dgcnn4/bn/gamma/Adam', 'dgcnn2/bn/beta/Adam', 'dgcnn2/bn/beta/Adam_1', 'dgcnn1/bn/gamma/Adam_1', 'dgcnn1/bn/gamma/Adam', 'dgcnn1/bn/gamma', 'fc2/weights', 'agg/bn/gamma', 'dgcnn2/biases/Adam_1', 'fc2/biases/Adam_1', 'dgcnn3/bn/beta/Adam', 'fc1/bn/beta/Adam', 'dgcnn1/bn/dgcnn1/bn/moments/Squeeze/ExponentialMovingAverage', 'fc2/bn/beta', 'dgcnn1/bn/beta/Adam', 'dgcnn2/bn/dgcnn2/bn/moments/Squeeze/ExponentialMovingAverage', 'dgcnn1/bn/beta', 'agg/weights/Adam_1', 'dgcnn1/bn/dgcnn1/bn/moments/Squeeze_1/ExponentialMovingAverage', 'dgcnn4/weights', 'dgcnn3/biases/Adam_1', 'dgcnn1/biases/Adam', 'beta1_power', 'dgcnn2/weights/Adam_1', 'fc3/biases/Adam_1', 'dgcnn4/weights/Adam_1', 'dgcnn1/biases', 'beta2_power', 'dgcnn1/biases/Adam_1', 'agg/biases', 'agg/bn/agg/bn/moments/Squeeze/ExponentialMovingAverage', 'fc1/bn/beta', 'agg/bn/agg/bn/moments/Squeeze_1/ExponentialMovingAverage', 'dgcnn3/bn/gamma', 'dgcnn3/bn/beta', 'dgcnn3/bn/dgcnn3/bn/moments/Squeeze_1/ExponentialMovingAverage', 'agg/weights', 'agg/weights/Adam', 'agg/biases/Adam', 'dgcnn1/bn/beta/Adam_1', 'agg/biases/Adam_1', 'dgcnn3/biases/Adam', 'Variable', 'fc3/biases/Adam', 'fc1/bn/gamma/Adam', 'dgcnn2/bn/gamma/Adam_1', 'agg/bn/beta/Adam', 'agg/bn/gamma/Adam_1', 'dgcnn2/biases', 'dgcnn2/biases/Adam', 'dgcnn3/bn/gamma/Adam', 'agg/bn/gamma/Adam', 'dgcnn1/weights/Adam', 'dgcnn2/bn/beta', 'dgcnn2/bn/dgcnn2/bn/moments/Squeeze_1/ExponentialMovingAverage', 'dgcnn2/bn/gamma', 'dgcnn2/bn/gamma/Adam', 'dgcnn2/weights', 'dgcnn2/weights/Adam', 'dgcnn4/bn/beta/Adam', 'dgcnn1/weights/Adam_1', 'dgcnn3/biases', 'agg/bn/beta/Adam_1', 'dgcnn3/bn/beta/Adam_1', 'fc2/bn/beta/Adam', 'dgcnn3/bn/dgcnn3/bn/moments/Squeeze/ExponentialMovingAverage', 'dgcnn3/bn/gamma/Adam_1', 'dgcnn3/weights', 'dgcnn3/weights/Adam', 'dgcnn3/weights/Adam_1']
        #variables = ['fc3/weights/Adam:0', 'fc3/weights:0', 'fc2/weights/Adam_1:0', 'fc2/bn/gamma/Adam:0', 'fc3/biases:0', 'fc2/weights/Adam:0', 'fc2/bn/gamma:0', 'fc2/bn/fc2/bn/moments/Squeeze_1/ExponentialMovingAverage:0', 'fc2/bn/fc2/bn/moments/Squeeze/ExponentialMovingAverage:0', 'fc2/biases/Adam:0', 'fc2/biases:0', 'fc1/weights/Adam_1:0', 'fc1/weights/Adam:0', 'fc1/weights:0', 'fc1/bn/gamma/Adam_1:0', 'fc1/bn/gamma:0', 'fc1/bn/fc1/bn/moments/Squeeze_1/ExponentialMovingAverage:0', 'fc1/bn/fc1/bn/moments/Squeeze/ExponentialMovingAverage:0', 'fc1/bn/beta/Adam_1:0', 'fc1/biases/Adam_1:0', 'fc1/biases/Adam:0', 'fc3/weights/Adam_1:0', 'fc1/biases:0', 'dgcnn4/weights/Adam:0', 'dgcnn4/bn/gamma/Adam_1:0', 'dgcnn4/bn/gamma:0', 'dgcnn4/bn/dgcnn4/bn/moments/Squeeze_1/ExponentialMovingAverage:0', 'dgcnn4/bn/dgcnn4/bn/moments/Squeeze/ExponentialMovingAverage:0', 'dgcnn4/bn/beta/Adam_1:0', 'fc2/bn/beta/Adam_1:0', 'dgcnn4/bn/beta:0', 'dgcnn4/biases/Adam_1:0', 'dgcnn4/biases/Adam:0', 'fc2/bn/gamma/Adam_1:0', 'dgcnn4/biases:0', 'dgcnn1/weights:0', 'agg/bn/beta:0', 'dgcnn4/bn/gamma/Adam:0', 'dgcnn2/bn/beta/Adam:0', 'dgcnn2/bn/beta/Adam_1:0', 'dgcnn1/bn/gamma/Adam_1:0', 'dgcnn1/bn/gamma/Adam:0', 'dgcnn1/bn/gamma:0', 'fc2/weights:0', 'agg/bn/gamma:0', 'dgcnn2/biases/Adam_1:0', 'fc2/biases/Adam_1:0', 'dgcnn3/bn/beta/Adam:0', 'fc1/bn/beta/Adam:0', 'dgcnn1/bn/dgcnn1/bn/moments/Squeeze/ExponentialMovingAverage:0', 'fc2/bn/beta:0', 'dgcnn1/bn/beta/Adam:0', 'dgcnn2/bn/dgcnn2/bn/moments/Squeeze/ExponentialMovingAverage:0', 'dgcnn1/bn/beta:0', 'agg/weights/Adam_1:0', 'dgcnn1/bn/dgcnn1/bn/moments/Squeeze_1/ExponentialMovingAverage:0', 'dgcnn4/weights:0', 'dgcnn3/biases/Adam_1:0', 'dgcnn1/biases/Adam:0', 'beta1_power:0', 'dgcnn2/weights/Adam_1:0', 'fc3/biases/Adam_1:0', 'dgcnn4/weights/Adam_1:0', 'dgcnn1/biases:0', 'beta2_power:0', 'dgcnn1/biases/Adam_1:0', 'agg/biases:0', 'agg/bn/agg/bn/moments/Squeeze/ExponentialMovingAverage:0', 'fc1/bn/beta:0', 'agg/bn/agg/bn/moments/Squeeze_1/ExponentialMovingAverage:0', 'dgcnn3/bn/gamma:0', 'dgcnn3/bn/beta:0', 'dgcnn3/bn/dgcnn3/bn/moments/Squeeze_1/ExponentialMovingAverage:0', 'agg/weights:0', 'agg/weights/Adam:0', 'agg/biases/Adam:0', 'dgcnn1/bn/beta/Adam_1:0', 'agg/biases/Adam_1:0', 'dgcnn3/biases/Adam:0', 'Variable:0', 'fc3/biases/Adam:0', 'fc1/bn/gamma/Adam:0', 'dgcnn2/bn/gamma/Adam_1:0', 'agg/bn/beta/Adam:0', 'agg/bn/gamma/Adam_1:0', 'dgcnn2/biases:0', 'dgcnn2/biases/Adam:0', 'dgcnn3/bn/gamma/Adam:0', 'agg/bn/gamma/Adam:0', 'dgcnn1/weights/Adam:0', 'dgcnn2/bn/beta:0', 'dgcnn2/bn/dgcnn2/bn/moments/Squeeze_1/ExponentialMovingAverage:0', 'dgcnn2/bn/gamma:0', 'dgcnn2/bn/gamma/Adam:0', 'dgcnn2/weights:0', 'dgcnn2/weights/Adam:0', 'dgcnn4/bn/beta/Adam:0', 'dgcnn1/weights/Adam_1:0', 'dgcnn3/biases:0', 'agg/bn/beta/Adam_1:0', 'dgcnn3/bn/beta/Adam_1:0', 'fc2/bn/beta/Adam:0', 'dgcnn3/bn/dgcnn3/bn/moments/Squeeze/ExponentialMovingAverage:0', 'dgcnn3/bn/gamma/Adam_1:0', 'dgcnn3/weights:0', 'dgcnn3/weights/Adam:0', 'dgcnn3/weights/Adam_1:0']
        
        # Variables before #43 belong to the feature extractor.
        #saver = tf.compat.v1.train.Saver(variables[0:44])
        
        model = ldgcnn.models.ldgcnn.calc_ldgcnn_feature(pointclouds_pl, is_training_pl)
        saver = tf.compat.v1.train.Saver()
        
        saver.restore(sess, "./ldgcnn/log/ldgcnn_model.ckpt")

        feed_dict_cnn = {pointclouds_pl: point_cloud, is_training_pl: False}

        print(type(point_cloud))
        #global_feature = np.squeeze(layers['global_feature'].eval(feed_dict=feed_dict_cnn))
        global_feature = np.squeeze(model.eval(feed_dict=feed_dict_cnn))

        print(f"Dimensions of the global feature vector: {global_feature.shape}")
        print(f"Middle numbers of tensor: {global_feature[500]}, {global_feature[501]}, {global_feature[502]}")
        print(type(global_feature))
        print("Whole tensor:")
        np.set_printoptions(threshold=2000)
        print(global_feature)
        np.set_printoptions(threshold=1000)

#global_feature = calc_ldgcnn_feature(point_cloud, tf.cast(False, tf.bool), None)
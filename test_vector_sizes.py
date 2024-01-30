#from GroupFacePytorch.models.group_face import GroupFace
#from GroupFacePytorch.system.data_loader import torch_loader
from ldgcnn.models.ldgcnn import calc_ldgcnn_feature
from utils.read_wrl import read_wrl
#import torch
#import cv2
import tensorflow as tf

######## Global feature of GroupFace
#image_path = "./test_data/F0001_AN01WH_F2D.bmp" #"./BU_3DFE/F0001/F0001_AN01WH_F2D.bmp"

#model = GroupFace(resnet=50)
#group_inter, final, group_prob, group_label = model(torch_loader(cv2.imread(image_path)).unsqueeze(0))
#feat = final / torch.norm(final, p=2, keepdim=False)
#feat = feat.detach().cpu().reshape(1, 1024).numpy()

#print(f"Dimensions GroupFace global feature vector {feat.shape}")

######## Global feature of LDGCNN
# Gives out tensor of size (N, 3) where N is amount of points
point_cloud = read_wrl("./test_data/F0001_AN01WH_F3D.wrl") #read_wrl("./BU_3DFE/F0001/F0001_AN01WH_F3D.wrl") 
print(f"Dimensions of point cloud {point_cloud.shape}")

#Convert from torch tensor to tensorflow tensor
point_cloud = tf.convert_to_tensor(point_cloud)

# Input point cloud needs to be size (B, N, 3) where B is batch size
point_cloud = tf.expand_dims(point_cloud, 0) # Add extra dimension so it fits the expected input size
print(f"New dimensions of point cloud {point_cloud.shape}")

# Send point cloud through feature extraction
with tf.compat.v1.Session() as sess:
    global_feature = calc_ldgcnn_feature(point_cloud, tf.cast(False, tf.bool), None).eval()
    print(f"Dimensions LDGCNN global feature vector {global_feature.shape}")
    print(f"First numbers of tensor: {global_feature[0]}, {global_feature[1]}, {global_feature[2]}")

#print(feat)
#print(global_feature)
#tf.print(global_feature)
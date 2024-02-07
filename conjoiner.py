import torch.nn as nn

# Create layers that hopefully sends features into common space
class Conjoiner(nn.Module):

    def __init__(self):
        super(Conjoiner, self).__init__()
        self.resizingLayer = nn.Linear(1024, 512) # LDGCNN prodices 1024 dimention feature vectors, so need to resize
        self.linearLayer1 = nn.Linear(512, 512) # Might not need this. If simplification is needed, remove it
        self.linearLayer2 = nn.Linear(512, 256)

    def forward(self, point_cloud_feature, img_feature):
        # Sending point cloud feature into common space
        pc_feat = self.resizingLayer(point_cloud_feature)
        pc_feat = self.linearLayer1(pc_feat)
        pc_feat = self.linearLayer2(pc_feat)
        
        # Sending image feature into common space
        img_feat = self.linearLayer1(img_feature)
        img_feat = self.linearLayer2(img_feat)
        
        return img_feat, pc_feat
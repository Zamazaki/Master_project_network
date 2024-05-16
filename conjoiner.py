import torch.nn as nn

# Layers that sends features into common space
class Conjoiner(nn.Module):

    def __init__(self):
        super(Conjoiner, self).__init__()
        
        self.linearLayer1 = nn.Linear(512, 512)
        self.linearLayer2 = nn.Linear(512, 256)

    def forward(self, img_feature, point_cloud_feature):
        pc_feat = self.linearLayer1(point_cloud_feature)
        pc_feat = self.linearLayer2(pc_feat)
        
        img_feat = self.linearLayer1(img_feature)
        img_feat = self.linearLayer2(img_feat)
        
        return img_feat, pc_feat 
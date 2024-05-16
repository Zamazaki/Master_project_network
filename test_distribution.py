import torch
import torchvision
import matplotlib.pyplot as plt 
from conjoiner import Conjoiner
import torch.nn.functional as nnf
from dataset_loader import DatasetLoaderExtraLabel
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pytorch_metric_learning import distances

# Load the test dataset
test_dataset = DatasetLoaderExtraLabel(training_csv="feature_vectors/train/same_modality_feature_pairings.csv", training_dir="feature_vectors/train/")
test_dataloader = DataLoader(test_dataset, num_workers=6, batch_size=1, shuffle=False)

# Setup variables
model_name = "same_modality_pairings_FaceGen"#"Final_BU4DFE_MultiSimilarityLoss"
#MODEL_PATH = "checkpoints/Final_BU4DFE_MultiSimilarityLoss_model_3.pt"
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
similarity_function = distances.CosineSimilarity() #nn.CosineSimilarity(dim=1) #nn.PairwiseDistance() ##eucledian_distance = nnf.pairwise_distance(output1, output2)

# Load the network
"""model = Conjoiner()
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval() # Set model to evaluation mode"""

distance_same_identity = []
distance_different_identity = []

modality_2d_same_identity = []
modality_2d_diff_identity = []

modality_3d_same_identity = []
modality_3d_diff_identity = []

# Set up progress bar
pbar = tqdm(total=len(test_dataloader))

# Test the network
for i, data in enumerate(test_dataloader,0):
  x0, x1, dimention, label = data
  #output1, output2 = model(x0, x1)  #model(x0.to(device), x1.to(device))
  #embedding = model(x0, x1) 
  #similarity = similarity_function(embedding) #similarity_function(x0, x1)
  similarity = similarity_function(x0, x1)
  #print(similarity)
  
  if int(dimention) == 2:
    if int(label) == 1:
        modality_2d_same_identity.append(similarity.item())
    else:
        modality_2d_diff_identity.append(similarity.item())
        
  else:
    if int(label) == 1:
        modality_3d_same_identity.append(similarity.item())
    else:
        modality_3d_diff_identity.append(similarity.item())
  
  pbar.update(1)   
    
  #print("Predicted Distance:", similarity.item())
  #print("Actual Label:", print_label)
  #print("\n")
  
  #if len(distance_same_identity) == 12:
  #  break
  
#print(f"Mean same distance: {np.average(distance_same_identity)}\n Mean different distance: {np.average(distance_different_identity)}")
#print(f"Mean 2d distance: {np.average(modality_2d)}\n Mean 3d distance: {np.average(modality_3d)}")

pbar.close()

#results = {"Same_identity": distance_same_identity, "Different_identity": distance_different_identity}
print(f"Same identity 2d: {modality_2d_same_identity}")
print(f"Same identity 3d: {modality_3d_same_identity}")
# 2D
plt.figure(0)
results = {"2d same identity": modality_2d_same_identity, "2d different identity": modality_2d_diff_identity}
sns.kdeplot(results, fill = True, multiple="layer", common_grid=True, legend=True)

plt.title('Density Plot of Pairing cosine distances FaceGen')
plt.xlabel('Cosine distance')
plt.xlim(0, 1)
plt.savefig(f"test_data/test_plot_{model_name}_2D.png")

# 3D
plt.figure(1)
results = {"3d same identity": modality_3d_same_identity, "3d different identity": modality_3d_diff_identity}
sns.kdeplot(results, fill = True, multiple="layer", common_grid=True, legend=True)

plt.title('Density Plot of Pairing cosine distances FaceGen')
plt.xlabel('Cosine distance')
plt.xlim(0.9, 1)
plt.savefig(f"test_data/test_plot_{model_name}_3D.png")

plt.figure(2)
results = {"2d same identity": modality_2d_same_identity, "2d different identity": modality_2d_diff_identity, "3d same identity": modality_3d_same_identity, "3d different identity": modality_3d_diff_identity}
sns.kdeplot(results, fill = True, multiple="layer", common_grid=True, legend=True)

plt.title('Density Plot of Pairing cosine distances FaceGen')
plt.xlabel('Cosine distance')
plt.xlim(0, 1)
plt.savefig(f"test_data/test_plot_{model_name}_all.png")
import torch
import torchvision
import matplotlib.pyplot as plt 
from conjoiner import Conjoiner
import torch.nn.functional as nnf
from dataset_loader import DatasetLoader
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn

# Load the test dataset
test_dataset = DatasetLoader(training_csv="feature_vectors/validation/feature_pairings.csv", training_dir="feature_vectors/validation/")
test_dataloader = DataLoader(test_dataset, num_workers=6, batch_size=1, shuffle=True)

# Setup variables
MODEL_PATH = "checkpoints/cosine_loss7_model_10.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
similarity_function = nn.CosineSimilarity(dim=1) ##eucledian_distance = nnf.pairwise_distance(output1, output2)

# Load the network
model = Conjoiner()
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval() # Set model to evaluation mode

distance_same_identity = []
distance_different_identity = []

# Test the network
for i, data in enumerate(test_dataloader,0):
  x0, x1, label = data
  #concat = torch.cat((x0, x1), 0)
  output1, output2 = model(x0.to(device), x1.to(device))

  similarity = similarity_function(output1, output2)
  #print(similarity)
  if int(label) == 1:
    print_label="Same identity"
    distance_same_identity.append(similarity.item())
  else:
    print_label="Different identities"
    distance_different_identity.append(similarity.item())
    
  #plt.imshow(torchvision.utils.make_grid(concat))
  print("Predicted Distance:", similarity.item())
  print("Actual Label:", print_label)
  print("\n")
  if len(distance_same_identity) == 12:
    break
print(f"Mean same distance: {np.average(distance_same_identity)}\n Mean different distance: {np.average(distance_different_identity)}")
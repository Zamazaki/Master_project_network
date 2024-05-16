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
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pytorch_metric_learning import distances

# Load the test dataset
test_dataset = DatasetLoader(training_csv="feature_vectors/test/feature_pairings.csv", training_dir="feature_vectors/test/")
test_dataloader = DataLoader(test_dataset, num_workers=6, batch_size=1, shuffle=False)

# Setup variables
model_name = "Final_FaceGen_cosine_zero_loss_margin"#"Final_BU4DFE_MultiSimilarityLoss"
MODEL_PATH = f"checkpoints/{model_name}_model_6.pt"
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
similarity_function = nn.CosineSimilarity(dim=1) #nn.PairwiseDistance() #nn.CosineSimilarity(dim=1) #distances.CosineSimilarity() #nn.PairwiseDistance() #distances.CosineSimilarity() #nn.CosineSimilarity(dim=1) #nn.PairwiseDistance() ##eucledian_distance = nnf.pairwise_distance(output1, output2)

# Load the network
model = Conjoiner()
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval() # Set model to evaluation mode

distance_same_identity = []
distance_different_identity = []

modality_2d = []
modality_3d = []

thresholds = [-0.5, -0.25, 0.0, 0.25, 0.5] #[0.4, 0.5, 0.6, 0.7, 0.8]
acc0 = {"tp":0, "fp": 0, "tn":0, "fn":0}
acc1 = {"tp":0, "fp": 0, "tn":0, "fn":0}
acc2 = {"tp":0, "fp": 0, "tn":0, "fn":0}
acc3 = {"tp":0, "fp": 0, "tn":0, "fn":0}
acc4 = {"tp":0, "fp": 0, "tn":0, "fn":0}


# Set up progress bar
pbar = tqdm(total=len(test_dataloader))

# Test the network
for i, data in enumerate(test_dataloader,0):
  x0, x1, label = data
  output1, output2 = model(x0, x1)  #model(x0.to(device), x1.to(device))
  #embedding = model(x0, x1) 
  #similarity = similarity_function(embedding) #similarity_function(x0, x1)
  similarity = similarity_function(output1, output2)
  #print(similarity)
  if int(label) == 1:
    #print_label="Same identity"
    distance_same_identity.append(similarity.item())
  else:
    #print_label="Different identities"
    distance_different_identity.append(similarity.item())
    
  if similarity > thresholds[0]:
    if int(label) == 1:
      acc0["tp"] += 1
    else:
      acc0["fp"] += 1
  else:
    if int(label) == 1:
      acc0["fn"] += 1
    else:
      acc0["tn"] += 1
      
  if similarity > thresholds[1]:
    if int(label) == 1:
      acc1["tp"] += 1
    else:
      acc1["fp"] += 1
  else:
    if int(label) == 1:
      acc1["fn"] += 1
    else:
      acc1["tn"] += 1
      
  if similarity > thresholds[2]:
    if int(label) == 1:
      acc2["tp"] += 1
    else:
      acc2["fp"] += 1
  else:
    if int(label) == 1:
      acc2["fn"] += 1
    else:
      acc2["tn"] += 1
      
  if similarity > thresholds[3]:
    if int(label) == 1:
      acc3["tp"] += 1
    else:
      acc3["fp"] += 1
  else:
    if int(label) == 1:
      acc3["fn"] += 1
    else:
      acc3["tn"] += 1
      
  if similarity > thresholds[4]:
    if int(label) == 1:
      acc4["tp"] += 1
    else:
      acc4["fp"] += 1
  else:
    if int(label) == 1:
      acc4["fn"] += 1
    else:
      acc4["tn"] += 1
  
  
  pbar.update(1)   
    
  #print("Predicted Distance:", similarity.item())
  #print("Actual Label:", print_label)
  #print("\n")
  
  #if len(distance_same_identity) == 12:
  #  break
  
pbar.close()
print(f"Mean same distance: {np.average(distance_same_identity)}\n Mean different distance: {np.average(distance_different_identity)}")

num_pairings = len(test_dataloader)
accuracies = [(acc0["tp"]+acc0["tn"])/num_pairings, (acc1["tp"]+acc1["tn"])/num_pairings, (acc2["tp"]+acc2["tn"])/num_pairings, (acc3["tp"]+acc3["tn"])/num_pairings, (acc4["tp"]+acc4["tn"])/num_pairings]

print(f"{thresholds[0]}: {acc0}")
print(f"{thresholds[1]}: {acc1}")
print(f"{thresholds[2]}: {acc2}")
print(f"{thresholds[3]}: {acc3}")
print(f"{thresholds[4]}: {acc4}")

plt.figure(1)
plt.plot(thresholds, accuracies)
plt.xlabel("Thresholds")  
plt.ylabel("Accuracy")  
plt.title("Accuracy by threshold") 
plt.savefig(f"test_data/accuracy_plot_{model_name}.png")

print(f"All accuracies: {accuracies}")
print(f"Mean accuracy: {sum(accuracies)/len(accuracies)}")

plt.figure(2)
results = {"Same_identity": distance_same_identity, "Different_identity": distance_different_identity}
sns.kdeplot(results, fill = True, multiple="layer", common_grid=True, legend=True)
plt.title('Density Plot of Pairing cosine distances') #cosine similarities
plt.xlabel('Cosine simailarity') #Cosine similarity
plt.xlim(-1, 1)
plt.savefig(f"test_data/test_plot_{model_name}.png")

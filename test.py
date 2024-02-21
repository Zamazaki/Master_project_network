import torch
import torchvision
import matplotlib.pyplot as plt 
from conjoiner import Conjoiner
import torch.nn.functional as nnf
from dataset_loader import DatasetLoader
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Load the test dataset
test_dataset = DatasetLoader(training_csv="feature_vectors/test/feature_pairings.csv", training_dir="feature_vectors/test/")
test_dataloader = DataLoader(test_dataset, num_workers=6, batch_size=1, shuffle=True)

# Setup variables
MODEL_PATH = "checkpoints/model_5.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the network
model = Conjoiner()
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval() # Set model to evaluation mode

# Test the network
count = 0
for i, data in enumerate(test_dataloader, 0): 
  x0, x1, label = data
  #concat = torch.cat((x0, x1), 0)
  output1, output2 = model(x0.to(device), x1.to(device))

  eucledian_distance = nnf.pairwise_distance(output1, output2)
    
  if int(label) == 1:
    label="Same identity"
  else:
    label="Different identities"
    
  #plt.imshow(torchvision.utils.make_grid(concat))
  print("Predicted Eucledian Distance:-", eucledian_distance.item())
  print("Actual Label:-",label)
  count = count + 1
  if count == 10:
    break
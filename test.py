import torch
import torchvision
import matplotlib.pyplot as plt 
from conjoiner import Conjoiner
import torch.nn.functional as nnf

# Load the test dataset
#test_dataset = SiameseDataset(training_csv=testing_csv,training_dir=testing_dir,
#                                        transform=transforms.Compose([transforms.Resize((105,105)),
#                                                                      transforms.ToTensor()
#                                                                      ])
#                                       )

#test_dataloader = DataLoader(test_dataset,num_workers=6,batch_size=1,shuffle=True)

# Setup variables
MODEL_PATH = "path/to/trained/model"
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
  concat = torch.cat((x0, x1), 0)
  output1, output2 = model(x0.to(device), x1.to(device))

  eucledian_distance = nnf.pairwise_distance(output1, output2)
    
  if label==torch.FloatTensor([[0]]):
    label="Same identity"
  else:
    label="Different identities"
    
  plt.imshow(torchvision.utils.make_grid(concat))
  print("Predicted Eucledian Distance:-", eucledian_distance.item())
  print("Actual Label:-",label)
  count = count + 1
  if count == 10:
    break
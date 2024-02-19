import torch
from conjoiner import Conjoiner
import matplotlib.pyplot as plt  
import numpy as np
from dataset_loader import DatasetLoader
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def save_plot(x_axis, y_axis, xlabel, ylabel, title):
    plt.plot(x_axis, y_axis)
    plt.xlabel(xlabel)  
    plt.ylabel(ylabel)  
    plt.title(title) 
    #plt.show()
    plt.savefig("plot_loss.png")

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Original code from: https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss


# Create train function
# Boilerplate code from: https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942
def train(train_dataloader):
    loss=[]     
    for i, data in enumerate(train_dataloader,0):
        feat0, feat1 , label = data
        feat0, feat1 , label = feat0.cuda(), feat1.cuda(), label.cuda()
        
        optimizer.zero_grad()
        output1, output2 = model(feat0, feat1)
        loss_contrastive = loss_function(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        loss.append(loss_contrastive.item())
    return model, loss.mean()

def eval(eval_dataloader):
    loss=[] 
    for i, data in enumerate(eval_dataloader,0):
      feature1, feature2 , label = data
      feature1, feature2 , label = feature1.cuda(), feature2.cuda() , label.cuda()
      output1, output2 = model(feature1, feature2)
      loss_contrastive = loss_function(output1, output2, label)
      loss.append(loss_contrastive.item())
    loss = np.array(loss)
    return loss.mean()/len(eval_dataloader)

############################ Ready network for training ####################

# Setting up parameters
NUM_EPOCHS = 10
LOSS_MARGIN = 1.0  # Possible to make it bigger, since we are dealing with cross-modality
MODEL_PATH = "path/to/trained/model"

training_dataset = DatasetLoader(
    "feature_vectors/train/feature_pairings.csv",
    "feature_vectors/train",
    transform=transforms.Compose(
        [transforms.ToTensor()]
    ),
)
train_dataloader = DataLoader(training_dataset, num_workers=6, batch_size=12, shuffle=True)

validation_dataset = DatasetLoader(
    "feature_vectors/validation/feature_pairings.csv",
    "feature_vectors/validation",
    transform=transforms.Compose(
        [transforms.ToTensor()]
    ),
)
eval_dataloader = DataLoader(validation_dataset, num_workers=6, batch_size=12, shuffle=True)

# Declare network
model = Conjoiner().cuda()

# Declare Loss Function
loss_function = ContrastiveLoss(LOSS_MARGIN)

# Declare Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0005)

# Set the device to cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load a previous checkpoint
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

counter=[]
iteration_number = 0
loss_each_epoch = []
eval_loss_each_epoch = []

# Training loop
for epoch in range(1, NUM_EPOCHS + 1):
    # Train the model
    model, mean_loss = train(train_dataloader)
    eval_loss = eval(eval_dataloader)
    
    loss_each_epoch.append(mean_loss)
    eval_loss_each_epoch.append(eval_loss)

    print("Epoch {}\n Current training loss {}\n Current eval loss {}\n".format(epoch, mean_loss, eval_loss)) #loss_contrastive.item()
    iteration_number += 12 # Change according to batch size
    counter.append(iteration_number)

    # Save model throughout training
    if epoch % 3 == 0 or mean_loss < min(loss_each_epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': mean_loss,
            }, f"checkpoints/model_{epoch}.pt")
            
save_plot(counter, loss_each_epoch, "Pairs seen", "Loss", "Loss per 12 face pairs seen")  

# Save model at the end of training
torch.save({
                'epoch': NUM_EPOCHS,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': mean_loss,
                }, f"checkpoints/model_{NUM_EPOCHS}.pt")

#torch.save(model.state_dict(), "checkpoints/model1.pt")
#print("Model Saved Successfully") 

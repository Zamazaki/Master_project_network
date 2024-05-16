import torch
from conjoiner import Conjoiner
import matplotlib.pyplot as plt  
import numpy as np
from dataset_loader import DatasetLoader
#import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as nnf
from pytorch_metric_learning import distances, losses, miners, reducers

def save_plot(x_axis, y_axis, xlabel, ylabel, title, save_name):
    plt.plot(x_axis, y_axis)
    plt.xlabel(xlabel)  
    plt.ylabel(ylabel)  
    plt.title(title) 
    plt.legend(["Trianing loss","Validation loss"])
    #plt.show()
    plt.savefig(save_name)

# Create train function
def train(train_dataloader):
    # Set up progress bar
    pbar = tqdm(total=len(train_dataloader))
    
    loss_storage=[]     
    for i, data in enumerate(train_dataloader, 0):
        feature1, feature2 , label = data
        if device == "cuda":
            feature1, feature2 , label = feature1.cuda(), feature2.cuda(), label.cuda()
        
        optimizer.zero_grad()
        output1, output2 = model(feature1, feature2)
        #embeddings = model(feature1, feature2)
        #indices_tuple = mining_func(embeddings, label)
        
        #loss = loss_function(embeddings, label) #indices_tuple 
        #loss = loss_function(cosine_dist(output1, output2), label)
        loss = loss_function(output1, output2, label)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_storage.append(loss.item())
        
        # Update progress bar
        pbar.update(1)   
    loss_total = np.array(loss_storage)
    
    # Close progress bar
    pbar.close()
    
    return model, loss_total.mean()

def eval(eval_dataloader):
    loss_storage=[]
    with torch.no_grad():
        for i, data in enumerate(eval_dataloader, 0):
            feature1, feature2 , label = data
            if device == "cuda":
                feature1, feature2 , label = feature1.cuda(), feature2.cuda() , label.cuda()
            
            output1, output2 = model(feature1, feature2)
            #embeddings = model(feature1, feature2)
            
            loss = loss_function(output1, output2, label)
            #loss = loss_function(cosine_dist(output1, output2), label)
            #loss = loss_function(embeddings, label)
            loss_storage.append(loss.item())
        loss_total = np.array(loss_storage)
    return loss_total.mean()

############################ Ready network for training ####################

# Setting up parameters
NUM_EPOCHS = 20
LOSS_MARGIN = 0  # Only for cosine, Hinge margin is 1 by default
#MODEL_PATH = "checkpoints/face3d_BU4DFE_cosine_model_10.pt"
epochs_in_prev_model = 0
model_name = "Final_BU4DFE_cosine_zero_loss_margin" #"Final_FaceGen_cosine"

training_dataset = DatasetLoader( 
    "feature_vectors/train/feature_pairings.csv",
    "feature_vectors/train",
)
train_dataloader = DataLoader(training_dataset, num_workers=1, batch_size=32, shuffle=True)

validation_dataset = DatasetLoader(
    "feature_vectors/validation/feature_pairings.csv",
    "feature_vectors/validation",
)
eval_dataloader = DataLoader(validation_dataset, num_workers=1, batch_size=32, shuffle=True)

# Set the device to cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Declare network
if device == "cuda":
    model = Conjoiner().cuda()
else:
    model = Conjoiner()

# Declare Loss Function
distance = distances.CosineSimilarity()
cosine_dist = nn.CosineSimilarity(dim=1)
reducer = reducers.DoNothingReducer()
loss_function = nn.CosineEmbeddingLoss(margin = LOSS_MARGIN) #nn.SoftMarginLoss() #nn.HingeEmbeddingLoss()  # losses.CircleLoss(m=0.25, gamma=256) #losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5) #nn.CosineEmbeddingLoss(margin = LOSS_MARGIN) #ContrastiveLossEuclidan(LOSS_MARGIN)
#mining_func = miners.MultiSimilarityMiner(epsilon=0.1)

# Declare Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.05) #lr=1e-3 #0.0005

# Load a previous checkpoint
"""checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']"""

counter=[]
iteration_number = 0
loss_each_epoch = []
eval_loss_each_epoch = []

# Make a learning rate scheduler that decreases learning rate as time goes
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1e-2, total_iters=10)

# Training loop
for epoch in range(1 + epochs_in_prev_model, NUM_EPOCHS + 1 + epochs_in_prev_model):
    # Train the model
    model.train(True)
    print("Epoch: "+str(epoch))
    model, mean_loss = train(train_dataloader)
    
    # Evaluate the model
    model.eval()
    eval_loss = eval(eval_dataloader)
    
    loss_each_epoch.append(mean_loss)
    eval_loss_each_epoch.append(eval_loss)

    print("Epoch {}\n Current training loss {}\n Current eval loss {}\n".format(epoch, mean_loss, eval_loss)) #loss_contrastive.item()
    iteration_number += 1 
    counter.append(iteration_number)

    # Save model throughout training
    #if epoch % 3 == 0 or not len(loss_each_epoch) == 0 or mean_loss < min(loss_each_epoch[:-1]):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': mean_loss,
        }, f"checkpoints/{model_name}_model_{epoch}.pt")
    print(f"Model checkpoints/{model_name}_model_{epoch}.pt saved successfully\n") 
            
save_plot(counter, loss_each_epoch, "Epoch", "Loss", "Loss per epoch", f"plots/{model_name}_plot_loss.png")  
save_plot(counter, eval_loss_each_epoch, "Epoch", "Loss", "Loss per epoch", f"plots/{model_name}_plot_eval_loss.png")  

# Save model at the end of training
torch.save({
                'epoch': NUM_EPOCHS + epochs_in_prev_model,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': mean_loss,
                }, f"checkpoints/{model_name}_model_{NUM_EPOCHS + epochs_in_prev_model}.pt")

print(f"Model checkpoints/{model_name}_model_{NUM_EPOCHS + epochs_in_prev_model}.pt saved successfully") 

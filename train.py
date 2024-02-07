import torch
from conjoiner import Conjoiner
import matplotlib.pyplot as plt  

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
def train():
    loss=[] 
    counter=[]
    iteration_number = 0
    for epoch in range(1, NUM_EPOCHS):
        for i, data in enumerate(train_dataloader,0):
            feat0, feat1 , label = data
            feat0, feat1 , label = feat0.cuda(), feat1.cuda(), label.cuda()
            
            optimizer.zero_grad()
            output1,output2 = model(feat0, feat1)
            loss_contrastive = loss_function(output1, output2,label)
            loss_contrastive.backward()
            optimizer.step()
            
        print("Epoch {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
        iteration_number += 10 # Change according to batch size
        counter.append(iteration_number)
        loss.append(loss_contrastive.item())

        # Save model throughout training
        if epoch % 3 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, f"checkpoints/model_{epoch}.pt")
            
    save_plot(counter, loss, "Pairs seen", "Loss", "Loss per 10 face pairs seen")   
    return model, loss

############################ Ready network for training ####################

# Setting up parameters
NUM_EPOCHS = 10
LOSS_MARGIN = 1.0  # Possible to make it bigger, since we are dealing with cross-modality
MODEL_PATH = "path/to/trained/model"

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

# Train the model
model, loss = train()

# Save model at the end of training
torch.save({
                'epoch': NUM_EPOCHS,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, f"checkpoints/model_{NUM_EPOCHS}.pt")

#torch.save(model.state_dict(), "checkpoints/model1.pt")
#print("Model Saved Successfully") 

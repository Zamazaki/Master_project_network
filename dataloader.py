import pandas
import os
import torch
import numpy as np

class Dataloader():
    def __init__(self, training_csv="feature_vectors/feature_pairings.csv", training_dir="feature_vectors/"):
        # Prepare the labels and feature paths
        self.train_df = pandas.read_csv(training_csv)
        self.train_df.columns =["feature_vector1", "feature_vector2", "label"]
        self.train_dir = training_dir    

    def __getitem__(self,index):
        # Getting the feature path
        feature1_path=os.path.join(self.train_dir,self.train_df.iat[index,0])
        feature2_path=os.path.join(self.train_dir,self.train_df.iat[index,1])
        
        # Loading the features
        feature1 = torch.load(feature1_path)
        feature2 = torch.load(feature2_path)
        
        return feature1, feature2 , int(self.train_df.iat[index,2])
    def __len__(self):
        return len(self.train_df)
    
# Test
#dataloader = Dataloader()
#for row in dataloader:
#    print(row)
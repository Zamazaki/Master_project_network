import pandas
import os
import torch
import numpy as np

class DatasetLoader():
    def __init__(self, training_csv = "feature_vectors/train/feature_pairings.csv", training_dir = "feature_vectors/train/", transform = None):
        # Prepare the labels and feature paths
        self.train_df = pandas.read_csv(training_csv)
        self.train_df.columns =["feature_vector1", "feature_vector2", "label"]
        self.train_dir = training_dir
        self.transform = transform
        
        # Check for cuda availability 
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __getitem__(self,index):
        # Getting the feature path
        feature1_path=os.path.join(self.train_dir,self.train_df.iat[index,0])
        feature2_path=os.path.join(self.train_dir,self.train_df.iat[index,1])
        
        # Loading the features
        feature1 = torch.load(feature1_path)  #.to(self.device)
        feature2 = torch.load(feature2_path)  #.to(self.device)
        
        # Apply image transformations
        if self.transform is not None:
            feature1 = self.transform(feature1)
            feature2 = self.transform(feature2)
        
        return feature1, feature2 , int(self.train_df.iat[index,2])
    
    def __len__(self):
        return len(self.train_df)
    
class DatasetLoaderExtraLabel():
    def __init__(self, training_csv = "feature_vectors/train/feature_pairings.csv", training_dir = "feature_vectors/train/", transform = None):
        # Prepare the labels and feature paths
        self.train_df = pandas.read_csv(training_csv)
        self.train_df.columns =["frame1","frame2","dimention","label"]
        self.train_dir = training_dir
        self.transform = transform
        
        # Check for cuda availability 
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __getitem__(self,index):
        # Getting the feature path
        feature1_path=os.path.join(self.train_dir,self.train_df.iat[index,0])
        feature2_path=os.path.join(self.train_dir,self.train_df.iat[index,1])
        
        # Loading the features
        feature1 = torch.load(feature1_path)  #.to(self.device)
        feature2 = torch.load(feature2_path)  #.to(self.device)
        
        # Apply image transformations
        if self.transform is not None:
            feature1 = self.transform(feature1)
            feature2 = self.transform(feature2)
        
        return feature1, feature2 , int(self.train_df.iat[index,2]), int(self.train_df.iat[index,3])
    
    def __len__(self):
        return len(self.train_df)
# Test
#dataloader = FeatureDataloader()
#for row in dataloader:
#    print(row)
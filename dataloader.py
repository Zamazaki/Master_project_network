import pandas
import os
import torch
import numpy as np

class Dataloader():
    def __init__(self, training_csv=None, training_dir=None, transform=None):
        # used to prepare the labels and images path
        self.train_df=pandas.read_csv(training_csv)
        self.train_df.columns =["feature_vector1","feature_vector2","label"]
        self.train_dir = training_dir    
        #self.transform = transform

    def __getitem__(self,index):
        # getting the image path
        feature1_path=os.path.join(self.train_dir,self.train_df.iat[index,0])
        feature2_path=os.path.join(self.train_dir,self.train_df.iat[index,1])
        # Loading the image
        feature1 = np.loadtxt(feature1_path,
                 delimiter=",", dtype=np.float32)
        feature2 = np.loadtxt(feature2_path,
                 delimiter=",", dtype=np.float32)
        #img0 = Image.open(image1_path)
        #img1 = Image.open(image2_path)
        #img0 = img0.convert("L")
        #img1 = img1.convert("L")
        # Apply image transformations
        #if self.transform is not None:
        #    img0 = self.transform(img0)
        #    img1 = self.transform(img1)
        return feature1, feature2 , torch.from_numpy(np.array([int(self.train_df.iat[index,2])],dtype=np.float32))
    def __len__(self):
        return len(self.train_df)
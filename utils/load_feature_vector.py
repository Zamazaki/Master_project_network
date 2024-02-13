import torch
#import pandas

#def csv_to_feature_vector(csv_path):
#    values = pandas.read_csv(csv_path)
#    return values

#test = csv_to_feature_vector('/cluster/home/emmalei/Master_project_network/feature_vectors/2d/feat2d_1.csv')
test = torch.load('/cluster/home/emmalei/Master_project_network/feature_vectors/2d/feat2d_1.pt')
print(type(test))
print(test)
# I will let both feature vectors be stored as numpy arrays
import os
import numpy as np

path_feat_2d = "feature_vectors/2d"
path_feat_3d = "feature_vectors/2d"

pairing_array = []

feature_vectors_2d = sorted(os.listdir(path_feat_2d))
feature_vectors_3d = sorted(os.listdir(path_feat_3d))

# Assuming we have an equal amount of 2d and 3d feature vectors
number_of_pairings = len(feature_vectors_2d)
quarter_amount = number_of_pairings//4

# Split the set of feature vectors into four parts
# If the amount is not dividable by 4, the last quarter gets the extra pairs
second_quarter_start = quarter_amount
third_quarter_start = quarter_amount*2
forth_quarter_start = quarter_amount*3

# TODO: combine the three first for loops 
for i in range(quarter_amount):
    pairing_array.append(feature_vectors_2d[i])
    pairing_array.append(feature_vectors_3d[i])
    pairing_array.append(str(1))

for i in range(quarter_amount):
    pairing_array.append(feature_vectors_2d[second_quarter_start + i])
    pairing_array.append(feature_vectors_3d[third_quarter_start + i])
    pairing_array.append(str(0))
    
for i in range(quarter_amount):
    pairing_array.append(feature_vectors_2d[third_quarter_start + i])
    pairing_array.append(feature_vectors_3d[second_quarter_start + i])
    pairing_array.append(str(0))

for i in range(forth_quarter_start, number_of_pairings):
    pairing_array.append(feature_vectors_2d[i])
    pairing_array.append(feature_vectors_3d[i])
    pairing_array.append(str(1))
    
np.savetxt("feature_vectors/feature_pairings.csv", pairing_array, delimiter=",", fmt='%s')
import os
import numpy as np

# NOTICE: Current system is set up for FaceGen generated face pairs,
# so a new system will be used for BU3D-FE and BU4D-FE

path_feat_2d = "feature_vectors/2d"
path_feat_3d = "feature_vectors/3d"

pairing_array = []

feature_vectors_2d = ["2d/"+ feat for feat in sorted(os.listdir(path_feat_2d))]
feature_vectors_3d = ["3d/"+ feat for feat in sorted(os.listdir(path_feat_3d))]

# Assuming we have an equal amount of 2d and 3d feature vectors
number_of_pairings = len(feature_vectors_2d)

# If the amount is not dividable by 12, we throw an exception
if not number_of_pairings % 12 == 0:
    raise Exception("Remember that the number of features must be dividable by 12. And that the first half must be female, while the second half must be male.")

# Find a quarter and a twelfth of the total amount
twelfth_amount = number_of_pairings//12
quarter_amount = number_of_pairings//4

# Split the set of feature vectors into 14 parts
second_segment_start = twelfth_amount*3
third_segment_start = twelfth_amount*4
forth_segment_start = twelfth_amount*5
fifth_segment_start = twelfth_amount*6
sixth_segment_start = twelfth_amount*7
seventh_segment_start = twelfth_amount*8
eighth_segment_start = twelfth_amount*9


# Half of dataset is correctly matched
for i in range(quarter_amount):
    # Correct pairing female.jpg female.obj
    pairing_array.append(feature_vectors_2d[i])
    pairing_array.append(feature_vectors_3d[i])
    pairing_array.append(str(1))
    
    # Correct pairing male.jpg male.obj
    pairing_array.append(feature_vectors_2d[eighth_segment_start + i])
    pairing_array.append(feature_vectors_3d[eighth_segment_start + i])
    pairing_array.append(str(1))

# Other half is incorrectly matched in various ways
for i in range(twelfth_amount):
    # Incorrect pairing female.jpg female.obj
    pairing_array.append(feature_vectors_2d[second_segment_start + i])
    pairing_array.append(feature_vectors_3d[third_segment_start + i])
    pairing_array.append(str(0))
    
    # Incorrect pairing female.jpg female.obj
    pairing_array.append(feature_vectors_2d[third_segment_start + i])
    pairing_array.append(feature_vectors_3d[second_segment_start + i])
    pairing_array.append(str(0))
    
    # Incorrect pairing female.jpg male.obj
    pairing_array.append(feature_vectors_2d[third_segment_start + i])
    pairing_array.append(feature_vectors_3d[forth_segment_start  + i])
    pairing_array.append(str(0))
    
    # Incorrect pairing male.jpg female.obj
    pairing_array.append(feature_vectors_2d[forth_segment_start + i])
    pairing_array.append(feature_vectors_3d[third_segment_start + i])
    pairing_array.append(str(0))
    
    # Incorrect pairing male.jpg male.obj
    pairing_array.append(feature_vectors_2d[fifth_segment_start + i])
    pairing_array.append(feature_vectors_3d[sixth_segment_start + i])
    pairing_array.append(str(0))
    
    # Incorrect pairing male.jpg male.obj
    pairing_array.append(feature_vectors_2d[sixth_segment_start + i])
    pairing_array.append(feature_vectors_3d[fifth_segment_start + i])
    pairing_array.append(str(0))
    
# Save the array as a .csv
np.savetxt("feature_vectors/feature_pairings.csv", pairing_array, delimiter=",", fmt='%s')
# Cross-modal 3D-2D facial verification through sibling networks (by Emma Lu Eikemo)
This is the GitHub repository where I stored the code to my master thesis project. It uses the BU-4DFE dataset for training and validation, and the BU-3DFE dataset for testing. 
It is also possible to use synthetic FaceGen generated 2D-3D face pairs, if you have access to the FaceGen API.

In order to run the training and testing code, the datasets needs to be set up correctly first. 
Unfortunatly the BU-3DFE and BU-4DFE are to big to be uploaded on GitHub, so you need to download them yourself. 
Once they are within the project folder, follow the steps below to turn them into fetaure vectors.

## Extract 2D feature vectors
1. Enter the conda environment for tensorflow. It can be found in the conda_environment_specifications folder. Simply enter the folder and enter the following in the terminal:
```
conda env create -f env_tensorflow.yml
conda activate env_tensorflow
```
2. Create feature vectors from BU-3DFE. Simply go to line 34 and 35 in `feature2d_BU3DFE.py` and replace them with the path to your BU-3DFE folder and the 2d test feature vector folder.
Then run the file with:
```
python feature2d_BU3DFE.py
```
3. Create feature vectors from BU-4DFE. Pretty much the same process as before, go to line 37 and 38 in `feature2d_BU4DFE.py` and replace them with the path to your BU-4DFE folder and the 3d train feature vector folder.
Then run the file with:
```
python feature2d_BU3DFE.py
```
To also make the validation set. Uncomment all code that says "validation", and comment out code that says "training". Switch output directory to the 2d validation folder, then run the file again.

4. If you have a FaceGen dataset, go to line 12 in `create_2D_feature_FaceGen.py` and set it to your folder containing FaceGen generated images. Then run the file:
```
python create_2D_feature_FaceGen.py
```
## Extract 3D feature vectors
1. Enter the conda environment for pytorch. It can be found in the conda_environment_specifications folder. Simply enter the folder and enter the following in the terminal:
```
conda env create -f env_pytorch.yml
conda activate env_tensorflow
```
2. Convert .wrl files of BU-3DFE and BU-4DFE into .obj files. Enter the `wrl2obj` folder, and replace line 83 and 84 of the `wrl2obj_BU3DFE_BU4DFE.py` file to specify input and output folders. Also comment/uncomment the code that is labeled BU3DFE/BU4DFE depending on which of the datasets you are currently trying to convert. Then run the file with:
```
python wrl2obj_BU3DFE_BU4DFE.py
```
To use the same validation set as in the thesis, replace line 83 and 84 of the `wrl2obj_BU4DFE_validation.py` file the same way as described above, and run it with:
```
python wrl2obj_BU4DFE_validation.py
```
3. Convert the .obj files to .pcd files. First you must build the converter. Enter the `obj2pcd` folder and enter the following in the terminal:
```
cmake .
make
```
Now convert the .obj files to .pcd files of all the datasets by replacing line 4 and 5 in the `convert_bulk.py` file with your chosen input and output folders and then run the script with:
```
python convert_bulk.py
```

4. Convert the .pcd files into feature vectors. Go to `/FacePointCloudNet/compute_face_descriptor_BERNARDO/compute_pc_normals_3D-descriptors_and_save_to_file.py` and replace line 239 and 240 with your chosen input and output folders. (Though for output you should probably put them in their correct folder within the `feature_vectors` folder to make the training setup work correctly. The structure is included in this repository.) Then run the code with:
```
python compute_pc_normals_3D-descriptors_and_save_to_file.py
```

## Training the model
The network module that should send the feature vectors to a common feature space can be found in `conjoiner.py`, feel free to experiment with it if you like. This module is the only one that gets trained in this network as the other feature extractors use checkpoints from trained models. 

To train the model, just run `train.py`. Hyperparameters and other such things can be adjusted and can be found from line 77 and down. 

To test the model, run `test.py`. It will show the result of the predictions on various cosine similarity thresholds that can be adjusted by editing line 38. It also gives the mean distances of positiva and negative pairs, and spits out some plots about the accuracy of the thresholds and the cosine similarities.

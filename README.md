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

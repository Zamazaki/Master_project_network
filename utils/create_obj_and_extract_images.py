import os
import glob

# To be used inside the Preview folder within the FaceGen SDK

# In order to run this in git bash, first run this line
# alias python='winpty python.exe'
# (or try the permanent solution: https://stackoverflow.com/questions/32597209/python-not-working-in-the-command-line-of-git-bash)

input_dir = "Auto_generated_faces/"
output_dir = "obj_faces/"
image_dir = "jpg_faces/"

# Delete old output directory
if(os.path.isdir(output_dir)):
    os.system(f"rm -rf {output_dir}")

# create new output directory
os.mkdir(output_dir)


# Find all images in the input directory
images = glob.glob(os.path.join(input_dir, '*.jpg'))
if(len(images) > 0 ):
    if(os.path.isdir(image_dir)):
        os.system(f"rm -rf {image_dir}") # Delete old image directory
    os.mkdir(image_dir) # create new output directory

for image in images:
    basename = os.path.basename(image)  # Example: "rand0001.jpg"
    os.replace(input_dir+""+basename, image_dir+""+basename) # Move all to image directory


# identify all the xml files in the input directory
files = glob.glob(os.path.join(input_dir, '*.fg'))
print(len(files))

# loop through each 
for fil in files:
    basename = os.path.basename(fil)  # Example: "rand0001.fg"
    filename = os.path.splitext(basename)[0] # Example: "rand0001"
    
    # Create .fgmesh file out of the .fg file and place it in the same folder as the .fg file
    os.system(f"fg3 cssm Face/Face {input_dir}{filename}.fg {input_dir}{filename}.fgmesh")

    # Create .obj file out of the .fgmesh file and place it in the newly created output file
    os.system(f"fgbl mesh convert {input_dir}{filename}.fgmesh {output_dir}{filename}.obj")
    


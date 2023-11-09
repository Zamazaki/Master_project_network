import os
import glob

# In order to run this in git bash, first run this line
# alias python='winpty python.exe'
# (or try the permanent solution: https://stackoverflow.com/questions/32597209/python-not-working-in-the-command-line-of-git-bash)

input_dir = "Auto_generated_faces/"
output_dir = "obj_faces/"
image_dir = "jpg_faces/"

fg3_path = "C:/Users/Public/Fg3Sdk3V2Full/Fg3Sdk3V2Full/bin/win/x64/vs22/release/fg3.exe"
fgbl_path = "C:/Users/Public/Fg3Sdk3V2Full/Fg3Sdk3V2Full/bin/win/x64/vs22/release/fgbl.exe"

# Used when I want to add additional faces to existing folders
RE_NUMBER = True
NEW_START_NUMBER = 2500

# Delete old output directory
if not RE_NUMBER:
    if(os.path.isdir(output_dir)):
        os.system(f"rm -rf {output_dir}")
    os.mkdir(output_dir) # create new output directory

# Find all images in the input directory
images = glob.glob(os.path.join(input_dir, '*.jpg'))
if(len(images) > 0 ):
    if not RE_NUMBER:
        if(os.path.isdir(image_dir)):
            os.system(f"rm -rf {image_dir}") # Delete old image directory
        os.mkdir(image_dir) # create new image directory


if RE_NUMBER:
    new_number = NEW_START_NUMBER
    for image in images:
        basename = os.path.basename(image)  # Example: "rand0001.jpg"
        os.replace(input_dir+""+basename, image_dir+"rand"+str(new_number)+".jpg") # Move all to image directory and change name
        new_number += 1
else:
    for image in images:
        basename = os.path.basename(image)  # Example: "rand0001.jpg"
        os.replace(input_dir+""+basename, image_dir+""+basename) # Move all to image directory


# identify all the .fg files in the input directory
files = glob.glob(os.path.join(input_dir, '*.fg'))
print(len(files))

if RE_NUMBER:
    new_number = NEW_START_NUMBER
    for fil in files:
        basename = os.path.basename(fil)  # Example: "rand0001.fg"
        filename = os.path.splitext(basename)[0] # Example: "rand0001"
        new_filename = f"rand{new_number}"

        # Create .fgmesh file out of the .fg file and place it in the same folder as the .fg file
        os.system(f"{fg3_path} apply ssm Face/Face {input_dir}{filename}.fg {input_dir}{new_filename}.fgmesh")

        # Create .obj file out of the .fgmesh file and place it in the newly created output file
        os.system(f"{fgbl_path} mesh convert {input_dir}{new_filename}.fgmesh {output_dir}{new_filename}.obj")
        
        # Remove the .fgmesh file after using it, so we save space
        os.system(f"rm {input_dir}{new_filename}.fgmesh")
        new_number += 1
else:
    for fil in files:
        basename = os.path.basename(fil)  # Example: "rand0001.fg"
        filename = os.path.splitext(basename)[0] # Example: "rand0001"

        # Create .fgmesh file out of the .fg file and place it in the same folder as the .fg file
        os.system(f"{fg3_path} apply ssm Face/Face {input_dir}{filename}.fg {input_dir}{filename}.fgmesh")

        # Create .obj file out of the .fgmesh file and place it in the newly created output file
        os.system(f"{fgbl_path} mesh convert {input_dir}{filename}.fgmesh {output_dir}{filename}.obj")
        
        # Remove the .fgmesh file after using it, so we save space
        os.system(f"rm {input_dir}{filename}.fgmesh")


import os
import shutil

obj_folder_path = "/cluster/home/emmalei/Master_project_network/wrl_obj_test"
pcd_folder_path = "/cluster/home/emmalei/Master_project_network/FacePointCloudNet/compute_face_descriptor_BERNARDO/wrl_pcd_test"

obj_paths = sorted(os.listdir(obj_folder_path))
for path in obj_paths:
    # Create .pcd file from .obj file
    os.system(f"./obj2pcd {obj_folder_path}/{path}")
    
    # Move the created .pcd file to folder for feature extraction
    shutil.move(f"{obj_folder_path}/{path[0:-4]}.pcd", f"{pcd_folder_path}/")
    
    # Move the created .pcd file to folder for feature extraction (FaceGen)
    #shutil.move(f"{obj_folder_path}/{path[0:-4]}.pcd", f"{pcd_folder_path}/feat3d_{feature_counter}.pcd")
    #print(f"Created feat3d_{feature_counter}.pcd and moved it to {pcd_folder_path}\n")
    
    print(f"Created {path[0:-4]}.pcd and moved it to {pcd_folder_path}\n")
    
    # Rename files in folder
    #shutil.move(f"{obj_folder_path}/{path}", f"{obj_folder_path}/feat3d_{path[-11:-3]}.obj")
    #print(f"{obj_folder_path}/feat3d_{path[-11:-3]}.obj")
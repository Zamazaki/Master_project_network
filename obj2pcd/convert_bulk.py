import os
import shutil

obj_folder_path = "/cluster/home/emmalei/Master_project_network/obj_face_batch"
pcd_folder_path = "/cluster/home/emmalei/Master_project_network/FacePointCloudNet/compute_face_descriptor_BERNARDO/pcd_folder_train"

obj_paths = sorted(os.listdir(obj_folder_path))
feature_counter = 0
for path in obj_paths:
    os.system(f"./obj2pcd {obj_folder_path}/{path}")
    shutil.move(f"{obj_folder_path}/{path[0:-4]}.pcd", f"{pcd_folder_path}/feat3d_{feature_counter}.pcd")
    print(f"Created feat3d_{feature_counter}.pcd and moved it to {pcd_folder_path}\n")
    
    feature_counter += 1

#./obj2pcd ../FacePointCloudNet/compute_face_descriptor_BERNARDO/test_folder/rand0009.obj
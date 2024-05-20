# Based on
# https://github.com/ftdlyc/wrl2obj/blob/master/wrl2obj.py
# Originally created by ftdlyc <yclu.cn@gmail.com>

# Original contains: Normals, textures, different face extraction
# The new face extraction allow for quadruplets, conversion to pure triangle format

# Note on format: http://www.c3.hu/cryptogram/vrmltut/part5.html

# TODO check copyright

import torch
from torch_geometric.data import Data
import re
from glob import glob
import os

# Normals and textures are not collected
# ONLY TESTED WITH VRML 2.0
# TODO forces triangulation
# Note, also a bit slow, possibly due to the regex
def read_wrl_and_create_obj(in_file, out_file):
    fp = open(in_file, "r")
    buf = fp.read()
    fp.close()

    # Extract vertexs
    vertex_pos = re.search(r"Coordinate.+?{.+?point.+?\[(.+?)\].+?}", buf, re.S).regs[1]
    vertex_part = buf[vertex_pos[0]:vertex_pos[1]]
    vertex_lists = re.findall(r"[-+]?\d*\.?\d+e?[-+]?\d* [-+]?\d*\.?\d+e?[-+]?\d* [-+]?\d*\.?\d+e?[-+]?\d*", vertex_part)
    vertices = []
    for s in vertex_lists:
        assert len(s.split(" ")) == 3
        [x, y, z] = s.split(" ")
        vertices.append([float(x), float(y), float(z)])

    # Extract faces
    face_pos = re.search(r"coordIndex.+?\[(.+?)\]", buf, re.S).regs[1]
    face_part = buf[face_pos[0]:face_pos[1]]
    face_lists = re.findall(r"(([-+]?\d+,? ?){2,})", face_part)
    # Original/alt: '[-+]?\d+, [-+]?\d+, [-+]?\d+, [-+]?\d+'
    # Alt,  use (([-+]?\d+,? ?){2,}([-+]?\d+ ?))   then get match[0], and dont remove ,
    faces = []
    for s in face_lists:
        assert(len(s) == 2)  # Whole group, then the last digit (TODO?)
        s = s[0].rstrip(",")  # Take first arg(full match) and remove trailing comma if it exist
        splitted = re.split(" |, ", s)  # Split on ", " OR " " (non standard)

        assert int(splitted[-1]) == -1
        splitted2 = [int(x) for x in splitted[0:-1]]  # You can add +1 here if converting to obj
        if len(splitted2) == 3:
            faces.append(splitted2)
        else:
            # Triangulation
            # TODO fix so it is optional
            # TODO pull triangluation out? Use then in utils
            assert len(splitted2) > 3
            for i in range(1, len(splitted2)-1):  # For quad it is from 1 -> 4-1 (not included)
                tri_corner0 = splitted2[0]
                tri_corner1 = splitted2[i]
                tri_corner2 = splitted2[i+1]
                faces.append([tri_corner0, tri_corner1, tri_corner2])

    pos = torch.tensor(vertices, dtype=torch.float)
    face = torch.tensor(faces, dtype=torch.long).t().contiguous()
    data = Data(pos=pos, face=face)
    
    #Save obj
    fp = open(out_file, 'w')
    
    for v in vertices:
        fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    
    # # Note: This only adds triplets, fails on less, produces wrong output on more
    # # The output should also be +1, as .obj is 1 indexed for faces
    for f in faces:
        fp.write('f %d %d %d\n' % (f[0]+1, f[1]+1, f[2]+1))
    
    fp.close()
    
    return data

input_folder = "/cluster/home/emmalei/Master_project_network/BU4DFE"
output_folder = "/cluster/home/emmalei/Master_project_network/wrl_obj_validation"

id_list = sorted(glob(os.path.join(input_folder, "*", "")))
id_list.remove("/cluster/home/emmalei/Master_project_network/BU4DFE/BU_WrlViewer_V2/")
#print(id_list)
chosen_frames = [3, 4, 26, 27]

for id in id_list:
    name_id = id[-5]+id[-3:-1] # Gets gender and number for use in naming convention
    expression_folder = sorted(os.listdir(id))
    for expression in expression_folder:
        name_expression = expression[:3].upper()
        for i in range(1, 5):
            input_name = os.path.join(input_folder, id, expression, f'{chosen_frames[i-1]:03d}.wrl') # File name of specific frame in expression of an identity
            output_name = f"{output_folder}/feat3d_{name_id}{name_expression}_{i}.obj" # New file name following naming convention
            read_wrl_and_create_obj(input_name, output_name)
            
            print(f"In: {input_name}\nOut: {output_name}\n")
            
# Naming covention: feat2d_F01ANG_1.pt
# F01 means female identity number 01
# ANG is the current expression
# 1 is the current intensity of the frame (goes from 1 to 4 for BU-3DFE, and 1 to 20 for BU-4DFE)
    
    
    # Mostly to debug:
    # Save obj
    # fp = open("001-001.obj", 'w')
    #
    # for v in vertexs:
    #     fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    #
    # # Note: This only adds triplets, fails on less, produces wrong output on more
    # # The output should also be +1, as .obj is 1 indexed for faces
    # for f in faces:
    #     fp.write('f %d %d %d\n' % (f[0]+1, f[1]+1, f[2]+1))
    #
    # fp.close()


#if __name__ == "__main__":
#    file_path = "/cluster/home/emmalei/Master_project_network/BU3DFE/F0001/F0001_AN01WH_F3D.wrl"
#    data = read_wrl(in_file=file_path)
#    print(data)
# Based on
# https://github.com/ftdlyc/wrl2obj/blob/master/wrl2obj.py
# Originally created by ftdlyc <yclu.cn@gmail.com>

# Original contains: Normals, textures, different face extraction
# The new face extraction allow for quadruplets, conversion to pure triangle format

# Note on format: http://www.c3.hu/cryptogram/vrmltut/part5.html

# TODO check copyright

import torch
import re


# Normals and textures are not collected
# ONLY TESTED WITH VRML 2.0
# TODO forces triangulation
# Note, also a bit slow, possibly due to the regex
def read_wrl(in_file):
    fp = open(in_file, "r")
    buf = fp.read()
    fp.close()

    # Extract vertices
    vertex_pos = re.search(r"Coordinate.+?{.+?point.+?\[(.+?)\].+?}", buf, re.S).regs[1]
    vertex_part = buf[vertex_pos[0]:vertex_pos[1]]
    vertex_lists = re.findall(r"[-+]?\d*\.?\d+e?[-+]?\d* [-+]?\d*\.?\d+e?[-+]?\d* [-+]?\d*\.?\d+e?[-+]?\d*", vertex_part)
    vertices = []
    for s in vertex_lists:
        assert len(s.split(" ")) == 3
        [x, y, z] = s.split(" ")
        vertices.append([float(x), float(y), float(z)])

    pos = torch.tensor(vertices, dtype=torch.float)
    
    return pos

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


if __name__ == "__main__":
    file_path = "/lhome/haakowar/Downloads/BU_3DFE - 1/F0001/F0001_AN01WH_F3D.wrl"
    data = read_wrl(in_file=file_path)
    print(data)

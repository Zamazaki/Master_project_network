import os
import sys
import re


def wrl2obj(wrl_file, save_obj_file):
  fp = open(wrl_file, 'r')
  raw = fp.read()
  fp.close()
  #print("Length of wrl file: "+str(len(raw)))
  
  ## extract vertexs
  vertex_pos = re.search(r'Coordinate.+?{.+?point.+?\[(.+?)\].+?}', raw, re.S).regs[1]
  vertex_part = raw[vertex_pos[0]:vertex_pos[1]]
  vertex_lists = re.findall(r'[-+]?\d*\.?\d+e?[-+]?\d* [-+]?\d*\.?\d+e?[-+]?\d* [-+]?\d*\.?\d+e?[-+]?\d*', vertex_part)
  vertexs = []
  for s in vertex_lists:
      [x, y, z] = s.split(' ')
      vertexs.append([float(x), float(y), float(z)])

  ## extract normals
  normal_pos = re.search(r'Normal.+?{.+?vector.+?\[(.+?)\].+?}', raw, re.S).regs[1]
  normal_part = raw[normal_pos[0]:normal_pos[1]]
  normal_lists = re.findall(r'[-+]?\d*\.?\d+e?[-+]?\d* [-+]?\d*\.?\d+e?[-+]?\d* [-+]?\d*\.?\d+e?[-+]?\d*', normal_part)
  normals = []
  for s in normal_lists:
      [nx, ny, nz] = s.split(' ')
      normals.append([float(nx), float(ny), float(nz)])

  ## extract textures
  texture_pos = re.search(r'Color.+?{.+?color.+?\[(.+?)\].+?}', raw, re.S).regs[1]
  texture_part = raw[texture_pos[0]:texture_pos[1]]
  texture_lists = re.findall(r'[-+]?\d*\.?\d+e?[-+]?\d* [-+]?\d*\.?\d+e?[-+]?\d* [-+]?\d*\.?\d+e?[-+]?\d*', texture_part)
  textures = []
  for s in texture_lists:
      [r, g, b] = s.split(' ')
      textures.append([float(r), float(g), float(b)])

  ## extract faces
  face_pos = re.search(r'coordIndex.+?\[(.+?)\]', raw, re.S).regs[1]
  face_part = raw[face_pos[0]:face_pos[1]]
  face_lists = re.findall(r'[-+]?\d+, [-+]?\d+, [-+]?\d+, [-+]?\d+', face_part)
  faces = []
  for s in face_lists:
      [i0, i1, i2, i3] = s.split(', ')
      faces.append([int(i0) + 1, int(i1) + 1, int(i2) + 1])

  ## save obj
  fp = open(save_obj_file, 'w')

  for v, c, n in zip(vertexs, textures, normals):
    fp.write('v %f %f %f %f %f %f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    fp.write('vn %f %f %f\n' % (n[0], n[1], n[2]))

  for f in faces:
    fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

  fp.close()

wrl_super_dir_path = "/cluster/home/emmalei/Master_project_network/BU_3DFE.nosync"


wrl2obj(wrl_file='/cluster/home/emmalei/Master_project_network/BU_3DFE.nosync/F0001/F0001_AN01WH_F3D.wrl', save_obj_file = '/cluster/home/emmalei/Master_project_network/test_data/obj_convert_test.obj')

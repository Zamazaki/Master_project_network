import sys
import os
import numpy as np
import argparse
from pathlib import Path
from glob import glob
from tqdm import tqdm
import torch
from sklearn.preprocessing import normalize

import pcl

from train.train_triplet import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for point cloud normal computing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "-num_points", type=int, default=20000, help="Number of points to train with"
    )
    parser.add_argument(
        "-weight_decay", type=float, default=1e-5, help="L2 regularization coeff"
    )
    parser.add_argument("-lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "-lr_decay", type=float, default=0.7, help="Learning rate decay gamma"
    )
    parser.add_argument(
        "-decay_step", type=float, default=5e3, help="Learning rate decay step"
    )
    parser.add_argument(
        "-bn_momentum", type=float, default=0.5, help="Initial batch norm momentum"
    )
    parser.add_argument(
        "-bnm_decay", type=float, default=0.5, help="Batch norm momentum decay gamma"
    )
    parser.add_argument(
        "-model_checkpoint", type=str, default=None, help="Checkpoint to start from"
    )
    parser.add_argument(
        "-cls_checkpoint", type=str, default=None, help="Checkpoint to start from"
    )
    parser.add_argument(
        "-epochs", type=int, default=30, help="Number of epochs to train for"
    )
    parser.add_argument(
        "-run_name",
        type=str,
        default="cls_run_1",
        help="Name for run in tensorboard_logger",
    )
    # loss Classifier
    parser.add_argument('--margin', type=float, default=0.4, metavar='MARGIN',
                        help='the margin value for the triplet loss function (default: 1.0')
    parser.add_argument('--num_triplet', type=int, default=10000, metavar='num_triplet',
                        help='the margin value for the triplet loss function (default: 1e4')

    parser.add_argument('--num_class', type=int, default=500,
                        help='number of people(class)')
    parser.add_argument('--classifier_type', type=str, default='AL',
                        help='Which classifier for train. (MCP, AL, L)')

    # BERNARDO
    parser.add_argument(
        "-dataset_path", type=str, default='',
        help="Path of dataset root folder containing 3D face reconstructions (OBJ or PLY format)"
    )
    parser.add_argument("-file_ext", type=str, default='.obj', help="file extension to identify correct data to be loaded")
    parser.add_argument("-dataset_size", type=str, default='whole', help="whole or subset")
    parser.add_argument("-crop_face_radius_from_nose_tip", type=float, default='90.0', help="set radius in mm to crop face from tip nose")

    return parser.parse_args()


# BERNARDO
class Tree:
    def walk(self, dir_path: Path):
        contents = list(dir_path.iterdir())
        for path in contents:
            if path.is_dir():  # extend the prefix and recurse:
                yield str(path)
                yield from self.walk(path)

    def get_all_sub_folders(self, dir_path: str):
        folders = [dir_path]
        for folder in Tree().walk(Path(os.getcwd()) / dir_path):
            # print(folder)
            folders.append(folder)
        folders.sort()
        return folders

    # TESTE
    def get_subsample_lfw_subjects_and_samples_names(self, dir_path):
        def load_lfw_subsamples():
            path_file = './lfw_subsamples_folders_with_3_images.txt'
            with open(path_file) as f:
                sujects_names = [line.replace('\n', '') for line in f]
                return sujects_names

        sujects_names = load_lfw_subsamples()
        sub_folders = []
        for name in sujects_names:
            sub_folders += self.get_all_sub_folders(dir_path + '/' + name)
        return sub_folders




def get_normals(cloud, radius=30):
    """
    FROM: https://pcl.gitbook.io/tutorial/part-2/part02-chapter03/part02-chapter03-normal-pcl-python
    The actual *compute* call from the NormalEstimation class does nothing internally but:
    for each point p in cloud P
        1. get the nearest neighbors of p
        2. compute the surface normal n of p
        3. check if n is consistently oriented towards the viewpoint and flip otherwise

    # normals: pcl._pcl.PointCloud_Normal,size: 26475
    # cloud: pcl._pcl.PointCloud
    """
    feature = cloud.make_NormalEstimation()
    feature.set_KSearch(radius)    # Use all neighbors in a sphere of radius 5 cm
    normals = feature.compute()

    print('normals:', normals[0])
    print('normals.to_array():', normals.to_array().shape)

    # normals = normals.to_array()
    # normals[:,0:3] = normals[:,0:3] * -1
    # normals = pcl.PointCloud_Normal(normals)  # TESTE
    normals = pcl.PointCloud_Normal(normals.to_array() * -1)  # TESTE

    # return normals            # original
    return normals.to_array()   # BERNARDO


def get_pointcloud_with_normals(cloud):
    cloud_with_normals = cloud.to_array()
    normals = get_normals(cloud)
    cloud_with_normals = np.hstack((cloud_with_normals, normals))
    return cloud_with_normals


def filter_points_by_radius(cloud, keypoint_ref, radius=90.0):
    keypoint_ref = np.expand_dims(np.asarray(keypoint_ref, dtype=np.float32), axis=0)
    # print('keypoint_ref:', keypoint_ref)
    searchPoint = pcl.PointCloud(keypoint_ref)
    # print('searchPoint:', searchPoint[0])
    kdtree = pcl.KdTreeFLANN(cloud)
    [ind, sqdist] = kdtree.radius_search_for_cloud(searchPoint, radius, cloud.size)
    # [ind, sqdist] = kdtree.nearest_k_search_for_cloud(searchPoint, 5)
    ind = ind[0]
    ind = ind[ind != 0]
    cloud = pcl.PointCloud(cloud.to_array()[ind])
    return cloud


def preprocess_pointcloud_with_normals(pc_with_normals):
    point_set = pc_with_normals
    # normalize
    point_set[:, 0:3] = (point_set[:, 0:3]) / 100 
    point_set = torch.from_numpy(point_set)

    input = point_set
    input = input.unsqueeze(0).contiguous()
    input = input.to("cuda", non_blocking=True)
  
    return input



def load_pc_and_compute_normals(args, model, folder):
    image_paths = sorted(os.listdir(folder))

    for image_path_OBJ in tqdm(image_paths):

        name = Path(image_path_OBJ).stem

        print('Loading point cloud:', image_path_OBJ)
        cloud_from_OBJ = pcl.load(folder+"/"+image_path_OBJ)


        # Can be used for cropping later
        #path_key_points = '/'.join(image_path_OBJ.split('/')[:-1]) + '/' + 'kpt68.npy'
        #key_points = pcl.load(path_key_points)
        #cloud_from_OBJ = filter_points_by_radius(cloud_from_OBJ, key_points[30], radius=args.crop_face_radius_from_nose_tip)

        pc_with_normals_from_OBJ = get_pointcloud_with_normals(cloud_from_OBJ)

        pc_with_normals_from_OBJ = preprocess_pointcloud_with_normals(pc_with_normals_from_OBJ)


        print('Computing 3D face descriptor ...')

        feat_from_OBJ = model.forward(pc_with_normals_from_OBJ)  # 1x512
        path_feat_norm_from_OBJ = f'/cluster/home/emmalei/Master_project_network/feature_vectors/test/3d/{name}.pt' #_3D_face_descriptor

        print('Saving 3D face descriptor:', path_feat_norm_from_OBJ, end=' ... ')
        

        torch.save(feat_from_OBJ.cpu().detach().numpy()[0], path_feat_norm_from_OBJ)
        print('Saved!')



def build_Pointnet_model(args):
    model = Pointnet(input_channels=3, use_xyz=True)
    model.cuda()
    # 512 is dimension of feature
    classifier = {
        'MCP': layer.MarginCosineProduct(512, args.num_class).cuda(),
        'AL': layer.AngleLinear(512, args.num_class).cuda(),
        'L': torch.nn.Linear(512, args.num_class, bias=False).cuda()
    }[args.classifier_type]

    # criterion = nn.TripletMarginLoss(margin=0.5, p=2)
    optimizer = optim.Adam(
        [{'params': model.parameters()}, {'params': classifier.parameters()}],
        lr=lr, weight_decay=args.weight_decay
    )

    print('compute_face_descriptor_BERNARDO.py: main(): Loading trained model...')
    if args.model_checkpoint is not None:
        checkpoint_status = pt_utils.load_checkpoint(
            model, optimizer, filename=args.model_checkpoint.split(".")[0]
        )
        if checkpoint_status is not None:
            it, start_epoch, best_loss = checkpoint_status

    model.eval()
    optimizer.zero_grad()
    return model



def main(args):
    # load dataset (LFW and TALFW)
    print('face_recognition_3d_descriptor.py: main(): Loading sub-folders of dataset', args.dataset_path, '...')

    model = build_Pointnet_model(args)
    load_pc_and_compute_normals(args, model, "wrl_pcd_test")



if __name__ == '__main__':


    sys.argv += ['-model_checkpoint', '/cluster/home/emmalei/Master_project_network/FacePointCloudNet/checkpoints/20191028_1000cls_model_best']

    sys.argv += ['-dataset_path', '/cluster/home/emmalei/Master_project_network/FacePointCloudNet/compute_face_descriptor_BERNARDO/wrl_pcd_test']

    sys.argv += ['-dataset_size', 'whole']

    args = parse_args()


    main(args)

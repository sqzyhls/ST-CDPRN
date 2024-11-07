import open3d as o3d
import numpy as np
import os
import os
from numpy import *
import torch

def read_all_file_name():
    file_path = '/media/deepsea/DATA/fz/pythonProject/pred/toyplane'
    file_name = os.listdir(file_path)
    return file_name




def chamfer_distance(pcl1, pcl2):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pcl1)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcl2)

    dist_pcl1_to_pcl2 = pcd2.compute_point_cloud_distance(pcd1)
    dist_pcl2_to_pcl1 = pcd1.compute_point_cloud_distance(pcd2)

    cd = np.mean(dist_pcl1_to_pcl2) + np.mean(dist_pcl2_to_pcl1)

    return cd
def torch_center_and_normalize(points,p="inf"):
    """
    a helper pytorch function that normalize and center 3D points clouds 
    """
    N = points.shape[0]
    center = points.mean(0)
    if p != "fro" and p!= "no":
        scale = torch.max(torch.norm(points - center, p=float(p),dim=1))
    elif p=="fro" :
        scale = torch.norm(points - center, p=p )
    elif p=="no":
        scale = 1.0
    points = points - center.expand(N, 3)
    points = points * (1.0 / float(scale))
    return points

file_names = read_all_file_name()
print(file_names) # 返回文件夹中的文件名
all_cd = []
for i,val in enumerate(file_names):
    # 读取点云数据
    point_cloud = o3d.io.read_point_cloud("/media/deepsea/DATA/fz/pythonProject/gt/toyplane/"+val)
    print(len(point_cloud.points))
    # 将点云数据转化为Numpy数组
    point_cloud_np = np.asarray(point_cloud.points)
    print(point_cloud_np.shape)
    point_cloud_np = torch.from_numpy(point_cloud_np).to(torch.float)
    point_cloud_np = torch_center_and_normalize(point_cloud_np,"inf")
    pre_cloud=o3d.io.read_point_cloud("/media/deepsea/DATA/fz/pythonProject/pred/toyplane/"+val)
    pre_np=np.asarray(pre_cloud.points)
    print(len(pre_cloud.points))
    pre_np = torch.from_numpy(pre_np).to(torch.float)
    pre_np = torch_center_and_normalize(pre_np,"inf")
    cd = chamfer_distance(point_cloud_np,pre_np)
    print("Average chamfer_distance",cd)
    all_cd.append(cd)
avg_cd = mean(all_cd)
print("avg_cd:",avg_cd)
# 将点云数据转化为张量
tensor = np.expand_dims(point_cloud_np, axis=0)  # 在第0维添加一个维度

# 打印张量的形状
print(tensor.shape)


# # 加载第一个点云文件
# pcd1 = o3d.io.read_point_cloud("gt.ply")

# # 加载第二个点云文件
# pcd2 = o3d.io.read_point_cloud("pre.ply")

# # 计算两个点云之间的倒角距离
# distance = pcd1.compute_point_cloud_distance(pcd2)

# print("倒角距离为:", distance)


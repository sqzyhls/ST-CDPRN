import open3d as o3d
import numpy as np

def chamfer_distance(point_cloud1, point_cloud2):
    # Create KDTree for fast nearest neighbor search
    kdtree1 = o3d.geometry.KDTreeFlann(point_cloud1)
    kdtree2 = o3d.geometry.KDTreeFlann(point_cloud2)

    # For each point in point cloud 1, find the nearest neighbor in point cloud 2
    distances_1to2 = []
    for point in point_cloud1.points:
        [_, dist, _] = kdtree2.search_knn_vector_3d(point, 1)
        distances_1to2.append(dist[0])

    # For each point in point cloud 2, find the nearest neighbor in point cloud 1
    distances_2to1 = []
    for point in point_cloud2.points:
        [_, dist, _] = kdtree1.search_knn_vector_3d(point, 1)
        distances_2to1.append(dist[0])

    # Compute Chamfer Distance as the sum of nearest neighbor distances in both directions
    chamfer_dist = np.mean(distances_1to2) + np.mean(distances_2to1)

    return chamfer_dist

# Load point cloud data
point_cloud1 = o3d.io.read_point_cloud("pre.ply")
point_cloud2 = o3d.io.read_point_cloud("gt.ply")

# Calculate Chamfer Distance between the two point clouds
cd = chamfer_distance(point_cloud1, point_cloud2)
print("Chamfer Distance: ", cd)
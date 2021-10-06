from ssl import PROTOCOL_TLS
from numpy.lib.function_base import _average_dispatcher, average
import open3d as o3d
import os
import numpy as np
import random
from pandas import DataFrame
import copy
import time
from pyntcloud import PyntCloud


def voxel_filter(point_cloud, leaf_size, filter_mode):
    start = time.time()
    filtered_points = []
    x_max, y_max, z_max = np.amax(point_cloud, axis=0)
    x_min, y_min, z_min = np.amin(point_cloud, axis=0)
    size_r = leaf_size
    Dx = (x_max - x_min)/size_r 
    Dy = (y_max - y_min)/size_r 
    Dz = (z_max - z_min)/size_r 
    h  = list()
    for i in range(len(point_cloud)):
        hx = np.floor((point_cloud[i][0] - x_min)/size_r)
        hy = np.floor((point_cloud[i][1] - y_min)/size_r)
        hz = np.floor((point_cloud[i][2] - z_min)/size_r)
        h.append(hx + hy*Dx + hz*Dx*Dy)

    h = np.array(h)
    h_indice = np.argsort(h)   # 这里h中从小到大排列，对应的indice     [1 2 4 3 0 5]
    h_sorted = h[h_indice]     # 将h从小到大进行排列 [2 3 4 5 6 9]

    # 将h值相同的点放入到同一个grid中，并进行筛选，并区分random和centroid两种滤波方式 
    count = 0
    for i in range(len(h_sorted) - 1):
        if h_sorted[i] == h_sorted[i+1]:
            continue
        else:
            if(filter_mode == 'centroid'):
                point_idx = h_indice[count: i+1]  # 因为取不到i+1,所以这里范围是[count: i+1]
                filtered_points.append(np.mean(point_cloud[point_idx], axis=0))
                count = i
            elif(filter_mode == 'random'):
                point_idx = h_indice[count: i+1]
                random_points = random.choice(point_cloud[point_idx])
                filtered_points.append(random_points)
                count = i

    filtered_points = np.array(filtered_points, dtype=np.float64)
    end = time.time()
    runing_time = end -start
    print("for '{}' this mode, the running time is {} ".format(filter_mode, runing_time))
    return filtered_points

def main():
     # 加载原始点云，txt处理
    point_cloud_raw = np.genfromtxt(r"/home/shaoxiang/Desktop/shenlan_pcl/modelnet40_normal_resampled/airplane/airplane_0001.txt", delimiter=",")  #为 xyz的 N*3矩阵
    # point_cloud_raw = np.genfromtxt(r"/home/shaoxiang/Desktop/shenlan_pcl/modelnet40_normal_resampled/plant/plant_0001.txt", delimiter=",")  #为 xyz的 N*3矩阵
    point_cloud_raw = DataFrame(point_cloud_raw[:, 0:3])  # 选取每一列 的 第0个元素到第二个元素   [0,3), 所以这里输出的是NX3的矩阵
    point_cloud_raw.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    point_cloud_pynt = PyntCloud(point_cloud_raw)  # 将points的数据 存到结构体中

    point_cloud_o3d_origin          = point_cloud_pynt.to_instance('open3d', mesh = False)
    point_cloud_o3d_filter_centroid = o3d.geometry.PointCloud() 
    point_cloud_o3d_filter_random   = o3d.geometry.PointCloud()  

    points = np.array(point_cloud_o3d_origin.points)
    print('the input points shape is ', points.shape)

    filtered_cloud_centroid = voxel_filter(points, 0.07, "centroid")
    print('the filtered_cloud_centroid shape is ', filtered_cloud_centroid.shape)
    point_cloud_o3d_filter_centroid.points = o3d.utility.Vector3dVector(filtered_cloud_centroid)

    filtered_cloud_random = voxel_filter(points, 0.07, "random")
    print('the filtered_cloud_random shape is ', filtered_cloud_random.shape)
    point_cloud_o3d_filter_random.points = o3d.utility.Vector3dVector(filtered_cloud_random)

    pcd_translation_z_centroid = copy.deepcopy(point_cloud_o3d_filter_centroid).translate((0,1.0,0))
    pcd_translation_z_random   = copy.deepcopy(point_cloud_o3d_filter_random).translate((0,2.0,0))
    o3d.visualization.draw_geometries([pcd_translation_z_centroid, pcd_translation_z_random, point_cloud_o3d_origin])

if __name__ == '__main__':
    main()




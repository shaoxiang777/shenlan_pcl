# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

from ssl import PROTOCOL_TLS
from numpy.lib.function_base import _average_dispatcher, average
import open3d as o3d
import os
import numpy as np
from scipy.spatial import KDTree
from pandas import DataFrame
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt
import copy


# 加载原始点云，txt处理
point_cloud_raw = np.genfromtxt(r"/home/shaoxiang/Desktop/shenlan_pcl/modelnet40_normal_resampled/airplane/airplane_0001.txt", delimiter=",")  #为 xyz的 N*3矩阵
# point_cloud_raw = np.genfromtxt(r"/home/shaoxiang/Desktop/shenlan_pcl/modelnet40_normal_resampled/plant/plant_0001.txt", delimiter=",")  #为 xyz的 N*3矩阵
point_cloud_raw = DataFrame(point_cloud_raw[:, 0:3])  # 选取每一列 的 第0个元素到第二个元素   [0,3), 所以这里输出的是NX3的矩阵
point_cloud_raw.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
point_cloud_pynt = PyntCloud(point_cloud_raw)  # 将points的数据 存到结构体中
point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)  # 实例化


def raw_pcl_vis():
    o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云
    # 从点云中获取点，只对点进行处理
    print('Some informations about PLC is ',point_cloud_o3d)              #打印点数


# 功能：计算PCA的函数
def PCA(data, sort = True):
    average_data = np.mean(data, axis = 0)                  # average_data 是 1*3
    decentration_matrix = data - average_data
    H = np.dot(decentration_matrix.T, decentration_matrix)  #求解协方差矩阵 H H是一个3*3的矩阵
    eigenvectors, eigenvalues, eigenvectors_T = np.linalg.svd(H)  # SVD求解特征值、特征向量

    if sort:
        sort = eigenvalues.argsort()[::-1]     # 降序排列
        eigenvalues = eigenvalues[sort]        # 索引
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors

def using_PCA_to_visualize_PCL():
    # 用PCA来分析点云主方向
    w, v = PCA(point_cloud_raw)
    point_cloud_vector1 = v[:, 0]
    point_cloud_vector2 = v[:, 1]
    point_cloud_vector3 = v[:, 2]
    point_cloud_vector = v[:, 0:3]
    print('The eigenvector size is ', v.shape)  # 这里验证了它的大小是3*3的
    print('The 1.PCA of this pointcloud is: ', point_cloud_vector1)
    print('The 2.PCA of this pointcloud is: ', point_cloud_vector2)
    print('The 3.PCA of this pointcloud is: ', point_cloud_vector3)

    point = [[0,0,0], point_cloud_vector1, point_cloud_vector2, point_cloud_vector3]
    lines = [[0,1], [0,2], [0,3]]
    colors = [[1,0,0],[0,1,0],[0,0,1]]

    # build target LineSet in open3d, used to show PCA
    line_set = o3d.geometry.LineSet(points = o3d.utility.Vector3dVector(point), lines = o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud_o3d, line_set])

def surface_normal_estimation():
    #循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)      # 将原始点云数据输入到KD,进行近邻取点
    normals = []
    print("There are ", point_cloud_raw.shape[0], " in this point cloud.")
    for i in range(point_cloud_raw.shape[0]):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[i], 10)  # 取10个临近点进行曲线拟合     返回值 [int, open3d.utility.IntVector, open3d.utility.DoubleVector]
        k_nearest_point = np.asarray(point_cloud_o3d.points)[idx, :]                # 找出每一点的10个临近点，类似于拟合成曲面，然后进行PCA找到特征向量最小的值，作为法向量
        w, v = PCA(k_nearest_point)
        normals.append(v[:, 2])  # 因为计算出来的点云中只有3个特征向量，所以最后一列是最无关的

    # 屏蔽结束
    normals = np.array(normals, dtype=np.float64)
    # TODO： 此处将法向量放在normals中
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([point_cloud_o3d], point_show_normal=True)
surface_normal_estimation()

# show point cloud in 2D 
def Point_Show_2D(pca_point_cloud):
    x = []
    y = []
    pca_point_cloud = np.asarray(pca_point_cloud)
    for i in range(10000):
        x.append(pca_point_cloud[i][0])
        y.append(pca_point_cloud[i][1])
    plt.title('Point Cloud 2D in x-y plane')
    plt.scatter(x, y)      # plt.scatter()函数用于生成一个scatter散点图
    plt.show()

# show point cloud in 3D
def Point_Show_3D(points, format_name):
    fig = plt.figure(dpi = 150)
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], cmap='spectral', s=2, linewidths=0, alpha=1, marker=".")
    plt.title('Point Cloud 3D for {name}'.format(name = format_name))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def downsample():
    w, v = PCA(point_cloud_raw)
    point_cloud_vector = v[0:2, 0:3]
    # The original data is processed by dimensionality reduction
    point_cloud_encode = (np.dot(point_cloud_vector, point_cloud_raw.T)).T   # 3*3 dot 3*N 结果为3*N 之后在转置为N*3
    print(point_cloud_encode.shape)
    Point_Show_2D(point_cloud_encode)

    # Use the main direction for ascending
    point_cloud_decode = (np.dot(point_cloud_vector.T, point_cloud_encode.T)).T
    Point_Show_3D(point_cloud_decode, "decoded PC")

    point_cloud_raw_1 = np.asarray(point_cloud_raw)
    Point_Show_3D(point_cloud_raw_1, "raw PC")
    
    pcd_raw = o3d.geometry.PointCloud()
    pcd_raw.points = o3d.utility.Vector3dVector(point_cloud_raw_1)
    # o3d.visualization.draw_geometries([pcd_raw])
    pcd_decode = o3d.geometry.PointCloud()
    pcd_decode.points = o3d.utility.Vector3dVector(point_cloud_decode)
    pcd_decode.paint_uniform_color([0, 0, 1.0])

    pcd_translation_z = copy.deepcopy(pcd_decode).translate((0,0.5,0))
    o3d.visualization.draw_geometries([pcd_translation_z, pcd_raw])



    



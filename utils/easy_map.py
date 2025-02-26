from utils.astar import grid_path
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import math

class easy_grid_map:
    def __init__(self,map,origin,resolution) -> None:
        self.map = map
        self.origin = np.array(origin)[0:2]
        self.resolution = resolution
        self.map_shape = map.shape
    
    def to_grid_frame(self,point_map_frame,point_frame='uv'):
        #给一系列图像坐标系中的点，换到地图坐标系下
        #用numpy.array格式输入
        point_map_frame = point_map_frame.copy().reshape((-1,2))
        xy_grid = (point_map_frame - self.origin)/self.resolution
        xy_grid = xy_grid.astype(np.int64)
        uv = xy_grid
        uv[:,1] = self.map_shape[0] - uv[:,1]
        if point_frame == 'uv':
            return uv
        else:
            return np.fliplr(uv)

    def to_map_frame(self,point_grid_frame,point_frame = 'uv'):
        #给一个uv或者rc格式的点，转换到map frame下
        #用numpy.array格式输入
        point_grid_frame = point_grid_frame.copy().reshape((-1,2))
        if point_frame=='uv':
            pass
        elif point_frame =='rc':
            point_grid_frame = np.flip(point_grid_frame,axis=1)
        else:
            print("Error point frame")
            return
        point_grid_frame[:,1] = self.map_shape[0] - point_grid_frame[:,1]

        xy_map = point_grid_frame * self.resolution + self.origin
        return xy_map

    def vis_map_points(self,points,point_frame = 'uv'):
        #point:n*2 array
        #point frame should choose from uv rc
        if point_frame=='uv':
            points = points.reshape((-1,2))
        elif point_frame =='rc':
            points = points.reshape((-1,2))
            points = np.flip(points,axis=1)
        else:
            print("Error point frame")
            return
        fig, ax = plt.subplots()
        ax.axis('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        ax.scatter(points[:,0], points[:,1], 10,color='red', marker='o')
        ax.imshow(self.map, cmap='gray')
        return fig,ax
        
    
    def calculate_path_between2points(self,point1,point2,vis_flag = False):
        distance_map = scipy.ndimage.distance_transform_edt(self.map == 0)
        calculate_grid_path = grid_path(self.map, distance_map,point1, point2)
        calculate_grid_path.get_path()
        if vis_flag:
            calculate_grid_path.vis_path()
        return calculate_grid_path.foundPath, calculate_grid_path.path_length
        # return calculate_grid_path.foundPath

    def random_points_on_map(self,num):
        #返回以row col给定的一系列点
        # 获取矩阵中值为0的点的坐标
        #设置随机种子
        np.random.seed(0)
        zero_indices = np.argwhere(self.map == 255)

        # 随机选择坐标
        random_indices = np.random.choice(zero_indices.shape[0], size=num, replace=False)

        # 获取随机选择的点的坐标
        random_points = zero_indices[random_indices]
        return random_points

def get_obstacle_distances(map_data, max_distance, point,resolution = 0.05):
    #map_data:map
    #max_distance: 传感器半径
    #point: 地图上的一个点，以rc坐标系给出
    #resolution: 地图分辨率
    x, y = point
    obstacle_distances = []
    max_distance_grid = max_distance // resolution
    for angle in range(360):
        radian = math.radians(angle)
        dx = math.cos(radian)
        dy = math.sin(radian)
        distance = 0
        while distance < max_distance_grid:
            x_check = int(round(x + dy * distance)) #注意dy增加行
            y_check = int(round(y + dx * distance)) #注意dx增加列数
            # if x_check < 0 or x_check >= map_data.shape[0] or y_check < 0 or y_check >= map_data.shape[1]:
            #     obstacle_distances.append(max_distance_grid)
            #     break
            if map_data[x_check, y_check] !=0 :
                obstacle_distances.append(distance)
                break
            distance += 1
        else:
            obstacle_distances.append(max_distance_grid+21)
    return np.array(obstacle_distances) * resolution
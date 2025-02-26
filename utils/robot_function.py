# this file contains some function that is useful
# most of them will not be used in this project

from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import copy
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
from scipy.optimize import minimize




def if_frontier(window):
    if 100 in window: 
        return False
    if 0 not in window: 
        return False
    if 255 not in window:
        return False
    return True



def detect_frontier(image):
    kernel = np.ones((3, 3), np.uint8)
    free_space = image ==255
    unknown = (image == 100).astype(np.uint8)
    # obs = (image == 0).astype(np.uint8)
    expanded_unknown = cv2.dilate(unknown, kernel).astype(bool)
    # expanded_obs = cv2.dilate(obs, kernel).astype(bool)
    # near = free_space & expanded_unknown & (~expanded_obs)
    near = free_space & expanded_unknown
    return np.column_stack(np.where(near)) #row, col

def calculate_entropy(array):
    num_bins = 20
    hist, bins = np.histogram(array, bins=num_bins)
    probabilities = hist / len(array)
    probabilities = probabilities[np.where(probabilities != 0)] 
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy



def sparse_point_cloud(data,min_dis):
    #use norm 1 for dis func
    #select the minimal value in value map
    new_data = []
    for now_data in data:
        add_flag = True
        for target_data in new_data:
            if abs(target_data[0] - now_data[0]) + abs(target_data[1] - now_data[1])<min_dis:
                add_flag = False
                break
        
        if add_flag:
            new_data.append(now_data)
    return np.array(new_data)

def expand_obstacles(map_data, expand_distance=2):
    map_binary = (map_data == 100).astype(np.uint8)
    kernel_size = 2 * expand_distance + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded_map_binary = cv2.dilate(map_binary, kernel)
    extended_map = copy.deepcopy(map_data)
    extended_map[expanded_map_binary == 1] = 100

    return extended_map

def find_local_max_rect(image, seed_point, map_origin, map_reso):
    #input: image and (x,y) in pixel frame format start point
    #output rect position of (x1, y1, x2, y2) in pixel frame
    def find_nearest_obstacle_position(image, x, y):
        obstacles = np.argwhere((image == 100) | (image == 255))
        point = np.array([[y, x]])
        distances = cdist(point, obstacles)
        min_distance_idx = np.argmin(distances)
        nearest_obstacle_position = obstacles[min_distance_idx]
        nearest_obstacle_position = np.array([nearest_obstacle_position[1],nearest_obstacle_position[0]])
        return nearest_obstacle_position
    
    vertex_pose_pixel = (np.array(seed_point) - np.array(map_origin))/map_reso
    x = int(vertex_pose_pixel[0])
    y = int(vertex_pose_pixel[1])
    if image[y,x] != 0:
        return [0,0,0,0]
    height, width = image.shape
    nearest_obs_index = find_nearest_obstacle_position(image, x, y)
    

    if nearest_obs_index[0] < x:
        x1 = nearest_obs_index[0]
        x2 = min(2*x - x1,width)
    else:
        x1 = max(2*x - nearest_obs_index[0],0)
        x2 = nearest_obs_index[0]
    
    if nearest_obs_index[1] < y:
        y1 = nearest_obs_index[1]
        y2 = min(2*y - y1,height)
    else:
        y1 = max(2*y - nearest_obs_index[1],0)
        y2 = nearest_obs_index[1]
    
    if x1 == x:
        y1 += 1
        y2 -= 1
    elif y1 ==y:
        x1 += 1
        x2 -= 1
    else:
        x1 += 1
        y1 += 1
        x1 -= 1
        y2 -= 1
        
    free_space_flag = [True, True, True, True] #up,left,down,right
    while True in free_space_flag:
        if free_space_flag[0]:
            if y1 < 1 or np.any(image[y1-1, x1:x2+1]):
                free_space_flag[0] = False
            else:
                y1 -= 1
        if free_space_flag[1]:
            if x1 < 1 or np.any(image[y1:y2+1, x1-1]):   
                free_space_flag[1] = False
            else:
                x1 -= 1
        if free_space_flag[2]:
            if y2 > height -2 or np.any(image[y2+1, x1:x2+1]):
                free_space_flag[2] = False
            else:
                y2 += 1
        if free_space_flag[3]:
            if x2 > width -2 or np.any(image[y1:y2+1, x2+1]):
                free_space_flag[3] = False
            else:
                x2 += 1
    x1 = x1 * map_reso + map_origin[0]
    x2 = x2 * map_reso + map_origin[0]
    y1 = y1 * map_reso + map_origin[1]
    y2 = y2 * map_reso + map_origin[1]
    return [x1,y1,x2,y2]


def outlier_rejection(input,dis_th = 0.1):
    #input: a list of estimation
    if len(input) < 4:
        return input

    estimated_center = []
    for now_input in input:
        R_map_i = R.from_euler('z', now_input[3][2], degrees=True).as_matrix()
        t_map_i = np.array([now_input[3][0],now_input[3][1],0]).reshape(-1,1)
        T_map_i = np.block([[R_map_i,t_map_i],[np.zeros((1,4))]])
        T_map_i[-1,-1] = 1

        R_nav_i1 = R.from_euler('z', now_input[2][2], degrees=True).as_matrix()
        t_nav_i1 = np.array([now_input[2][0],now_input[2][1],0]).reshape(-1,1)
        T_nav_i1 = np.block([[R_nav_i1,t_nav_i1],[np.zeros((1,4))]])
        T_nav_i1[-1,-1] = 1

        R_i1_i = R.from_euler('z', now_input[4][2], degrees=True).as_matrix()
        t_i1_i = np.array([now_input[4][0],now_input[4][1],0]).reshape(-1,1)
        T_i1_i = np.block([[R_i1_i,t_i1_i],[np.zeros((1,4))]])
        T_i1_i[-1,-1] = 1

        T_nav_map = T_nav_i1 @ T_i1_i @  np.linalg.inv(T_map_i) 
        rot = R.from_matrix(T_nav_map[0:3,0:3]).as_euler('xyz',degrees=True)[2]
        estimated_center.append([T_nav_map[0,-1], T_nav_map[1,-1], rot])

    estimated_center = np.array(estimated_center)
    estimated_center[:,2] /= 4
    kdtree = KDTree(estimated_center)
    K = 2
    maintan_ratio = 0.5

    distances, indices = kdtree.query(estimated_center, k=K)
    nearest_dis = distances[:,1]
    while len (np.where(nearest_dis<dis_th)[0] ) < maintan_ratio * len(input):
        dis_th = dis_th*1.1
    
    good_index = np.where(nearest_dis<dis_th)[0]
    new_input = [input[i] for i in good_index]
    # print("origin number:",len(input), "  final number:", len(new_input))

    return new_input


def change_frame(point_1, T_1_2):
    input_length = len(point_1)
    if input_length==2:
        point_1 = [point_1[0],point_1[1],0]

    R_1_2 = R.from_euler('z', T_1_2[2], degrees=False).as_matrix()
    t_1_2 = np.array([T_1_2[0],T_1_2[1],0]).reshape(-1,1)
    T_1_2 = np.block([[R_1_2,t_1_2],[np.zeros((1,4))]])
    T_1_2[-1,-1] = 1
    
    R_1_point = R.from_euler('z', point_1[2], degrees=False).as_matrix()
    t_1_point = np.array([point_1[0],point_1[1],0]).reshape(-1,1)
    T_1_point = np.block([[R_1_point,t_1_point],[np.zeros((1,4))]])
    T_1_point[-1,-1] = 1

    T_2_point =  np.linalg.inv(T_1_2) @ T_1_point
    rot = R.from_matrix(T_2_point[0:3,0:3]).as_euler('xyz',degrees=False)[2]

    result = [T_2_point[0,-1], T_2_point[1,-1], rot]
    return result[0:input_length]

def change_frame_multi(points_1, T_1_2):

    R_1_2 = R.from_euler('z', T_1_2[2], degrees=False).as_matrix()
    t_1_2 = np.array([T_1_2[0],T_1_2[1],0]).reshape(-1,1)
    T_1_2 = np.block([[R_1_2,t_1_2],[np.zeros((1,4))]])
    T_1_2[-1,-1] = 1
    
    result = []
    for point_1 in points_1:
        input_length = len(point_1)
        if input_length==2:
            point_1 = [point_1[0],point_1[1],0]

        R_1_point = R.from_euler('z', point_1[2], degrees=False).as_matrix()
        t_1_point = np.array([point_1[0],point_1[1],0]).reshape(-1,1)
        T_1_point = np.block([[R_1_point,t_1_point],[np.zeros((1,4))]])
        T_1_point[-1,-1] = 1

        T_2_point =  np.linalg.inv(T_1_2) @ T_1_point
        rot = R.from_matrix(T_2_point[0:3,0:3]).as_euler('xyz',degrees=False)[2]

        result.append([T_2_point[0,-1], T_2_point[1,-1], rot])

    result = np.array(result)
    return result[:,0:input_length]

def change_color(img,ori_color,new_colore):
    new_img = copy.deepcopy(img)
    for ori,new in zip(ori_color,new_colore):
        new_img[img == ori] = new
    
    return new_img

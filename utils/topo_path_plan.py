import numpy as np
from utils.astar import topo_map_path
from utils.easy_map import easy_grid_map
from tqdm import tqdm
import torch
from utils.robot_function import sparse_point_cloud

def free_space_line_map(point1,point2,now_global_map):
    # check whether a line cross a free space
    # point1 and point2 in pixel frame, use UV
    height, width = now_global_map.shape
    #use uv input
    # x1, y1 = point1
    # x2, y2 = point2
    #use rc input
    y1, x1 = point1
    y2, x2 = point2

    distance = max(abs(x2 - x1), abs(y2 - y1))

    step_x = (x2 - x1) / distance
    step_y = (y2 - y1) / distance

    for i in range(int(distance) + 1):
        x = int(x1 + i * step_x)
        y = int(y1 + i * step_y)
        if x < 0 or x >= width or y < 0 or y >= height or now_global_map[y, x] != 0:
            # if now_global_map[y, x] != 255:#排除掉经过unknown的部分
            return False
    return True
# 通用的Bresenham算法
def GenericBresenhamLine(point1,point2,now_global_map):
    height, width = now_global_map.shape
    y1, x1 = point1
    y2, x2 = point2
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    # 根据直线的走势方向，设置变化的单位是正是负
    s1 = 1 if ((x2 - x1) > 0) else -1
    s2 = 1 if ((y2 - y1) > 0) else -1
    # 根据斜率的大小，交换dx和dy，可以理解为变化x轴和y轴使得斜率的绝对值为（0,1）
    boolInterChange = False
    if dy > dx:
        dy, dx = dx, dy
        boolInterChange = True
    # 初始误差
    e = 2 * dy - dx
    x = x1
    y = y1
    for i in range(0, int(dx + 1)):
        if now_global_map[y, x] != 255:
            # if now_global_map[y, x] != 255:#排除掉经过unknown的部分
            return False
    
        if e >= 0:
            # 此时要选择横纵坐标都不同的点，根据斜率的不同，让变化小的一边变化一个单位
            if boolInterChange:
                x += s1
            else:
                y += s2
            e -= 2 * dx
        # 根据斜率的不同，让变化大的方向改变一单位，保证两边的变化小于等于1单位，让直线更加均匀
        if boolInterChange:
            y += s2
        else:
            x += s1
        e += 2 * dy
    return True
def calculate_dis(x):
    sum_x = np.sum(np.square(x),1)#先求对应元素的平方，然后按列相加，得到(n,1)列向量

    dis_mat = sum_x  + (sum_x- 2*np.dot(x,x.T)).T
    
    return np.sqrt(dis_mat)
def obtain_a_sub_graph(adj_list,start,n):
    sub_graph_list = [False for i in range(n)]

    neighbor_list = [start]
    in_group_index = [start]

    while True:
        now = neighbor_list.pop(0)
        if now in adj_list.keys():
            neighbors = adj_list[now]

            for now_pair in neighbors:
                if now_pair[0] not in in_group_index:
                    neighbor_list.append(now_pair[0])
                    in_group_index.append(now_pair[0])
            
        if len(neighbor_list) == 0:
            break

    for now_index in in_group_index:
        sub_graph_list[now_index] = True
        
    return sub_graph_list

def obtain_all_sub_graph(adj_list,n):
    check_list = [False for i in range(n)]
    sub_graph_list = []
    for i in range(n):
        if check_list[i]:
            continue
        now_sub_graph = obtain_a_sub_graph(adj_list,i,n)
        sub_graph_list.append(now_sub_graph)
        for j in range(n):
            if now_sub_graph[j]:
                check_list[j] = True
    return sub_graph_list

def connect_2_sub_graph(node_dis_mat,subg1,subg2):
    min_dis = 1e9
    min_index1 = -1
    min_index2 = -1
    n  = len(subg1)
    for i in range(n):
        if subg1[i] != True:
            continue

        for j in range(n):
            if subg2[j] != True:
                continue
            now_dis = node_dis_mat[i][j]
            if now_dis < min_dis:
                min_dis = now_dis
                min_index1 = i
                min_index2 = j
    return min_index1,min_index2


import os
class calculate_path_on_topo():
    def __init__(self,input_image,vis = False, sample_rate = 50, sparse_dis = 5,dis_mat_path = None,dir_use = False) -> None:
        #sample_rate: every 50 point sample a node
        #use a topological map for planning
        #generate a topological map
        if not dir_use or (not os.path.exists(dis_mat_path)):
            self.device = "cuda"
            self.input_image = input_image

            free_space_num = np.sum(input_image == 255)
            node_num = int(free_space_num/sample_rate)
            # print(node_num)
            easy_map = easy_grid_map(input_image,[0,0],0.2)
            # self.tensor_image = torch.from_numpy(input_image).to(self.device)

            node_point = easy_map.random_points_on_map(node_num)
            # print(node_num)
            node_point = sparse_point_cloud(node_point, sparse_dis)
            node_num = len(node_point)
            # print(node_num)

            if vis:
                easy_map.vis_map_points(node_point,"rc")
            
            #create distance mat of n points
            print(len(node_point))

            node_dir_dis_mat = calculate_dis(node_point)

            adj_list = dict()
            for i in range(node_num - 1):
                for j in range(i+1,node_num):
                    if GenericBresenhamLine(node_point[i],node_point[j],input_image):
                        cost = node_dir_dis_mat[i][j]
                        if i not in adj_list.keys():
                            adj_list[i] = []
                        if j not in adj_list.keys():
                            adj_list[j] = []
                        adj_list[i].append((j,cost))
                        adj_list[j].append((i,cost))
            
            # print("finish all work")

            sub_graph_vector = obtain_all_sub_graph(adj_list,node_num)
            #connect the nearest one

            main_sub_graph = sub_graph_vector[0]
            for i in range(1,len(sub_graph_vector)):
                now_sub_graph = sub_graph_vector[i]
                index1,index2 = connect_2_sub_graph(node_dir_dis_mat,main_sub_graph,now_sub_graph)
                if index1 not in adj_list.keys():
                    adj_list[index1] = []
                if index2 not in adj_list.keys():
                    adj_list[index2] = []
                adj_list[index1].append((index2,node_dir_dis_mat[index1][index2]))
                adj_list[index2].append((index1,node_dir_dis_mat[index1][index2]))

                main_sub_graph = np.logical_or(main_sub_graph,now_sub_graph)
            # print("finish main_sub_graph")
            node_dis = np.zeros((node_num,node_num))

            for i in range(node_num - 1):
                target = [j for j in range(i+1,node_num)]
                topo_map = topo_map_path(adj_list,i, target)
                topo_map.get_path()
                length_list = topo_map.path_length
                for j in range(i+1,node_num):
                    node_dis[i][j] = length_list[j - i - 1]
                    node_dis[j][i] = length_list[j - i - 1]
            
            self.node_point = node_point
            self.node_dis = node_dis

            if not dis_mat_path is None:
                np.savez(dis_mat_path,  node_point = node_point, node_dis = node_dis)
        else:
            dis_mat = np.load(dis_mat_path)
            self.node_point = dis_mat["node_point"]
            self.node_dis = dis_mat["node_dis"]
            self.input_image = input_image
            self.device = "cuda"
    
    def find_suitable_point(self,point):
        #point: 2*1
        free_line_node = []
        free_line_index = []
        for index,now_node in enumerate(self.node_point):
            if abs(now_node[0] - point[0]) + abs(now_node[1] - point[1]) > 20:
                continue
            if GenericBresenhamLine(now_node,point,self.input_image):
                free_line_node.append(now_node)
                free_line_index.append(index)

        if len(free_line_node) == 0:
            #use the nearest point
            min_index = np.argmin(np.sum((self.node_point - point)**2,axis=1))
            free_line_node = [self.node_point[min_index]]
            free_line_index = [min_index]
        return np.array(free_line_node),free_line_index

        
    def path_plan(self,point):
        #point[0]:start, point[1]:end

        # min_index_list = []
        # for now_point in point:
        #     #find minimal point
        #     min_index = np.argmin(np.sum((self.node_point - now_point)**2,axis=1))
        #     min_index_list.append(min_index)
        start_nodes,start_index_list = self.find_suitable_point(point[0])
        end_nodes,end_index_list = self.find_suitable_point(point[1])

        min_dis = 1e9
        for start, start_index in zip(start_nodes,start_index_list):
            for end, end_index in zip(end_nodes,end_index_list):
                now_dis = np.sum((start - point[0])**2)**0.5 + np.sum((end - point[1])**2)**0.5 + self.node_dis[start_index][end_index]
                if now_dis < min_dis:
                    min_dis = now_dis
                    min_start = start
                    min_end = end   

        return min_dis

        
        
        
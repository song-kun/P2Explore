from typing import Optional, Union
import torch
from tqdm import tqdm
from torchvision.utils import make_grid
from PIL import Image
from pathlib2 import Path
import yaml
import cv2
import copy
import scipy.ndimage
from compare_exp.model import UNet
def load_yaml(yml_path: Union[Path, str], encoding="utf-8"):
    if isinstance(yml_path, str):
        yml_path = Path(yml_path)
    with yml_path.open('r', encoding=encoding) as f:
        cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        return cfg



def train_one_epoch(trainer, loader, optimizer, device, epoch):
    trainer.train()
    total_loss, total_num = 0., 0

    with tqdm(loader, dynamic_ncols=True, colour="#ff924a") as data:
        for images, _ in data:
            optimizer.zero_grad()

            x_0 = images.to(device)
            loss = trainer(x_0)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_num += x_0.shape[0]

            data.set_description(f"Epoch: {epoch}")
            data.set_postfix(ordered_dict={
                "train_loss": total_loss / total_num,
            })

    return total_loss / total_num


def save_image(images: torch.Tensor, nrow: int = 8, show: bool = True, path: Optional[str] = None,
               format: Optional[str] = None, to_grayscale: bool = False, **kwargs):
    """
    concat all image into a picture.

    Parameters:
        images: a tensor with shape (batch_size, channels, height, width).
        nrow: decide how many images per row. Default `8`.
        show: whether to display the image after stitching. Default `True`.
        path: the path to save the image. if None (default), will not save image.
        format: image format. You can print the set of available formats by running `python3 -m PIL`.
        to_grayscale: convert PIL image to grayscale version of image. Default `False`.
        **kwargs: other arguments for `torchvision.utils.make_grid`.

    Returns:
        concat image, a tensor with shape (height, width, channels).
    """
    images = images * 0.5 + 0.5
    grid = make_grid(images, nrow=nrow, **kwargs)  # (channels, height, width)
    #  (height, width, channels)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    im = Image.fromarray(grid)
    if to_grayscale:
        im = im.convert(mode="L")
    if path is not None:
        im.save(path, format=format)
    if show:
        im.show()
    return grid


def save_sample_image(images: torch.Tensor, show: bool = True, path: Optional[str] = None,
                      format: Optional[str] = None, to_grayscale: bool = False, **kwargs):
    """
    concat all image including intermediate process into a picture.

    Parameters:
        images: images including intermediate process,
            a tensor with shape (batch_size, sample, channels, height, width).
        show: whether to display the image after stitching. Default `True`.
        path: the path to save the image. if None (default), will not save image.
        format: image format. You can print the set of available formats by running `python3 -m PIL`.
        to_grayscale: convert PIL image to grayscale version of image. Default `False`.
        **kwargs: other arguments for `torchvision.utils.make_grid`.

    Returns:
        concat image, a tensor with shape (height, width, channels).
    """
    images = images * 0.5 + 0.5

    grid = []
    for i in range(images.shape[0]):
        # for each sample in batch, concat all intermediate process images in a row
        t = make_grid(images[i], nrow=images.shape[1], **kwargs)  # (channels, height, width)
        grid.append(t)
    # stack all merged images to a tensor
    grid = torch.stack(grid, dim=0)  # (batch_size, channels, height, width)
    grid = make_grid(grid, nrow=1, **kwargs)  # concat all batch images in a different row, (channels, height, width)
    #  (height, width, channels)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    im = Image.fromarray(grid)
    if to_grayscale:
        im = im.convert(mode="L")
    if path is not None:
        im.save(path, format=format)
    if show:
        im.show()
    return grid

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

def create_cm(y,yp,th_list,y_target_list):
    #y: 0,100,255
    #yp: -1~1
    #th_list: a few th to get the sperated label
    #y_target_list: the target label for each th
    total_len = len(y)
    #create y_label
    change_dict = {}
    for index,now_target in enumerate(y_target_list):
        change_dict[now_target] = index
    
    y_label = np.zeros(total_len,dtype=int)
    for index,now_y in enumerate(y):
        y_label[index] = change_dict[now_y]

        
    yp_label = np.zeros(total_len,dtype=int)
    for index,now_yp in enumerate(yp):
        now_index = 0
        for now_th in th_list:
            if now_yp < now_th:
                break
            now_index += 1
        yp_label[index] = now_index
    
    cm = confusion_matrix(y_label, yp_label)
    result = classification_report(y_label, yp_label,digits=5)

    return cm,result

def vis_cm(confusion_matrix,classes):

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)  #按照像素显示出矩阵
    plt.title('confusion_matrix')#改图名
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=-45)
    plt.yticks(tick_marks, classes)
    classNumber = len(classes)
    
    thresh = confusion_matrix.max() / 2.
    #iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
    #ij配对，遍历矩阵迭代器
    iters = np.reshape([[[i,j] for j in range(classNumber)] for i in range(classNumber)],(confusion_matrix.size,2))
    for i, j in iters:
        plt.text(j, i, '%.5f' % confusion_matrix[i,j],va='center',ha='center')   #显示对应的数字
    
    plt.ylabel('Ture')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.show()

def get_local_map(global_map, next_goal, local_map_size,defailt_value = 255):
    # 计算切片的起始和结束位置
    start_x = max(0, next_goal[0] - local_map_size)
    end_x = min(global_map.shape[0], next_goal[0] + local_map_size)
    start_y = max(0, next_goal[1] - local_map_size)
    end_y = min(global_map.shape[1], next_goal[1] + local_map_size)
    
    # 创建一个新的局部地图数组，初始化为255
    local_map = np.full((2 * local_map_size, 2 * local_map_size), defailt_value, dtype=np.uint8)
    
    # 计算原始图像数据在局部地图数组中的位置
    offset_x_start = max(0, local_map_size - next_goal[0])
    offset_x_end = min(2 * local_map_size, global_map.shape[0] - next_goal[0] + local_map_size)
    offset_y_start = max(0, local_map_size - next_goal[1])
    offset_y_end = min(2 * local_map_size, global_map.shape[1] - next_goal[1] + local_map_size)
    
    # 将原始图像数据拷贝到新的局部地图数组中
    local_map[offset_x_start:offset_x_end, offset_y_start:offset_y_end] = global_map[start_x:end_x, start_y:end_y]
    
    return local_map

def expand_obs(image,obs_index = 0,unknown_index = 100,free_index = 255,kernel_size = 5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    obs_space = (image ==obs_index).astype(np.uint8)
    free_bool = image == free_index

    expanded_obs = cv2.dilate(obs_space, kernel).astype(bool)
    obs = expanded_obs & (~free_bool)
    new_image = copy.deepcopy(image)
    new_image[obs] = obs_index
    return new_image

def th_segmentation(img,th_list,target_value):
    # segment the img into n value
    # for i-th value: it will be larger than th_list[i] and <= th_list[i+1]
    new_img = copy.deepcopy(img)
    for i in range(len(th_list)-1):
        index = np.logical_and(img > th_list[i], img <= th_list[i+1])
        new_img[index] = target_value[i]
    
    return new_img

def to_original_frame(new_map,ori_point,origin_map_size):
    map_size = new_map.shape
    local_map_size = int(map_size[0]//2)

    re_map = np.full(origin_map_size, 100, dtype=np.uint8)

    start_x = max(0, ori_point[0] - local_map_size)
    end_x = min(origin_map_size[0], ori_point[0] + local_map_size)
    start_y = max(0, ori_point[1] - local_map_size)
    end_y = min(origin_map_size[1], ori_point[1] + local_map_size)
    
    
    # 计算原始图像数据在局部地图数组中的位置
    offset_x_start = max(0, local_map_size - ori_point[0])
    offset_x_end = min(2 * local_map_size, origin_map_size[0] - ori_point[0] + local_map_size)
    offset_y_start = max(0, local_map_size - ori_point[1])
    offset_y_end = min(2 * local_map_size, origin_map_size[1] - ori_point[1] + local_map_size)
    
    # 将原始图像数据拷贝到新的局部地图数组中
    re_map[start_x:end_x, start_y:end_y] = new_map[offset_x_start:offset_x_end, offset_y_start:offset_y_end]

    return re_map


def sparse_point_cloud_with_value(data,min_dis,value_map):
    #use norm 1 for dis func
    #select the minimal value in value map
    new_data = []
    for now_data in data:
        add_flag = True
        for index,target_data in enumerate(new_data):
            if abs(target_data[0] - now_data[0]) + abs(target_data[1] - now_data[1])<min_dis:
                add_flag = False
                break
        
        if add_flag:
            new_data.append(now_data)
        else:
            if value_map[now_data[0],now_data[1]] < value_map[target_data[0],target_data[1]]:
                new_data[index] = now_data
    return np.array(new_data)

def extract_door(img,debug = False):
    # 100 is the label of the obstacle
    #extract the place of door

    obs_map = img != 0

    distance_map = scipy.ndimage.distance_transform_edt(obs_map)

    ker_x = np.array([[1, -2, 1]])
    ker_y = np.array([[1], [-2], [1]])
    ker_xy = np.array([[-1, 0,1], [0, 0,0], [1,0,-1]])/4

    # 对图像进行卷积操作
    gradient_xx = scipy.ndimage.convolve(distance_map, ker_x)
    gradient_yy = scipy.ndimage.convolve(distance_map, ker_y)
    gradient_xy = scipy.ndimage.convolve(distance_map, ker_xy)
    det = gradient_xx * gradient_yy - gradient_xy ** 2

    door_index = np.where(det < -0.4)
    door_index_list = np.array(list(zip(door_index[0],door_index[1])))
    door_index_list = sparse_point_cloud_with_value(door_index_list,4,det)
    door_index_list = np.array([index for index in door_index_list if img[index[0],index[1]] == 255]) #in free space

    #room index
    room_index =  np.where(np.logical_and(det >0.8, gradient_xx< -0.5, gradient_yy < -0.5))
    room_index_list = np.array(list(zip(room_index[0],room_index[1])))
    room_index_list = sparse_point_cloud_with_value(room_index_list,8,-det)

    #room index has to be in the free space
    room_index_list = np.array([index for index in room_index_list if img[index[0],index[1]] == 255])


    if debug:
        plt.imshow(img,cmap="gray")
        # plt.imshow(distance_map)
        # plt.scatter(index[1],index[0],2,color='red')
        plt.scatter(door_index_list[:,1],door_index_list[:,0],4)
        plt.scatter(room_index_list[:,1],room_index_list[:,0],4,color = 'red')
        plt.show()
    return door_index_list, room_index_list #index[0] is row, index[1] is col


def flood_fill(grid_map, room_center,fill_map,fill_value):
    #find the connect components in grid_map and fill the fill_map with fill_value
    grid_map = copy.deepcopy(grid_map)
    map_shape = grid_map.shape
    center_value = grid_map[room_center[0], room_center[1]]
    if center_value == 0:
        change_value = 1
    else:
        change_value = 0

    stack = [room_center]
    while len(stack) > 0:
        now_cell = stack.pop()
        if now_cell[0] <0 or now_cell[0]>=map_shape[0] or now_cell[1] <0 or now_cell[1]>=map_shape[1]:
            continue
        if grid_map[now_cell[0], now_cell[1]] == center_value:
            grid_map[now_cell[0], now_cell[1]] = change_value
            fill_map[now_cell[0], now_cell[1]] = fill_value
            stack.append([now_cell[0] + 1, now_cell[1]])
            stack.append([now_cell[0] - 1, now_cell[1]])
            stack.append([now_cell[0], now_cell[1] + 1])
            stack.append([now_cell[0], now_cell[1] - 1])

    return fill_map


def create_segmentation(img,door_point,room_point,debug = False):
    expand_map = copy.deepcopy(img)
    door_size = 2
    for index, now_door in enumerate(door_point):
        expand_map[now_door[0]-door_size:now_door[0]+door_size+1,now_door[1]-door_size:now_door[1]+door_size+1] = 0
    
    #assign the connected door with same index
    connected_dict = {}
    for index, now_door in enumerate(door_point):
        connected_dict[index] = []
        mask1 = np.zeros(img.shape,bool)
        mask1[now_door[0]-door_size:now_door[0]+door_size+1,now_door[1]-door_size:now_door[1]+door_size+1] = 1
        for index2, now_door2 in enumerate(door_point):
            mask2 = np.zeros(img.shape,bool)
            mask2[now_door2[0]-door_size:now_door2[0]+door_size+1,now_door2[1]-door_size:now_door2[1]+door_size+1] = 1
            if np.any(np.logical_and(mask1,mask2)):
                connected_dict[index].append(index2)
        
    selected_index = {}
    for now_index in range(len(door_point)):
        selected_index[now_index] = False

    mask_map = np.zeros(img.shape,dtype=np.int16)
    assign_index = -1
    for now_key in connected_dict.keys():
        value = connected_dict[now_key]
        add_a_value_flag = False
        for now_value in value:
            if not selected_index[now_value]:
                now_door = door_point[now_value]
                mask_map[now_door[0]-door_size:now_door[0]+door_size+1,now_door[1]-door_size:now_door[1]+door_size+1] = assign_index
                selected_index[now_value] = True
                add_a_value_flag = True
        if add_a_value_flag:
            assign_index -= 1

    #assign the color of each room
    for index, now_room in enumerate(room_point):
        if mask_map[now_room[0], now_room[1]] != 0:
            continue
        mask_map = flood_fill(expand_map, now_room, mask_map, index + 1)

    if debug:
        # plt.imshow(expand_map)
        # plt.show()
        show_mask =  mask_map > 0
        alpha_channel = np.where(show_mask, 1.0, 0.0)  # show_mask为True时透明度为0.5，否则为0（完全透明）
        plt.imshow(mask_map,cmap="RdPu",alpha=alpha_channel)
        plt.title("Mask Map Visualization")
        show_mask =  mask_map < 0
        alpha_channel = np.where(show_mask, 1.0, 0.0)  # show_mask为True时透明度为0.5，否则为0（完全透明）
        plt.imshow(-mask_map,cmap="Blues",alpha=alpha_channel)
        plt.title("Mask Map Visualization")
        plt.show()
    
    return mask_map


def componenets_connection_check(img1,img2,com1,com2,connection_type = "4"):
    #检查在img1中，com1是否与img2中的com2相连
    #img1和img2的shape必须要一样
    if connection_type == "8":
        kernel = np.ones((3,3),np.uint8)
    elif connection_type == "4":
        kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
    index1 = img1 == com1
    index1.dtype = np.uint8
    expand1 = cv2.dilate(index1, kernel).astype(bool)
    index2 = img2 == com2
    return np.any(np.logical_and(expand1,index2))

        
class map_predictor():
    def __init__(self,model_path) -> None:
        self.device = "cuda"
        self.origin_map_size = 60 #120 * 120
        self.fig_size = 64
        self.net = UNet(n_channels=1, n_classes=1,bilinear = False).to(self.device)
        state_dict = torch.load(model_path,weights_only=True)
        self.net.load_state_dict(state_dict)
        self.last_predicted_map = None

        
    
    def predict_map(self,now_map,now_points,debug = False):
        self.last_predicted_map = copy.deepcopy(now_map) #如果注释了就用过去所有的预测信息
        for now_point in now_points:
            local_map = get_local_map(now_map,now_point,self.origin_map_size)
            local_map = expand_obs(local_map,obs_index=0,free_index=255,unknown_index=100,kernel_size=3)
            resized_local_map = cv2.resize(local_map,dsize=(self.fig_size,self.fig_size),fx=1,fy=1,interpolation=cv2.INTER_NEAREST)
            resized_local_map = torch.from_numpy(resized_local_map).float().unsqueeze(0).unsqueeze(0).to(self.device)
            #normalize img

            resized_local_map = resized_local_map*2.0/255 - 1.0

            predicted_map = self.net(resized_local_map) #perform the prediction
            predicted_map = predicted_map.cpu().detach().numpy().squeeze()
            predicted_map = (predicted_map+1)/2*255

            predicted_map = th_segmentation(predicted_map, [np.min(predicted_map) - 1,80,120,np.max(predicted_map) +1],  [0,100,255])
            resized_predicted_map = cv2.resize(predicted_map,dsize=(self.origin_map_size*2,self.origin_map_size*2),fx=1,fy=1,interpolation=cv2.INTER_NEAREST)
            #change the image to original position
            resized_predicted_map = to_original_frame(resized_predicted_map,now_point,now_map.shape)

            already_known_index = now_map!=100
            resized_predicted_map[already_known_index] = now_map[already_known_index]

            if self.last_predicted_map is None:
                self.last_predicted_map = resized_predicted_map
            else:
                update_index = resized_predicted_map != 100
                self.last_predicted_map[update_index] = resized_predicted_map[update_index]


        if debug:
            plt.subplot(121)
            plt.imshow(now_map,cmap="gray")
            plt.subplot(122)
            plt.imshow(resized_predicted_map,cmap="gray")
            plt.show()
        
        return self.last_predicted_map

class room():
    def __init__(self,label,has_frontier = False,close=False) -> None:
        self.label = label
        self.has_frontier = has_frontier
        self.close = close # if close, not connected unknown area
        self.priority = 5

class create_room_topo():
    def __init__(self,original_map, predicted_map, predicted_seg,debug = False, room_include_door = False, target = None) -> None:
        #original_map: map without any prediction
        #predicted_map: predicted map
        #predicted_seg: predicted segmentation map, including rooms and doors

        self.original_map = original_map
        self.predicted_map = predicted_map
        self.predicted_seg = predicted_seg

        self.optimal_room_status = [False, 0] #对于一个算出来的最佳房间，表示其状态，第0个值代表room是否是closed，第1个值代表深度
        self.selected_room_label = None #debug'
        self.debug = debug
        self.room_include_door = room_include_door
        self.target = target

        self.closed_room_better_rate = 0.8
        self.single_connnect_room_better_rata = 2

        room_max_index = int(np.max(predicted_seg))
        door_max_index = int(np.max(-predicted_seg))

        #init the room dict
        self.room_dict = {}
        for now_room_index in range(1,room_max_index+1):
            #skip the room with small size
            now_room = predicted_seg == now_room_index
            if np.sum(now_room) >= 20:
                self.room_dict[now_room_index] = room(now_room_index)

        #create the connection between rooms
        self.edge_list = [] #connect two label
        self.door_room_dict = {}

        for now_door_index in range(1,door_max_index+1):
            self.door_room_dict[-now_door_index] = []
            
            connected_room_list = []
            for now_room in self.room_dict.values():
                now_room_index = now_room.label
                if componenets_connection_check(predicted_seg,predicted_seg,now_room_index,-now_door_index):
                    connected_room_list.append(now_room_index)
                    self.door_room_dict[-now_door_index].append(now_room_index)

            for i in range(len(connected_room_list)):
                for j in range(i+1,len(connected_room_list)):
                    self.edge_list.append((connected_room_list[i],connected_room_list[j]))
        
        #create the room dict
        free_space = original_map ==255
        unknown = (original_map == 100).astype(np.uint8)
        expanded_unknown = cv2.dilate(unknown, np.ones((3, 3), np.uint8)).astype(bool)
        frontier_index = np.logical_and(free_space,expanded_unknown)

        unknown_pos = original_map == 100
        for now_room_index in self.room_dict.keys():
            now_room = predicted_seg == now_room_index
            #把门一起加进来判断是否为closed room
            if self.room_include_door:
                for now_door in self.door_room_dict.keys():#room: room with connected doors
                    if now_room_index in self.door_room_dict[now_door]:
                        now_room = np.logical_or(now_room,predicted_seg == now_door)
            # has_frontier = componenets_connection_check(now_room,unknown_pos,1,1,connection_type="8") 
            has_frontier = np.any(np.logical_and(now_room,frontier_index))
            

            closed = 1- componenets_connection_check(predicted_map,predicted_seg,100,now_room_index) 
            self.room_dict[now_room_index].has_frontier = has_frontier
            self.room_dict[now_room_index].close = closed
        
    def create_adj_list(self):
        adj_list = {}
        for now_room in self.room_dict.values():
            adj_list[now_room.label] = []
        for now_edge in self.edge_list:
            adj_list[now_edge[0]].append(now_edge[1]) 
            adj_list[now_edge[1]].append(now_edge[0])
        self.adj_list = adj_list

    def mask_rooms(self,rooms):
        #给定一个list rooms，把这些房间和所有与其联通的门都标识成1
        self.selected_room_label = rooms
        return_flag = np.zeros(self.predicted_seg.shape,dtype=bool)
        for now_robot_room_label in rooms:
            return_flag[self.predicted_seg == now_robot_room_label] =1
            #与这个房间相连的门也作为可选择区域
            if self.room_include_door:
                for now_door in self.door_room_dict.keys():
                    if now_robot_room_label in self.door_room_dict[now_door]:
                        return_flag[self.predicted_seg == now_door] =1
        
        return return_flag

    def select_next_room_with_label(self,now_robot_room_label_list):
        #利用广度有限搜索去搜索可能存在的closed room with frontier
        room_prio_mask = np.ones(self.predicted_seg.shape,dtype=float) * 100

        visited_flag_dict = {}
        for now_room in self.room_dict.values():
            visited_flag_dict[now_room.label] = False
        
        node_queue = []
        #基于当前机器人的位置，构建每个房间的value map
        for i in now_robot_room_label_list:
            node_queue.append((i,0))

        while len(node_queue) > 0:
            now_room_label, now_dis = node_queue.pop(0) #now_dis, distance from the root
            #如果当前房间没有被访问过，设置他的优先级
            if not visited_flag_dict[now_room_label]:
                visited_flag_dict[now_room_label] = True
                now_room = self.room_dict[now_room_label]
                now_prio = now_dis  
                if now_room.has_frontier and now_room.close:
                    now_prio -= self.closed_room_better_rate
                if len(self.adj_list[now_room_label]) == 1:
                    now_prio -= self.single_connnect_room_better_rata
                room_prio_mask[self.predicted_seg == now_room_label] = now_prio
                self.room_dict[now_room_label].priority = now_prio
                # if self.debug:
                #     print("room",now_room_label,"priority",now_prio)

            #add the child node
            for child_room_label in self.adj_list[now_room_label]:
                if not visited_flag_dict[child_room_label]:
                    node_queue.append((child_room_label, now_dis+1))
        
        #处理一下door附近的增益，door附近的增益为与其相邻房间的增益的平均
        for now_door in self.door_room_dict.keys():
            prio_list = []
            for now_room_label in self.door_room_dict[now_door]:
                now_room_prio = self.room_dict[now_room_label].priority
                prio_list.append(now_room_prio)
            if len(prio_list) == 0:
                room_prio_mask[self.predicted_seg == now_door] = 3
            else:
                room_prio_mask[self.predicted_seg == now_door] = np.mean(prio_list)
        
        #还可能存在一些位置，没有被认为是door或者room，但是存在前沿点，这些点也应该被认为是有价值的
        unassigned_index = np.logical_and(self.predicted_map == 255, room_prio_mask > 99)
        #对于这些位置先简单处理一下，直接给成3
        room_prio_mask[unassigned_index] = 3
        return room_prio_mask


    def selec_next_room(self,now_robot_pose):
        #now_robot_pose: [row,col]
        now_label = self.predicted_seg[now_robot_pose[0],now_robot_pose[1]]
        
        self.create_adj_list()

        if now_label ==0:
            return None
        if now_label > 0:
            now_robot_room_label = [now_label]
            if not now_label in self.room_dict.keys(): #有些位置可能被标记为房间，但是在room topo中没有显示
                return None 
        else:
            #robot in the door
            now_robot_room_label = self.door_room_dict[now_label]
        if self.debug:
            print("now_robot_room_label",now_robot_room_label)
        selected_room_mask = self.select_next_room_with_label(now_robot_room_label)

        return selected_room_mask

    def select_optimal_room(self,closed_status_list,dis_list):
        #给定两个list，选择最优的房间
        #closed_status_list: 1 for closed, 0 for not closed
        #dis_list: the distance from the room of robot
        #首先比较是否有closed room，如果有closed room，选择closed room; 其次选择最近的房间
        #如果有多个closed room，选择最近的closed room
        if True in closed_status_list:
            min_dis = 1e9
            for index, now_closed_status in enumerate(closed_status_list):
                if now_closed_status:
                    if dis_list[index] < min_dis:
                        min_dis = dis_list[index]
                        optimal_index = index
            return optimal_index
        else:
            min_dis = 1e9
            for index, now_dis in enumerate(dis_list):
                if now_dis < min_dis:
                    min_dis = now_dis
                    optimal_index = index
            return optimal_index
    
    def vis_topo(self,robot_pose = None,save_path = None):
        original_map = copy.deepcopy(self.floor_plan)
        original_map[original_map == 100] = 150
        room_shape = np.max(original_map.shape)
        fig, axes = plt.subplots(1,1, figsize=(self.original_map.shape[1]//30, original_map.shape[0]//30))
        axes.imshow(original_map,cmap="gray",zorder = 1)

        show_mask =  self.predicted_seg > 0
        alpha_channel = np.where(show_mask, 0.5, 0.0)  # show_mask为True时透明度为0.5，否则为0（完全透明）
        axes.imshow(self.predicted_seg,cmap="RdPu",alpha=alpha_channel,zorder = 2)
        show_mask =  self.predicted_seg < 0
        alpha_channel = np.where(show_mask, 0.5, 0.0)  # show_mask为True时透明度为0.5，否则为0（完全透明）
        axes.imshow(-self.predicted_seg,cmap="Blues",alpha=alpha_channel,zorder = 2)
        
        #vis node
        room_center_dict = {}
        vis_room_center_list = [[],[],[],[]]
        for now_room in self.room_dict.values():
            now_room_pos = self.predicted_seg == now_room.label
            if not np.any(now_room_pos):
                continue
            now_center = np.mean(np.argwhere(now_room_pos),axis=0)
            room_center_dict[now_room.label] = now_center
            if now_room.has_frontier and now_room.close:
                vis_room_center_list[0].append(now_center)
                # plt.scatter(now_center[1],now_center[0],70,color= np.array([192,255,62])/255, marker='o',zorder = 5,label = "1")
            elif now_room.has_frontier and (not now_room.close):
                vis_room_center_list[1].append(now_center)
                # plt.scatter(now_center[1],now_center[0],70,color= np.array([131,139,139])/255, marker='o',zorder = 5)
            elif (not now_room.has_frontier) and (now_room.close):
                vis_room_center_list[2].append(now_center)
                # plt.scatter(now_center[1],now_center[0],70,color= np.array([139,71,38])/255, marker='o',zorder = 5)
            else:
                vis_room_center_list[3].append(now_center)
            plt.text(now_center[1]+3,now_center[0],str(now_room.label),color = "black",fontsize=8)

        # for now_door in self.door_room_dict.keys():
        #     now_door_pos = self.predicted_seg == now_door
        #     if not np.any(now_door_pos):
        #         continue
        #     now_center = np.mean(np.argwhere(now_door_pos),axis=0)
        #     plt.text(now_center[1],now_center[0],str(now_door),color = "yellow",fontsize=8)



        
        vis_room_center_list = [np.array(i).reshape((-1,2)) for i in vis_room_center_list]
        plt.scatter(vis_room_center_list[0][:,1],vis_room_center_list[0][:,0],7000//room_shape,color= np.array([192,255,62])/255, marker='o',zorder = 5,label = "closed room with frontiers")
        plt.scatter(vis_room_center_list[1][:,1],vis_room_center_list[1][:,0],7000//room_shape,color= np.array([131,139,139])/255, marker='o',zorder = 5,label = "not closed")
        plt.scatter(vis_room_center_list[2][:,1],vis_room_center_list[2][:,0],7000//room_shape,color= np.array([139,71,38])/255, marker='o',zorder = 5,label = "fully explored")
        plt.scatter(vis_room_center_list[3][:,1],vis_room_center_list[3][:,0],7000//room_shape,color= np.array([165, 42, 42])/255, marker='o',zorder = 5,label = "predicted")

        if not (robot_pose is None):
            plt.scatter(robot_pose[1],robot_pose[0],150,color= np.array([255,215,0])/255, marker='*',zorder = 6,label = "Robot Pose")
        
        #vis dege
        for now_edge in self.edge_list:
            edge1 = now_edge[0]
            edge2 = now_edge[1]
            center1 = room_center_dict[edge1]
            center2 = room_center_dict[edge2]
            plt.plot([center1[1],center2[1]],[center1[0],center2[0]],color = [0,0,0],zorder = 4)
        
        # fig.legend()
        axes.axis("off")
        if save_path is not None:
            plt.savefig(save_path,dpi=300)
            plt.close()
        else:
            plt.show()
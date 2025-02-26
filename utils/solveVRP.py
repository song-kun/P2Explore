import os
import numpy as np

class VRP_solver:
    def __init__(self,points,robot_pose):#输入当前机器人，其他机器人的id list
        self.M_large = -99
        self.points = points
        self.robot_pose = robot_pose

        self.n_points = -1
        self.n_robot = -1
        self.max_iter = 100
        self.scale = 1

    def distance_function(self,pose1,pose2):
        #建议重载一下这个函数
        return np.linalg.norm(pose1 - pose2)

    def global_path_to_each_robot(self,path,cosmat):
        n_robot = self.n_robot
        n_point = self.n_points
        problem_dim = n_robot + n_point + 1

        #return a dict
        final_path = {}
        path_length = {}
        for i in range(n_robot):
            final_path[i+1] = []
            path_length[i+1] = 0
        
        now_robot = -1

        for now_path_point in path:
            if now_path_point <2 or now_path_point>problem_dim:
                now_robot = -1
            else:
                if now_robot == -1:
                    now_robot = now_path_point
                else:
                    try:
                        #有可能path第一个不是机器人的位置
                        #例如：对于一个机器人情况，正确结果应该为[1,2,.....]
                        #但是可能出现[1,5,....]，这种情况下会报错
                        final_path[now_robot-1].append(now_path_point - 1 - n_robot)
                    except:
                        pass
        
        if len(final_path[1]) == 0:
            final_path[1] = [1]
        for now_robot in final_path.keys():
            now_path = final_path[now_robot]
            if len(now_path) == 0:
                continue
            path_length[now_robot] += cosmat[now_robot,now_path[0]+self.n_robot]
            for i in range(len(now_path)-1):
                path_length[now_robot] += cosmat[now_path[i]+self.n_robot,now_path[i+1]+self.n_robot]
        for now_robot in path_length.keys():#去掉scale
            path_length[now_robot]/=self.scale

        return final_path,path_length



    def writeTSPLIBfile_FE(self,fname_tsp,CostMatrix,user_comment,pwd,tsplib_dir,problemType,n_robot):
        dims_tsp = len(CostMatrix)
        name_line = 'NAME : ' + fname_tsp + '\n'
        comment_line = 'COMMENT : ' + user_comment + '\n'
        tsp_line = 'TYPE : ' + problemType + '\n'
        vehicle_line = 'VEHICLES :' + str(n_robot) + '\n'
        dimension_line = 'DIMENSION : ' + str(dims_tsp) + '\n'
        edge_weight_type_line = 'EDGE_WEIGHT_TYPE : ' + 'EXPLICIT' + '\n' # explicit only
        edge_weight_format_line = 'EDGE_WEIGHT_FORMAT: ' + 'FULL_MATRIX' + '\n'
        display_data_type_line ='DISPLAY_DATA_TYPE: ' + 'NO_DISPLAY' + '\n' # 'NO_DISPLAY'
        edge_weight_section_line = 'EDGE_WEIGHT_SECTION' + '\n'
        eof_line = 'EOF\n'
        Cost_Matrix_STRline = []
        for i in range(0,dims_tsp):
            cost_matrix_strline = ''
            for j in range(0,dims_tsp-1):
                cost_matrix_strline = cost_matrix_strline + str(int(CostMatrix[i][j])) + ' '

            j = dims_tsp-1
            cost_matrix_strline = cost_matrix_strline + str(int(CostMatrix[i][j]))
            cost_matrix_strline = cost_matrix_strline + '\n'
            Cost_Matrix_STRline.append(cost_matrix_strline)
        
        fileID = open((pwd + tsplib_dir + fname_tsp + '.tsp'), "w")
        fileID.write(name_line)
        fileID.write(comment_line)
        fileID.write(tsp_line)
        fileID.write(vehicle_line)
        fileID.write(dimension_line)
        fileID.write(edge_weight_type_line)
        fileID.write(edge_weight_format_line)
        fileID.write(edge_weight_section_line)
        for i in range(0,len(Cost_Matrix_STRline)):
            fileID.write(Cost_Matrix_STRline[i])
        
        fileID.write("DEPOT_SECTION\n1\n-1\n")
        fileID.write(eof_line)
        fileID.close()

        fileID2 = open((pwd + tsplib_dir + fname_tsp + '.par'), "w")

        problem_file_line = 'PROBLEM_FILE = ' + pwd + tsplib_dir + fname_tsp + '.tsp' + '\n' # remove pwd + tsplib_dir
        optimum_line = f'OPTIMUM {self.M_large*self.n_robot*self.scale}' + '\n'
        move_type_line = 'MOVE_TYPE = 5' + '\n'
        patching_c_line = 'PATCHING_C = 3' + '\n'
        patching_a_line = 'PATCHING_A = 2' + '\n'
        runs_line = f'RUNS = {self.max_iter}' + '\n'
        object_line = 'MTSP_OBJECTIVE = MINMAX' + '\n'
        tour_file_line = 'TOUR_FILE = ' + pwd + tsplib_dir + fname_tsp + '.txt' + '\n'

        fileID2.write(problem_file_line)
        fileID2.write(optimum_line)
        fileID2.write(move_type_line)
        fileID2.write(patching_c_line)
        fileID2.write(patching_a_line)
        fileID2.write(runs_line)
        fileID2.write(tour_file_line)
        fileID2.write(object_line)
        fileID2.close()
        return fileID, fileID2

    def solveVRP(self,robot_frontier_mat = None, frontier_mat = None, fname_tsp='TSP_file',problemType='ATSP'):
        #CostMatrix[i,j]: the cost from node i to node j
        #预处理CostMatrix
        if robot_frontier_mat is None and frontier_mat is None:
            CostMatrix = self.create_distance_mat(self.points,self.robot_pose)
            self.n_robot = len(self.robot_pose)
            self.n_points = len(self.points)
            
        else:
            #给定从机器人到前沿点代价和前沿点之间距离代价矩阵情况
            self.n_robot = robot_frontier_mat.shape[0]
            self.n_points = robot_frontier_mat.shape[1]

            CostMatrix = self.create_distance_mat_from_2mat(robot_frontier_mat,frontier_mat)
        CostMatrix = CostMatrix*self.scale
        #None frontier
        if len(CostMatrix)==2:
            return {0:[]}, 0

        user_comment = "a comment by the user"
        # Change these directories based on where you have 
        # a compiled executable of the LKH TSP Solver
        lkh_dir = '/'
        tsplib_dir = '/'
        lkh_cmd = 'LKH'
        pwd = os.path.dirname(os.path.abspath(__file__))
        fileID1,fileID2 = self.writeTSPLIBfile_FE(fname_tsp,CostMatrix,user_comment,pwd,tsplib_dir,problemType,self.n_robot)
        run_lkh_cmd =  pwd + lkh_dir  + lkh_cmd + ' ' + pwd + tsplib_dir + fname_tsp + '.par'
        os.system(f"{run_lkh_cmd} > /dev/null 2>&1")
        # os.system(f"{run_lkh_cmd}")
        #准备读取文件
        # 打开文件
        file = open(os.path.join(pwd , fname_tsp + '.txt'), 'r')  # 'example.txt' 是文件的路径和名称，'r' 表示以只读模式打开文件
        # 读取文件内容
        content = file.read()
        # 关闭文件
        file.close()
        # 打印文件内容
        index = content.find('TOUR_SECTION')
        index_EOF = content.find('EOF')
        # 从 TOUR_SECTION 后面截取字符串
        tour_section = content[index + len('TOUR_SECTION'):index_EOF].strip()
        # 将字符串按行分割，并忽略空行
        lines = tour_section.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        # 提取数字
        numbers = [int(line) for line in lines]
        #提取距离
        # length_index = content.find('COMMENT : Length =')
        # length = int(content[length_index + len('COMMENT : Length ='):].strip().split('\n')[0])
        # path_length = length/self.scale - self.M_large * n_robot
        

        rm_sol_cmd = 'rm' + ' ' + pwd + tsplib_dir+ fname_tsp + '.txt'
        os.system(rm_sol_cmd) 
        rm_sol_cmd = 'rm' + ' ' + pwd + tsplib_dir+ fname_tsp + '.par'
        os.system(rm_sol_cmd) 
        rm_sol_cmd = 'rm' + ' ' + pwd + tsplib_dir+ fname_tsp + '.tsp'
        os.system(rm_sol_cmd) 

        result_path,path_length = self.global_path_to_each_robot(numbers,CostMatrix)
        return result_path,path_length


    def create_distance_mat(self,points,robot_pose):
        n_point = len(points)
        n_robot = len(robot_pose)

        CostMatrix = np.zeros((n_point + n_robot + 1,n_point + n_robot + 1,))

        M_inf = self.M_large*np.ones(n_robot)
        CostMatrix[0,1:n_robot+1] = M_inf
        C_robot_to_frontier = np.zeros((n_robot,n_point))
        C_frontier_to_robot = np.zeros((n_point,n_robot))
        C_frontier_to_frontier = np.zeros((n_point,n_point))
        for i in range(n_robot):
            for j in range(n_point):
                C_robot_to_frontier[i,j] = self.distance_function(robot_pose[i],points[j])

        for i in range(n_point):
            for j in range(n_robot):
                C_frontier_to_robot[i,j] = self.distance_function(points[i],robot_pose[j])

        for i in range(n_point):
            for j in range(n_point):
                C_frontier_to_frontier[i,j] = self.distance_function(points[i],points[j])
        
        CostMatrix[1:n_robot+1,1+n_robot:] = C_robot_to_frontier
        CostMatrix[1+n_robot:,1:n_robot+1] = C_frontier_to_robot
        CostMatrix[1+n_robot:,1+n_robot:] = C_frontier_to_frontier

        return CostMatrix

    def create_distance_mat_from_2mat(self,C_robot_to_frontier,C_frontier_to_frontier):
        n_point = self.n_points
        n_robot = self.n_robot

        CostMatrix = np.zeros((n_point + n_robot + 1,n_point + n_robot + 1,))

        M_inf = self.M_large*np.ones(n_robot)
        CostMatrix[0,1:n_robot+1] = M_inf
        
        CostMatrix[1:n_robot+1,1+n_robot:] = C_robot_to_frontier
        CostMatrix[1+n_robot:,1:n_robot+1] = C_robot_to_frontier.T
        CostMatrix[1+n_robot:,1+n_robot:] = C_frontier_to_frontier

        return CostMatrix


if __name__ == '__main__':
    #利用贪心算法求解mTSP，每次的前沿点都被当前最优的选择处理
    import matplotlib.pyplot as plt
    import time
    num_points = 10
    # multi robot
    n_robot = 1
    robots_pose = np.random.random((n_robot,2))
    rand_points = np.random.random((num_points,2))
    points = rand_points

    start = time.time()
   
    nowSolver = VRP_solver(points,robots_pose)
    visiting_order,length = nowSolver.solveVRP('TSP_file','ATSP')
    end = time.time()
    print('visit order: ', visiting_order)
    print('total length of path:', length)
    print(f"{num_points} points, {len(robots_pose)} robot, used time: {end-start}")

    for i in range(n_robot):
        plt.scatter(robots_pose[i,0],robots_pose[i,1],s=2000,marker='*')
    
    for keys in visiting_order.keys():
        now_value = visiting_order[keys]
        plt.scatter(robots_pose[keys-1,0],robots_pose[keys-1,1],s=2000,marker='*')
        if len(now_value)==0:
            continue
        plt.plot([robots_pose[keys-1,0],points[now_value[0]-1,0]],[robots_pose[keys-1,1],points[now_value[0]-1,1]],'o-', linewidth=2)
        for i in range(len(now_value)-1):
            plt.plot([points[now_value[i]-1,0],points[now_value[i+1]-1,0]],[points[now_value[i]-1,1],points[now_value[i+1]-1,1]],'o-', linewidth=2)

    plt.legend(['robot1','robot2'])
    plt.axis('equal')
    plt.show()
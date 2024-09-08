
# Make sure to have the server side running in CoppeliaSim:
# in a child script of a CoppeliaSim scene, add following command
# to be executed just once, at simulation start:
#
# simRemoteApi.start(19999)
#
# then start simulation, and run this program.
#
# IMPORTANT: for each successful call to simxStart, there
# should be a corresponding call to simxFinish at the end!
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
def euclid(x1,x2):
    return np.sqrt((x1[0]-x2[0])**2 + (x1[1] -x2[1])**2)
def manhattan(x1,x2):
    return np.abs(x1[0]- x2[0]) + np.abs(x1[1]- x2[1])

def Astar(begin, end, map,heuristic):

    #khởi tạo mảng giá trị
    parent = np.zeros([map.shape[0],map.shape[1],2],dtype='int') #mảng cha, dùng để backtrack đường đi khi di chuyển được tới đích
    weight = np.ones([map.shape[0],map.shape[1]]) *9999 #trọng số, quyết định xem node nào được chọn tiếp theo
    visited = np.zeros([map.shape[0],map.shape[1]]) # mảng lưu những tọa độ đã được ghé thăm

    # khơi tạo giá trị tại vị trí bắt đầu
    # không có tuple nó sẽ trả về giá trị nguyên 2 hàng
    print(tuple(begin),begin)
    visited[tuple(begin)] = 1 #mặc định tọa độ bắt đầu đã được ghé thăm
    weight[tuple(begin)] = 0 #trọng số tại điểm bắt đầu luôn là một

    # 8 hướng robot có thể đi
    move_set = np.array([
        [0,1],
        [1,0],
        [-1,0],
        [0,-1],
        [1,1],
        [1,-1],
        [-1,1],
        [-1,-1]
    ])

    # lưu những tọa độ trong danh sách visit vào một mảng
    pos = begin
    pos_list = np.empty((0,2),dtype='int')
    pos_list = np.vstack((pos_list,pos))

    #bắt đầu vòng lặp, mảng rỗng thì nghỉ, tức là thể ghé thăm thêm ô nào khác
    while len(pos_list) != 0:
        for move in move_set:
            temp_pos = move + pos
            # check từng vị trí hàng xóm
            if 0 <= temp_pos[0] <= map.shape[0]-1 \
                    and 0 <= temp_pos[1] <= map.shape[1]-1: #check điều kiện biên

                # check điều kiện vật cản và đã đi qua,
                # viết xuống dưới như này vì điều kiện biên phải được ưu tiên kiếm tra trước
                if visited[tuple(temp_pos)] == 0 \
                        and map[tuple(temp_pos)] == 0:

                    # cost di chuyển từ begin đến vị trí hiệm tại
                    f = euclid(pos,temp_pos)

                    # h là cost ước lượng từ vị trí hiện tại đến đích,
                    # cách ước lượng có thể dùng euclid, manhattan, và diogonal distance

                    # sử dụng euclid
                    if heuristic == "euclid":
                        h = euclid(temp_pos,end)

                    # sử dụng manhattan
                    elif heuristic == "manhattan":
                        h = manhattan(temp_pos,end)
                    #còn cái cuối mà lười quá, euclid đủ dùng :)

                    #đoạn này giống dijkstra, về cơ bản trọng số cũ mà bé hơn tức là đường đi hiện tại đang xét không ngon bằng
                    # -> bỏ qua
                    # ngược lại bé hơn tức là đi đường mới xét này sẽ cho khoảng cách ngắn nhất tính tới thời điểm hiện tại
                    # -> cập nhật danh sách những hàng xóm khả dụng, bố mẹ của điểm hiện tại, và mảng visit để lần sau
                    if weight[tuple(temp_pos)] > weight[tuple(pos)] + f + h:
                        weight[tuple(temp_pos)] = weight[tuple(pos)] + f + h
                        parent[tuple(temp_pos)] = pos
                        pos_list = np.vstack((pos_list,temp_pos))
        visited[tuple(pos)] = 1

        index = np.where((pos_list == pos).all(axis=1))[0]
        pos_list = np.delete(pos_list,index,axis  = 0)

        min_weight = 996699
        for position in pos_list:
            if weight[tuple(position)] < min_weight:
                min_weight = weight[tuple(position)]
                min_pos = position
        pos = np.array(min_pos)
        if  np.array_equal(pos,end):
            break
    res = np.empty((0,2),dtype = 'int')
    if weight[tuple(end)] != 9999:
        x = np.array(end,dtype = 'int')
        res = np.vstack((res,x))
        while not np.array_equal(x,begin):
            x = np.array(parent[tuple(x)])
            res = np.vstack((res, x))
    #np.savetxt("data.csv",weight,delimiter=',')

    return np.flip(res, 0),visited

def plot_map(original_map,path,visit):
    map = np.zeros((original_map.shape))

    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if visit[i][j] == 1:
                map[i][j] = 1

            if original_map[i][j] == 1:

                map[i][j] = 2
    for value in path:
        map[tuple(value)] = 3
    map[tuple(begin)] = 4
    map[tuple(end)] = 4

    cmap = colors.ListedColormap(['white', 'yellow', 'black', 'red','blue'])
    bounds = [0, 1, 2, 3, 4,5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.figure(figsize=(8, 8))
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.title("Mô phỏng đường đi bằng A*")
    plt.pcolor(map, cmap=cmap, edgecolors='gray', linewidths=1.5,norm = norm)
    plt.show()



def cope_to_py_pos(x):
    res = np.array([0,0],dtype= 'int')
    res[0] = 10 - x[1]*2
    res[1] = 10 + x[0] * 2
    return res

def py_to_cope_pos(x):
    res = np.array([0,0],dtype= 'float')
    res[0] = (x[1] - 10)/2
    res[1] = (10 - x[0])/2
    return res






def move3goal(clientID,left_motor,right_motor,robotHandle1,robotHandle2,goal,total_error,last_error,index):
    d_m2g = 0.15
    a_m2g = 70
    R = 0.03
    L = 0.1165
    max_speed = 5
    Kp = 200
    Ki = 30
    Kd = 50
    dt = 0.05
    x, robotPosition = sim.simxGetObjectPosition(clientID,robotHandle1,-1,sim.simx_opmode_buffer)
    robotPosition = np.array(robotPosition)
    x, robotHeading = sim.simxGetObjectOrientation(clientID,robotHandle2,-1,sim.simx_opmode_buffer)
    robotHeading = np.array(robotHeading)

    x, arr1, x, x, x = sim.simxReadProximitySensor(clientID, front_handle, sim.simx_opmode_buffer)
    if arr1 == 1 and index != 1:
        flag = True
        sim.simxSetJointTargetVelocity(clientID, left_motor, 0, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetVelocity(clientID, right_motor, 0, sim.simx_opmode_oneshot)
        return flag
    #tinh v
    v_m2g = goal - robotPosition[0:2]

    dis = np.sqrt(v_m2g[0]**2 + v_m2g[1]**2)

    nom = np.linalg.norm(v_m2g)
    if nom != 0:
        v_m2g = a_m2g * v_m2g / nom
    if dis < d_m2g:
        flag = True

    else:
        flag = False

    desired_orientation = np.arctan2(v_m2g[1],v_m2g[0])
    error_angle = desired_orientation - robotHeading[2]

    if abs(error_angle) > np.pi:
        if error_angle < 0:
            error_angle = error_angle + 2*np.pi
        else:
            error_angle = error_angle - 2*np.pi

    total_error += error_angle
    omega = Kp * error_angle + Ki* (error_angle - last_error)*0.05 +Kd*total_error/0.05
    #omega = Kp *error_angle
    last_error = error_angle
    v = np.linalg.norm(v_m2g)
    vr = (2 * v + omega * L) / 2 * R
    vl = (2 * v - omega * L) / 2 * R

    if vl > max_speed:
        vl = max_speed
    elif vl < -max_speed:
        vl = -max_speed

    if vr > max_speed:
        vr = max_speed
    elif vr < -max_speed:
        vr = -max_speed

    if not flag:
        sim.simxSetJointTargetVelocity(clientID,left_motor,vl,sim.simx_opmode_oneshot)
        sim.simxSetJointTargetVelocity(clientID, right_motor, vr, sim.simx_opmode_oneshot)
    else:
        sim.simxSetJointTargetVelocity(clientID, left_motor, 0, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetVelocity(clientID, right_motor, 0, sim.simx_opmode_oneshot)
        return flag


try:
    import sim
except:
    print ('--------------------------------------------------------------')
    print ('"sim.py" could not be imported. This means very probably that')
    print ('either "sim.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "sim.py"')
    print ('--------------------------------------------------------------')
    print ('')

import time

print ('Program started')
sim.simxFinish(-1) # just in case, close all opened connections
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim
if clientID!=-1:
    print ('Connected to remote API server')

    x, left_motor = sim.simxGetObjectHandle(clientID,'motor_left', sim.simx_opmode_blocking)
    x, right_motor = sim.simxGetObjectHandle(clientID,'motor_right', sim.simx_opmode_blocking)
    x, robotHandle1 = sim.simxGetObjectHandle(clientID, "GPS",sim.simx_opmode_blocking)
    x, robotHandle2 = sim.simxGetObjectHandle(clientID, "GyroSensor", sim.simx_opmode_blocking)
    x, front_handle = sim.simxGetObjectHandle(clientID, 'front_prox', sim.simx_opmode_blocking)
    x, arr1, e, q, w = sim.simxReadProximitySensor(clientID, front_handle, sim.simx_opmode_streaming)
    x, Orierobot = sim.simxGetObjectOrientation(clientID, robotHandle2,-1,sim.simx_opmode_streaming)
    x, Posirobot = sim.simxGetObjectPosition(clientID, robotHandle1, -1, sim.simx_opmode_streaming)
    total_error = 0
    last_error = 0
    map = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],

            [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],

            [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],

            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],

            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],

            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],

            [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],

            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],

            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],

            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],

            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],

            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],

            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],

            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],

            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],

            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],

            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],

            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],

            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],

            [1, 1, 1, 1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    begin = np.array([-4, -4])
    end = np.array([3.5, -4])
    begin = cope_to_py_pos(begin)
    end = cope_to_py_pos(end)
    while 1:
        path, visit = Astar(begin, end, map, heuristic="euclid")
        plot_map(map, path, visit)
        print(path)
        for i in range(1,len(path)):
            flag = False
            while 1:
                goal = py_to_cope_pos(path[i])
                print(goal)
                a = move3goal(clientID,left_motor,right_motor,robotHandle1,robotHandle2,goal,total_error,last_error,i)
                if a:
                    x, arr1, x, x, x = sim.simxReadProximitySensor(clientID, front_handle, sim.simx_opmode_buffer)
                    if arr1 == 1 and i != 1:
                        flag = True
                    break
            if flag:
                begin = np.array(path[i-1])
                print(begin)
                map[tuple(path[i])] = 1
                break
        if not flag:
            break

else:
    print ('Failed connecting to remote API server')
print ('Program ended')
import numpy as np
import csv
import g2o
from optimizer import PoseGraphOptimization
import icp
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
    
def main():
    plt_map_no_op = True   # plot initial map
    plt_odom_no_op = True  # plot initial odometry
    plt_map_op = True       # plot optimized map
    plt_odom_op = True      # plot optimized odometry
    use_info = False        # using covariance information, else identity matrix
    ############################################
    #                                          #
    #          DATA PROCESSING PART            #
    #                                          #
    ############################################
    
    # READ POSE DATA FROM CSV. posedata = [pose0,poes1,...]
    # pose0 = numpy matrix 4x4 (SE3)
    posedata = []
    with open('data/pose.csv', 'r') as csvfile:
        readcsv = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in readcsv:
            pose = []
            for i in range(len(row)):
                pose.append(float(row[i]))
            r = R.from_quat([pose[3],pose[4],pose[5],pose[6]]).as_dcm()
            t = np.array([[pose[0]],[pose[1]],[pose[2]]])
            RT = np.vstack((np.hstack((r,t)),np.array([0,0,0,1])))
            posedata.append(RT)

    # READ LIDAR DATA FROM CSV. lidardata = [lidar0,lidar1,...]
    # lidar0 = [[x0,x1,x2,x3,...],[y0,y1,y2,y3,...],[0,0,0,0,...],[1,1,1,1,...]] (2D in 3D! : Numpy matrix 4xn)
    lidardata = []
    with open('data/lidar.csv', 'r') as csvfile:
        readcsv = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in readcsv:
            x = []
            y = []
            pointNum = len(row) / 2
            for i in range(pointNum):
                x.append(float(row[2 * i])+0.30)
                y.append(row[2 * i + 1])
            lidardata.append(np.array([x, y, np.zeros(len(x)),np.ones(len(x))]).astype(np.float32))
            
            

    ############################################
    #                                          #
    #      SLAM MAIN PART (ASSIGNMENTS)        #
    #                                          #
    ############################################

    # nodes = [node0, node1, ...]
    # node0 = [pose0,lidar0,posediff with node before]
    nodes = [[posedata[0],lidardata[0],np.eye(4)]]
    
    for i in range(1,len(posedata)):
    ####################################################
    # ASSIGNMENTS 1 : CALCULATE POSE DIFF & MAKE NODE  #
    #                                                  #
    # Create nodes at distance or angle intervals.     #
    #                                                  #
    # POSEDIFF : 4x4 numpy matrix of pose difference   #
    # POSE BEFORE : LAST NODE'S POSE ( nodes[-1][0] )  #
    # POSE NOW    : CURRENT POSE     ( posedata[i]  )  #
    ####################################################
        poseDiff =  np.dot(np.linalg.inv(nodes[-1][0]), posedata[i]) # Calculate pose diff in 4x4 matrix
        distDiff =  np.linalg.norm(np.transpose(poseDiff)[3][:3], 2) #Calculate euclidean distance between two node (using posediff)
        yawDiff  = R.from_dcm(poseDiff[0:3,0:3]).as_euler('zyx')[0] # Robot is in 2D in this lab, so just use Yaw angle
        # If enough distance(0.1[m]) or angle(30[deg]) difference, create node
        if (distDiff > 0.1 or abs(yawDiff)/3.141592*180 > 30):
            nodes.append([posedata[i],lidardata[i],poseDiff])


    #############################################################################
    # ASSIGNMENTS 2 : ADD VERTEX AND ODOMETRY EDGE                              #
    #                                                                           #
    # Add vertex of each node and add odometry constraint edge                  #
    # to optimizer.                                                             #
    # FUNCTION1: optimizer.add_vertex(index, g2o.Isometry3d(pose),fixed)        #
    #            index : int, pose : numpy 4x4 mat, fixed : boolean             #
    # FUNCTION2: optimizer.add_edge([src,dst], g2o.Isometry3d(diff),information)#
    #            src: source index, dst: destination index                      #
    #            diff: diff mat (4x4) between source and destination            #   
    #            information: 6x6 numpy matrix of information                   #
    # TIP : You can set simple identity matrix for information.                 #
    #       It will work but not accurate.                                      #
    #############################################################################
    
    # Define optimizer
    optimizer = PoseGraphOptimization();
    
    #Add first node as a fixed vertex. (True = fixed, False = non-fixed)
    optimizer.add_vertex(0, g2o.Isometry3d(nodes[0][0]),True)

    if use_info:
        x_l = [nodes[0][0][0][3]]
        y_l = [nodes[0][0][1][3]]
        yaw_l = [R.from_dcm(nodes[0][0][0:3,0:3]).as_euler('zyx')[0]]
        cov_l = []
        info_l = [np.eye(6)]
    
    for i in range(1,len(nodes)):
        optimizer.add_vertex(i, g2o.Isometry3d(nodes[i][0]), False)
        
        if use_info:
            x_l.append(nodes[i][0][0][3])
            y_l.append(nodes[i][0][1][3])
            yaw_l.append(R.from_dcm(nodes[i][0][0:3,0:3]).as_euler('zyx')[0])

            cov_l = [x_l, y_l, yaw_l]
            cov_l = np.cov(cov_l)
            cov_l = np.insert(cov_l, 2, [0,0,0], axis = 1)
            cov_l = np.insert(cov_l, 2, [0,0,0], axis = 1)
            cov_l = np.insert(cov_l, 2, [0,0,0], axis = 1)
            cov_l = np.insert(cov_l, 2, [0,0,0,0,1,0], axis = 0)
            cov_l = np.insert(cov_l, 2, [0,0,0,1,0,0], axis = 0)
            cov_l = np.insert(cov_l, 2, [0,0,1,0,0,0], axis = 0)
            info_l.append(np.linalg.inv(cov_l))

            optimizer.add_edge([i-1,i],g2o.Isometry3d(nodes[i][2]),
                           information=info_l[-1])
        else:
            optimizer.add_edge([i-1,i],g2o.Isometry3d(nodes[i][2]),
                           information=np.eye(6))

    #############################################################################
    #                                                                           #
    # VISUALIZE LIDAR POINTS INTO GLOBAL. (BEFORE OPTIMIZATION)                 #
    #                                                                           #
    #############################################################################

    if plt_map_no_op:
        for i in range(0,len(nodes)):
            LiDAR = nodes[i][1][0:4];
            LiDAR = np.dot(nodes[i][0], LiDAR)
            plt.scatter(LiDAR[0], LiDAR[1], c='b', marker='o',s=0.2)

        print("Close the plot window to continue...")
        plt.show()

    if plt_odom_no_op:
        for i in range(0,len(nodes)):
            pose = nodes[i][0]
            x = pose[0][3]
            y = pose[1][3]
            plt.scatter(x, y, c='b', marker='o',s=0.2)
        
        print("Close the plot window to continue...")
        plt.show()

    #############################################################################
    # ASSIGNMENTS 3 : FIND LOOP CLOSURE                                         #
    #                                                                           #
    # Simply, you can put all pair in the matching pair, it will be work.       #
    # How can you reduce pairs for less computation? (option)                   #
    #                                                                           #
    #############################################################################     
    #####################################################################################
    # ASSIGNMENTS 4 : MATCHING PAIRS, OPTIMIZE!                                         #
    #                                                                                   #
    # FUNCTION1: T,D,I = icp.icp(dstPoints,srcPoints,tolerance,max_iterations)          #
    #            T : Transformation from src node to dst node (3x3 matrix:2D matching!) #
    #            D : Distances between corresponding points in srcPoints and dstPoints  #
    #            I : Total iterations           					                    #
    #											                                        #
    # Apply initial translation to dst point cloud! if not, icp will inaccurate         #
    #											                                        #
    #####################################################################################         

    threshold = 0.8
    search_cnt = 0
    matching_pairs = []

    for src in range(len(nodes) -1):
        src_pose = optimizer.get_pose(src)
        for dst in range(src + 1, len(nodes)):
            dst_pose = optimizer.get_pose(dst)
            dis_err = np.linalg.norm(src_pose.t - dst_pose.t, 2)

            if dis_err < threshold:
                search_cnt += 1
                srcLiDAR = nodes[src][1][0:4]
                dstLiDAR = nodes[dst][1][0:4]

                srcRT = np.insert(src_pose.R, 3, src_pose.t, axis=1)
                srcRT = np.insert(srcRT, 3, [0, 0, 0, 1], axis=0)
                dstRT = np.insert(dst_pose.R, 3, dst_pose.t, axis=1)
                dstRT = np.insert(dstRT, 3, [0, 0, 0, 1], axis=0)

                srcPoint = srcLiDAR
                dst2srcRT = np.dot(np.linalg.inv(srcRT), dstRT)
                dstPoint = np.dot(dst2srcRT, dstLiDAR)

                #DON'T HAVE TO CHANGE MATCHING FUNCTION
                T, distances, iterations = icp.icp(dstPoint[0:2].T,srcPoint[0:2].T,
                                                    tolerance=0.000001,max_iterations=100)
                #### MAKE 3x3 matrix into 4x4 matrix ####
                T = np.insert(T, 2, [0, 0, 0], axis=1)
                T = np.insert(T, 2, [0, 0, 1, 0], axis=0)

                #DRAWING FUNCTION FOR CHECKING ICP DONE WELL : Source blue, Dest green, Dest after ICP red
                
                #dstTrans = np.dot(T, dstPoint)
                #plt.scatter(dstPoint[0], dstPoint[1], c='g', marker='o',s=0.2)
                #plt.scatter(srcPoint[0], srcPoint[1], c='b', marker='o',s=0.2)
                #plt.scatter(dstTrans[0], dstTrans[1], c='r', marker='o',s=0.2)
                #plt.show()
                
                if(np.average(distances) < 0.05 and np.abs(src - dst) > 20):	# ADD CONDITION OF MATCHING SUCCESS (ex: mean of distances less then 0.05 [m])
                    matching_pairs.append([src, dst])
                    print("{:>3} th Matching {:>3} => {:>3} , Iter: {:>3} , ME: {:>4}"
                            .format(len(matching_pairs), src, dst, iterations, np.round(np.average(distances), 4)))
                    if use_info:
                        optimizer.add_edge([src,dst], g2o.Isometry3d(np.dot(T, dst2srcRT)),
                                    information=info_l[dst])
                    else:
                        optimizer.add_edge([src,dst], g2o.Isometry3d(np.dot(T, dst2srcRT)),
                                    information=np.eye(6))
                    optimizer.optimize()
    print("=========================================================")
    print("Number of searched pairs: {}, Number of matched pairs: {}"
            .format(search_cnt, len(matching_pairs)))
    print("=========================================================")

    #############################################################################
    #                                                                           #
    # VISUALIZE LIDAR POINTS INTO GLOBAL (AFTER OPTIMIZATION)                   #
    #                                                                           #
    #############################################################################

    if plt_map_op:
        for i in range(0,len(nodes)):
            dstLiDAR = nodes[i][1][0:4];
            rt = optimizer.get_pose(i)
            T = np.insert(rt.R, 3, rt.t, axis=1)
            T = np.insert(T, 3, [0, 0, 0, 1], axis=0)
            dstLiDAR = np.dot(T, dstLiDAR)
            plt.scatter(dstLiDAR[0], dstLiDAR[1], c='b', marker='o',s=0.2)
            
        print("Close the plot window to continue...")
        plt.show()

    if plt_odom_op:
        for i in range(0,len(nodes)):
            rt = optimizer.get_pose(i)
            x = rt.t[0]
            y = rt.t[1]
            plt.scatter(x, y, c='b', marker='o',s=0.2)

        for src, dst in matching_pairs:
            rt_src = optimizer.get_pose(src)
            rt_dst = optimizer.get_pose(dst)
            x_src = rt_src.t[0]
            y_src = rt_src.t[1]
            x_dst = rt_dst.t[0]
            y_dst = rt_dst.t[1]
            xx = [x_src, x_dst]
            yy = [y_src, y_dst]
            plt.plot(xx, yy, c='r', linewidth=1.0)
    
        print("Close the plot window to continue...")
        plt.show()

    optimizer.save_g2o('afterSLAM.g2o')



if __name__ == '__main__':
    main()

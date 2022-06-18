#############
###IMPORTS###
#############

import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from glob import glob
import time
import pandas as pd
import mediapipe as mp
import keras
import tensorflow as tf
import pickle
import math
from scipy.spatial import distance 
import matplotlib.lines as mlines
import matplotlib.cm as cm
from math import atan2

##########################
###GPU INFERENCE CONFIG###
##########################

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

############################
###USER DEFINED VARIABLES###
############################
print('Type left_wrist or right_wrist to analyze one of them \n')
wrist_to_analyze = input()
print('Type the number of iterations you want to perform for the wrist')
n_test = int(input())

###############
###VARIABLES###
###############

#Initialize the Mediapipe module with its corresponding parameters
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.5, static_image_mode=True)

square_size = 0.043 #meters
width_chessboard = 5
height_chessboard = 3

#Setup RGSClinic 15º tilt

markers_gt_coordinates = np.array([
    [-0.035999950021505356,  0.841], 
    [-0.21099995076656342,  0.871],
    [-0.2749999463558197,  0.697],
    [-0.35999995470046997, 0.897], 
    [-0.47099995613098145,  0.772], 
    [0.20200004935264587, 0.870],
    [0.26200006663799286, 0.681],
    [0.3290000343322754, 0.870],
    [0.45900004863739014, 0.780],   
])

#Select the number of keypoints to detect 
number_keypoints_to_detect = 6
#Hips are needed in calibration
indices_landmark_interest_calibration = [11, 12, 13, 14, 15 ,16, 23, 24]

#Select the indices of the landmarks of interest to detect from the output of the previous cell
indices_landmark_interest = [11, 12, 13, 14, 15 ,16]

###############
###FUNCTIONS###
###############

#Function that allows to select the wrist coordinates regarding left or right arm. 
#The index correspond to the predicted list where the order is: [right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist]
def index_to_select(wrist):
    if wrist == 'right_wrist':
        mediapipe_index = 2
    else:
        mediapipe_index = 5      
    return(mediapipe_index)

def markers_to_analyze(wrist):
    if wrist =='right_wrist':
        markers_to_analyze = [0,1,2,3,4,5,6]
    else:
        markers_to_analyze = [0,1,2,5,6,7,8]
    return(markers_to_analyze)
        

def markers_drawing(wrist):
    if wrist =='right_wrist':
        circle_list= [
            (977,719),
            (1274,743),
            (1333,455),
            (1517,828),
            (1670, 625),
            (683,777),
            (598,502)]
    else:
        circle_list = [
            (977,719),
            (1274,743),
            (1333,455),
            (683,777),
            (598,502),
            (426,892),
             (259,693)]
        
    return(circle_list)

def split_3_coordinates_array(array):
    m = 0
    r = 3
    splitted_coords = []
    for i in range(0,int(len(array)/3)):
        splitted_coords.append(array[m:r])
        m = r
        r = r+3
    return(np.array(splitted_coords))

def mp_inference_for_calibration (frame):
    results = pose.process(frame)
    image_height, image_width, _ = frame.shape

    #3D coordinates
    x_3d = []
    y_3d = []
    z_3d = []

    x_2d = []
    y_2d = []
    start_index = 0
    
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            if start_index in indices_landmark_interest_calibration:
                if landmark.visibility>0.4:
                    
                    x_3d.append(-landmark.x)
                    y_3d.append(-landmark.y)
                    z_3d.append(-landmark.z)

                    x_2d.append(landmark.x*image_width)
                    y_2d.append(landmark.y*image_height)
                    start_index = start_index+1
                    
                else: 
                    x_3d.append(np.NaN)
                    y_3d.append(np.NaN)
                    z_3d.append(np.NaN)

                    x_2d.append(np.NaN)
                    y_2d.append(np.NaN)
                    start_index = start_index+1
            else:
                start_index = start_index+1
    
        #Changing order to meet: right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist, left_hip, right_hip
        x_3d = [x_3d[1],x_3d[3],x_3d[5],x_3d[0],x_3d[2],x_3d[4],x_3d[6], x_3d[7]]
        y_3d = [y_3d[1],y_3d[3],y_3d[5],y_3d[0],y_3d[2],y_3d[4],y_3d[6],y_3d[7]]
        z_3d = [z_3d[1],z_3d[3],z_3d[5],z_3d[0],z_3d[2],z_3d[4],z_3d[6], z_3d[7]]
        
    
        x_middle_point_hip_3d = (x_3d[-1]+x_3d[-2])/2
        y_middle_point_hip_3d = (y_3d[-1]+y_3d[-2])/2

        x_3d_corrected = [x-x_middle_point_hip_3d for x in x_3d]
        y_3d_corrected = [y-y_middle_point_hip_3d for y in y_3d]
        z_3d_corrected = z_3d
        
        #Getting the keypoint coordinates represented in WorldOrigin placed in middle hip keypoint
        x_3d_world_origin = x_3d_corrected[:-2]
        z_3d_world_origin = z_3d_corrected[:-2]
        y_3d_world_origin = y_3d_corrected[:-2]

    else: 
        x_3d_world_origin = np.zeros(number_keypoints_to_detect)
        x_3d_world_origin[:] = np.nan
        
        y_3d_world_origin = np.zeros(number_keypoints_to_detect)
        y_3d_world_origin[:] = np.nan
        
        z_3d_world_origin = np.zeros(number_keypoints_to_detect)
        z_3d_world_origin[:] = np.nan
        
        x_2d = np.zeros(number_keypoints_to_detect)
        x_2d[:] = np.nan
        y_2d = np.zeros(number_keypoints_to_detect)
        y_2d[:] = np.nan

        x_middle_point_hip_3d = np.nan
        y_middle_point_hip_3d = np.nan

    return(x_3d_world_origin, y_3d_world_origin, z_3d_world_origin, x_middle_point_hip_3d, y_middle_point_hip_3d, x_2d, y_2d)


def mediapipe_inference(frame, x_middle_point_hip_3d_calibration, y_middle_point_hip_3d_calibration): 
    results = pose.process(frame)
    image_height, image_width, _ = frame.shape

    #3D coordinates
    x_3d = []
    y_3d = []
    z_3d = []

    x_2d = []
    y_2d = []
    start_index = 0
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            if start_index in indices_landmark_interest:
                if landmark.visibility>0.4:
                    x_3d.append(-landmark.x)
                    y_3d.append(-landmark.y)
                    z_3d.append(-landmark.z)

                    x_2d.append(landmark.x*image_width)
                    y_2d.append(landmark.y*image_height)
                    start_index = start_index+1
                else: 
                    x_3d.append(np.NaN)
                    y_3d.append(np.NaN)
                    z_3d.append(np.NaN)

                    x_2d.append(np.NaN)
                    y_2d.append(np.NaN)
                    start_index = start_index+1
            else:
                start_index = start_index+1
                    
        #Changing order to meet: right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist, left_hip, right_hip
        x_3d = [x_3d[1],x_3d[3],x_3d[5],x_3d[0],x_3d[2],x_3d[4]]
        y_3d = [y_3d[1],y_3d[3],y_3d[5],y_3d[0],y_3d[2],y_3d[4]]
        z_3d = [z_3d[1],z_3d[3],z_3d[5],z_3d[0],z_3d[2],z_3d[4]]

        x_3d_corrected = [x-x_middle_point_hip_3d_calibration for x in x_3d]
        y_3d_corrected = [y-y_middle_point_hip_3d_calibration for y in y_3d]
        z_3d_corrected = z_3d  

    else:

        x_3d_corrected = np.zeros(number_keypoints_to_detect)
        x_3d_corrected[:] = np.nan
        
        y_3d_corrected = np.zeros(number_keypoints_to_detect)
        y_3d_corrected[:] = np.nan
        
        z_3d_corrected = np.zeros(number_keypoints_to_detect)
        z_3d_corrected[:] = np.nan
        
        x_2d = np.zeros(number_keypoints_to_detect)
        x_2d[:] = np.nan
        y_2d = np.zeros(number_keypoints_to_detect)
        y_2d[:] = np.nan
        
    return(x_3d_corrected, y_3d_corrected, z_3d_corrected, x_2d, y_2d)

def calibration_picture():
    print('A picture will be taken in 2 secs. Please show all your upper body')
    for i in range(1,2):
            cap1 = cv2.VideoCapture(0)
            success, frame1 = cap1.read()
            frame1 = cv2.resize(frame1,(1920,1080))
            frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2RGB)
            frame1.flags.writeable = False
            cap1.release()
    return(frame1)


################################################
###Camera extrinsic and intrinsic parameters ###
################################################

#termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((width_chessboard*height_chessboard,3), np.float32) #9x6 np.zeros array
objp[:,:2] = np.mgrid[0:width_chessboard,0:height_chessboard].T.reshape(-1,2)
objp = objp * square_size


images = glob('../WorldOriginForMediaPipe3D/*.jpg')

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane
processed_images = []

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    
    ret, corners = cv2.findChessboardCorners(gray, (width_chessboard,height_chessboard), None)
    
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (width_chessboard,height_chessboard), corners2, ret)
        cv2.imshow('img', img)
        processed_images.append(img)
        time.sleep(2)
        if cv2.waitKey(500) and 0xFF==ord('q'):
            break
            
cv2.destroyAllWindows()

#compute intrinsic parameter matrix and rvecs and tvecs for each picture
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#Not considering distortion coefficients
dist_coeffs = np.zeros((5,1))
#Obtaining the rvecs and tvecs for the specific image with the origin in the World Origin
found, rvecs_new, tvecs_new = cv2.solvePnP(objpoints[0],imgpoints[0], mtx, dist_coeffs)
#Rotation matrix
R = cv2.Rodrigues(rvecs_new)[0]
roll = atan2(R[1,2], R[2,2])
roll = 90-((roll*360)/(2*np.pi))
h1 = float(tvecs_new[1]/np.cos(np.deg2rad(roll)))
h2 = float(tvecs_new[2]*np.cos(np.deg2rad(90-roll)))
H = h1+h2
i = float(tvecs_new[2]*np.cos(np.deg2rad(roll)))
d = float(tvecs_new[0])
final_tvecs = np.array([d, i, H])

#Projection matrix
P = np.append(R, final_tvecs.reshape(3,1), axis=1) 
new_row = np.array([0.0, 0.0, 0.0, 1.0]).reshape(1,4)

#Converting P in a 4x4 matrix
P = np.append(P, new_row, axis=0)
#Computing camera matrix inverted
cam_matrix_inverted = np.linalg.inv(P)

###########################
### Calibration picture ###
###########################

frame = calibration_picture()
x_3d_calibration, y_3d_calibration, z_3d_calibration, x_middle_point_hip_3d_calibration, y_middle_point_hip_3d_calibration, x_2d_calibration, y_2d_calibration = mp_inference_for_calibration(frame)

array_3d_coords = np.zeros([6,3])
array_3d_coords[:,0] = x_3d_calibration
array_3d_coords[:,1] = y_3d_calibration
array_3d_coords[:,2] = z_3d_calibration

df_3D = pd.DataFrame(array_3d_coords.reshape(1,18))
df_3D = df_3D.append(pd.DataFrame(np.zeros([1,18]))).reset_index().drop(columns=['index'])

while df_3D.isna().sum().sum()>1: 
    print('Joints have not been detected properly - Another picture will be taken \n')
    print(array_3d_coords)
    frame = calibration_picture()

    x_3d_calibration, y_3d_calibration, z_3d_calibration, x_middle_point_hip_3d_calibration, y_middle_point_hip_3d_calibration, x_2d_calibration, y_2d_calibration = mp_inference_for_calibration(frame)
    array_3d_coords = np.zeros([6,3])
    array_3d_coords[:,0] = x_3d_calibration
    array_3d_coords[:,1] = y_3d_calibration
    array_3d_coords[:,2] = z_3d_calibration

    df_3D = pd.DataFrame(array_3d_coords.reshape(1,18))
    df_3D = df_3D.append(pd.DataFrame(np.zeros([1,18]))).reset_index().drop(columns=['index'])
print('\n Picture taken')


######################
###METRIC PROCEDURE###
######################

#Only 7 markers are analyzed for each wrist. Oposite-extrem ones to the wrist are not considered. 
gamming_image = cv2.imread('./gamming_zone.jpg')

#Llista on hi haurà n_test arrays differents de 12,2 shape. 
n_test_performed_list_results = []

for j in range(0,n_test):
    markers_pred_coordinates = np.zeros([7,2])
    df = pd.DataFrame(columns=np.arange(0,12,1))
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if cap.isOpened():
        for i in range(0,len(markers_pred_coordinates)):
            try:  
                
                marker_numbers = markers_to_analyze(wrist_to_analyze)
                circle_list = markers_drawing(wrist_to_analyze)
                gamming_image = cv2.circle(gamming_image, circle_list[i], radius=75, color =(0,255,0),thickness=-1)
                gamming_image = cv2.putText(gamming_image, str(i+1), (circle_list[i][0]-20,circle_list[i][1]+20), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255,0,0), thickness = 12)

                cv2.imshow('Guide', gamming_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                print('Put your wrist on marker number '+str(marker_numbers[i]+1))
                time.sleep(1.5)
                success, frame = cap.read ()
                frame = cv2.resize(frame,(1920,1080))
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
        
                #Getting the coordinates in 3D world origin
                x_3d, y_3d, z_3d, x_2d, y_2d= mediapipe_inference(frame,x_middle_point_hip_3d_calibration, y_middle_point_hip_3d_calibration)
                array_3d_coords = np.zeros([6,3])
                array_3d_coords[:,0] = x_3d
                array_3d_coords[:,1] = y_3d
                array_3d_coords[:,2] = z_3d
                 #Replacing the last array by the detected coordinates
                df_3D.iloc[1] = array_3d_coords.reshape(1,18)
                df_3D = df_3D.fillna(method='ffill')
                corrected_array = np.array(df_3D.iloc[1]).reshape(1,18)
                df_3D.iloc[0] = corrected_array 
                
                world_origin_coords = np.zeros([6,3])
                world_origin_coords[:,0] = [df_3D.iloc[1][0], df_3D.iloc[1][3],df_3D.iloc[1][6], df_3D.iloc[1][9],df_3D.iloc[1][12], df_3D.iloc[1][15]]
                world_origin_coords[:,1] = [df_3D.iloc[1][2], df_3D.iloc[1][5],df_3D.iloc[1][8], df_3D.iloc[1][11],df_3D.iloc[1][14], df_3D.iloc[1][17]]
                world_origin_coords[:,2] = [df_3D.iloc[1][1], df_3D.iloc[1][4],df_3D.iloc[1][7], df_3D.iloc[1][10],df_3D.iloc[1][13], df_3D.iloc[1][16]]
                
                #Translating and rotating
                output_list_coords_camera_frame = []
                for l in range(0,len(world_origin_coords)):
                    coords_3d_camera_frame = cam_matrix_inverted@np.append(world_origin_coords[l,:],1.0).reshape(4,1)
                    coords_3d_camera_frame = - coords_3d_camera_frame[0:-1]
                    output_list_coords_camera_frame.append(coords_3d_camera_frame.reshape(1,3))
                print('output_list')
                print(output_list_coords_camera_frame)
                z_predicted = np.array(output_list_coords_camera_frame).reshape(1,18)
                z_predicted_splitted = split_3_coordinates_array(z_predicted[0])
                print(z_predicted_splitted)
                markers_pred_coordinates[i] = np.array([z_predicted_splitted[index_to_select(wrist_to_analyze)][0], z_predicted_splitted[index_to_select(wrist_to_analyze)][2]])

            except KeyboardInterrupt:
                print('Camera is closed')
                cap.release() 
                
                
        n_test_performed_list_results.append(markers_pred_coordinates)

    print('Process Finished')
    cap.release()
    cv2.destroyAllWindows()
    gamming_image = cv2.imread('./gamming_zone.jpg')
    

###################################
### COMPUTING EUCLIDEAN DISTANCE###
###################################

#Select just the ground-truth markers based on which wrist is being analyzed
final_markers_gt_coordinates = []
marker_numbers = markers_to_analyze(wrist_to_analyze)
for marker in marker_numbers:
    final_markers_gt_coordinates.append(markers_gt_coordinates[marker])
final_markers_gt_coordinates = np.array(final_markers_gt_coordinates)

#Euclidean distance list containing n_test times the euclidean distance computed between ground-truth and predicted values
n_euc_results = []
for array in n_test_performed_list_results:
    euc_results = []
    for i in range(0,len(final_markers_gt_coordinates)):
        euc_results.append(distance.euclidean(array[i],final_markers_gt_coordinates[i]))
    n_euc_results.append(euc_results)

#Smoothing euclidean distance results
n_final_results = []
for euc_array in n_euc_results:
    final_results = []
    for result in euc_array:
        corrected_result = result-0.03
        if corrected_result < 0:
            corrected_result = 0.0
        else:
            pass
        final_results.append(corrected_result)  
    n_final_results.append(final_results)


print('The mean error in [cm] is: '+str(sum(n_final_results[0])/len(n_final_results[0])*100))
print('The std in [cm] is: ' +str(pd.DataFrame(n_final_results[0]).std()[0]*100))

#######################################
###PROJECTING ERROR VS GT UPON TABLE###
#######################################

markers_type = ['s','p','P','x','D','v','^']
final_array = np.array(n_test_performed_list_results)
fig = plt.figure(figsize=(10,8))

#Plotting ground-truth values
plt.scatter(markers_gt_coordinates[:,0],markers_gt_coordinates[:,1], color='green', s=70,label='Ground-Truth')

#Plotting predictions
for test_performed in n_test_performed_list_results:
    for i in range(0,len(test_performed)):
        plt.scatter(test_performed[i,0], test_performed[i,1], color='red',s=70, marker=markers_type[i])

#Computing centroid and plotting it
predicted_markers_centroid_coordinates = []
for i in range(0,7):
    centroid_coordinates = (sum(final_array[:,i,0])/len(final_array[:,i,0]), sum(final_array[:,i,1])/len(final_array[:,i,1]))
    predicted_markers_centroid_coordinates.append(centroid_coordinates)
    plt.scatter(centroid_coordinates[0], centroid_coordinates[1], color='blue', s=70, marker='*')

centroid_euc_results = []
for i in range(0,len(final_markers_gt_coordinates)):
    centroid_euc_results.append(distance.euclidean(predicted_markers_centroid_coordinates[i],final_markers_gt_coordinates[i]))    

#Lines between GT markers and Centroids
markers_gt_coordinates_splitted = markers_gt_coordinates[marker_numbers]
for i in range(0,len(centroid_euc_results)):
        plt.plot([markers_gt_coordinates_splitted[i][0],predicted_markers_centroid_coordinates[i][0]], [markers_gt_coordinates_splitted[i][1],predicted_markers_centroid_coordinates[i][1]], color='black')

plt.title('Marker Testing - Predictions VS Ground-truth')
plt.ylabel('Meters - Depth coordinate')
plt.xlabel('Meters - Width coordinate')

handles, labels = plt.gca().get_legend_handles_labels()
blue_star = mlines.Line2D([], [], color='blue', marker='*',markersize=15, label='Prediction centroid')
red_dot = mlines.Line2D([], [], color='red', marker='.',markersize=20, label='Prediction keypoint')
handles.extend([blue_star, red_dot])
fig.legend(handles=handles)
plt.savefig('./output/error_projection_'+str(wrist_to_analyze)+'.jpg')




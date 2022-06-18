import mediapipe as mp
import numpy as np
import cv2
import time
import variables

#Initialize the Mediapipe module with its corresponding parameters
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.7)

def split_3_coordinates(values_array): 
    output = []
    for i in range(0, len(values_array)):
        m = 0
        r = 3
        frame_coordinates = []
        for j in range(0,int(len(values_array[i])/3)):
            frame_coordinates.append(values_array[i][m:r])
            m = r
            r = r+3
        output.append(frame_coordinates)
    return(output)

def mediapipe_inference(frame):
    
    results = pose.process(frame)
    frame_list1 = []
    start_index = 0
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            #If landmark visibility is not 0.4 save a NaN and this value will be replaced by the previous recorded value !!!!
            #Reference & first value is the callibraition picture 
            if start_index in variables.indices_landmark_interest:
                if landmark.visibility>0.4:
                    image_hight, image_width, _ = frame.shape
                    frame_list1.append(landmark.x*image_width)
                    frame_list1.append(landmark.y*image_hight)
                    start_index = start_index+1
                else:
                    frame_list1.append(np.NaN)
                    frame_list1.append(np.NaN)
                    start_index = start_index+1
            else: 
                start_index = start_index+1
    else:
        frame_list1 = np.zeros(variables.number_keypoints_to_detect*2)
        frame_list1[:] = np.nan    
        
    return(frame_list1)
        
def callibration_picture():
    print('A picture will be taken in 2 secs. Please show all your upper body')
    #time.sleep(2)
    for i in range(1,2):
        cap1 = cv2.VideoCapture(0)
        success, frame1 = cap1.read()
        frame1 = cv2.resize(frame1,(1920,1080))
        frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2RGB)
        frame1.flags.writeable = False
        cap1.release()
    return(frame1)

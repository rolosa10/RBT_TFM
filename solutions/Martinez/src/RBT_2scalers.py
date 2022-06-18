#############
###Imports###
#############
import numpy as np
from glob import glob
import pandas as pd
import keras
import pickle
import tensorflow as tf
import socket
import time
import cv2
from functions import split_3_coordinates
from functions import mediapipe_inference
from functions import callibration_picture

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
#Socket definition
ip_receiver = '127.0.0.1'
port_receiver = 5005
experiment = '006_model/lvl3_exp_decay'

#Path of scaler and nn weights:
model = keras.models.load_model('../RBT_Gravity_Roadmap/models/'+experiment+'/model.h5')
scaler_2D = pickle.load(open('../RBT_Gravity_Roadmap/models/'+experiment+'/scaler_2D.pkl','rb'))
scaler_3D = pickle.load(open('../RBT_Gravity_Roadmap/models/'+experiment+'/scaler_3D.pkl','rb'))                    

##########################
###Callibration picture###
##########################

frame = callibration_picture()
detected_keypoints = mediapipe_inference(frame)
detected_keypoints_correct_order = [detected_keypoints[2], detected_keypoints[3], detected_keypoints[6], detected_keypoints[7], detected_keypoints[10], detected_keypoints[11], detected_keypoints[0],detected_keypoints[1], detected_keypoints[4], detected_keypoints[5], detected_keypoints[8], detected_keypoints[9]]

#If there is one NaN in callibration picture (joint not detected properly) or no keypoints have been detected --> Another picture is taken
while pd.DataFrame(detected_keypoints).T.isna().sum().sum()>= 1 or len(detected_keypoints) == 0: 
    print('Joints have not been detected properly - Another picture will be taken \n')
    print(detected_keypoints)
    frame = callibration_picture()
    
    detected_keypoints= mediapipe_inference(frame)
    detected_keypoints_correct_order = [detected_keypoints[2], detected_keypoints[3], detected_keypoints[6], detected_keypoints[7], detected_keypoints[10], detected_keypoints[11], detected_keypoints[0],detected_keypoints[1], detected_keypoints[4], detected_keypoints[5], detected_keypoints[8], detected_keypoints[9]]



df = pd.DataFrame(detected_keypoints_correct_order).T
df = df.append(pd.DataFrame(np.zeros([1,12]))).reset_index().drop(columns=['index'])

#############
###RBT#######
#############

#Frame number - 0 is the callibration picture, the following the detected webcam frames
n = 0

#Initialize stream capture
cap = cv2.VideoCapture(0)

prueba_time_mp = []

try: 
    while cap.isOpened():
        n = n+1
        sucess,frame = cap.read()
        frame = cv2.resize(frame,(1920,1080))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False

        #MediaPipe inference
        detected_keypoints = mediapipe_inference(frame)
        detected_keypoints_correct_order = [detected_keypoints[2], detected_keypoints[3], detected_keypoints[6], detected_keypoints[7], detected_keypoints[10], detected_keypoints[11], detected_keypoints[0], detected_keypoints[1], detected_keypoints[4], detected_keypoints[5], detected_keypoints[8], detected_keypoints[9]]
        #Replace the iloc[1] values for the detected ones
        df.iloc[1] = np.array(detected_keypoints_correct_order).reshape(1,12)
        #Replace the NaN values of the detected keypoints by previous values row in the same column
        df = df.fillna(method='ffill')
        
        #Replace iloc[0] values for the ffilled row
        array2replace = np.array(df.iloc[1]).reshape(1,12)
        df.iloc[0] = array2replace
        
        #Change column order to meet right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist
        #df = df[[2,3,6,7,10,11,0,1,4,5,8,9]]
        
        if all(isinstance(n,float) for n in list(df.iloc[1])) == True:

        #NN inference
            X = np.array([list(df.iloc[1])]).astype(float)
            X_scaled = scaler_2D.transform(X)
            z_predicted = np.float32(model.predict(X_scaled) *scaler_3D.scale_ + scaler_3D.mean_)
            df_pred_3d = pd.DataFrame(split_3_coordinates(z_predicted))
            
            #Sending udp output
            msg = bytes(str(df_pred_3d.iloc[0].tolist()),'utf-8')
            print(f'Sending {msg} to {ip_receiver}:{port_receiver}')
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(msg,(ip_receiver, port_receiver))
            print(df_pred_3d)
        
except KeyboardInterrupt:
    print('Camera is closed')
    cap.release()
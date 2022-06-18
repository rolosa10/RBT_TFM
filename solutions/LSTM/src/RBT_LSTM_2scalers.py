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
import cv2
from functions import mediapipe_inference
from functions import callibration_picture

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
#Socket definition
ip_receiver = '127.0.0.1'
port_receiver = 5005
experiment = 'Gravity/2nd_model_mixamo'

#Path of scaler and nn weights:
model = keras.models.load_model('../models/'+experiment+'/model.h5')
scaler_2D = pickle.load(open('../models/'+experiment+'/scaler_2D.pkl','rb'))
scaler_3D = pickle.load(open('../models/'+experiment+'/scaler_3D.pkl','rb'))                    

##########################
###Callibration picture###
##########################

window_2D_detections = np.zeros([5,12])

for i in range(0,5):
    print('Calibration picture number '+str(i+1)+' from 5 is being taken')
    frame = callibration_picture()
    detected_keypoints = mediapipe_inference(frame)
    detected_keypoints_correct_order = np.array([detected_keypoints[2], detected_keypoints[3], detected_keypoints[6], detected_keypoints[7], detected_keypoints[10], detected_keypoints[11], detected_keypoints[0],detected_keypoints[1], detected_keypoints[4], detected_keypoints[5], detected_keypoints[8], detected_keypoints[9]]).reshape(1,12)

    #If there is one NaN in callibration picture (joint not detected properly) or no keypoints have been detected --> Another picture is taken
    while pd.DataFrame(detected_keypoints).T.isna().sum().sum()>= 1 or len(detected_keypoints) == 0: 
        print('Joints have not been detected properly - Another picture will be taken \n')
        print(detected_keypoints)
        frame = callibration_picture()
        
        detected_keypoints= mediapipe_inference(frame)
        detected_keypoints_correct_order = np.array([detected_keypoints[2], detected_keypoints[3], detected_keypoints[6], detected_keypoints[7], detected_keypoints[10], detected_keypoints[11], detected_keypoints[0],detected_keypoints[1], detected_keypoints[4], detected_keypoints[5], detected_keypoints[8], detected_keypoints[9]]).reshape(1,12)
    window_2D_detections[i] = detected_keypoints_correct_order


df = pd.DataFrame(window_2D_detections) 

#############
###RBT#######
#############

#Initialize stream capture
cap = cv2.VideoCapture(0)


try: 
    while cap.isOpened():
        sucess,frame = cap.read()
        frame = cv2.resize(frame,(1920,1080))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False

        #MediaPipe inference
        detected_keypoints = mediapipe_inference(frame)
        detected_keypoints_correct_order = np.array([detected_keypoints[2], detected_keypoints[3], detected_keypoints[6], detected_keypoints[7], detected_keypoints[10], detected_keypoints[11], detected_keypoints[0], detected_keypoints[1], detected_keypoints[4], detected_keypoints[5], detected_keypoints[8], detected_keypoints[9]]).reshape(1,12)
        #Dropping t-4 frame
        df = df.drop(df.index[0]).reset_index().drop(columns=['index'])
        #Appending data to the last row (5th row)
        df = df.append(pd.DataFrame(detected_keypoints_correct_order)).reset_index().drop(columns=['index'])
        #Replace the NaN values of the detected keypoints by previous values row in the same column
        df = df.fillna(method='ffill')
    
        
        

        #NN inference

        X = np.flip(np.array(df), axis=0).astype(float) #Input sequence reversed as suggested by Sutsekever et al
        #X = np.array(df).astype(float) ## Input sequence without being reversed
        X_scaled = scaler_2D.transform(X)
        z_predicted = np.float32(model.predict(X_scaled.reshape(1,5,12))*scaler_3D.scale_ + scaler_3D.mean_)
        #df_pred_3d = pd.DataFrame(split_nd_coordinates_array(z_predicted[0][4],3))
        
        #Sending udp output
        msg = bytes(str(z_predicted[0][4].tolist()),'utf-8')
        print(f'Sending {msg} to {ip_receiver}:{port_receiver}')
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(msg,(ip_receiver, port_receiver))
        
except KeyboardInterrupt:
    print('Camera is closed')
    cap.release()
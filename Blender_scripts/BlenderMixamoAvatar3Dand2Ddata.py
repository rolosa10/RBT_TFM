import mathutils
import bpy
from bpy_extras.object_utils import world_to_camera_view
import math

# Timecode Cuepoint corresponding to current mocap.
# Find it into "Cues.txt" file

cue = 0.0


#Export CSV
csv_file="C:\\Users\\Eodyne-TestStation\Desktop\\mixamo_fbx\\output_csv\\JoelAnimation.csv"
file=open(csv_file, 'w+')


# Frames per second
fps = 90;

#import bge --> Blender herarchy
sce = bpy.context.scene
ob = bpy.context.object
cam = bpy.data.objects['KinectCam']

#Standing2Sit FBX
ob.rotation_euler = (math.radians(90),0,math.radians(180))
#ob.location = (0, -0.2, -1.1)
ob.location = (0, -1.5, -2.8)

#Skeleton BaseName
skeletonName = "Armature"

#CSV format 
header = "timecode"



JointsOfInterest = {
"mixamorig1:RightShoulder": "RightShoulder",
"mixamorig1:RightForeArm": "RightElbow",
"mixamorig1:RightHand":"RightHand",
"mixamorig1:LeftShoulder": "LeftShoulder",
"mixamorig1:LeftForeArm": "LeftElbow",
"mixamorig1:LeftHand": "LeftHand"}


for i in range(0,2):
    for pbone in ob.pose.bones:
        if pbone.name in list(JointsOfInterest.keys()):
            if i == 0: 
                header = header+","+JointsOfInterest[pbone.name]
                header = header+","+JointsOfInterest[pbone.name]
                header = header+","+JointsOfInterest[pbone.name]
            if i == 1: 
                header = header+","+JointsOfInterest[pbone.name]+"_2D"
                header = header+","+JointsOfInterest[pbone.name]+"_2D"

#Write header
file.writelines(header+"\n")        

#Create log entries
#Just to check the 10 first frames. If you want to analyze the full video: sce.frame_end+1
for frame in range(sce.frame_start, sce.frame_end+1):
    sce.frame_set(frame)
    #Time code --> 3 decimal digit accuracy 
    entry = ("%.3f" % (cue+(frame/fps)))
    #3D entries
    for pbone in ob.pose.bones:
        if pbone.name in list(JointsOfInterest.keys()):
            #Just projecting to the object world
            #It returns x, y, z coordinates but z is the height and y is the depth  
            pos = ob.matrix_world@pbone.head
            #Displacing prosition to Non-Gravity Protocol
            pos[1] = pos[1] #+ 1.350
            pos[2] = pos[2] #- 0.40
            #Projecting to camera frame
            output = cam.matrix_world.inverted() @ pos
            output[2] = -output[2]
            
            entry = entry+","+("%.3f"% output[0])
            entry = entry+","+("%.3f"% output[1])
            entry = entry+","+("%.3f"% output[2])
          
    for pbone in ob.pose.bones:
        if pbone.name in list(JointsOfInterest.keys()):
            pos = ob.matrix_world@pbone.head
            
            #Displacing prosition to Non-Gravity Protocol
            pos[1] = pos[1] #+ 1.350
            pos[2] = pos[2] #- 0.40

            pos2d = world_to_camera_view(bpy.context.scene, cam, pos)
            entry = entry + "," + ("%.3f" % (pos2d[0] * 1920))
            entry = entry + "," + ("%.3f" % ((1-pos2d[1]) * 1080))
            
    file.writelines(entry + "\n")
file.close()
print('CSV generated')
        

        
        
        
   
        
        
        

        
        
        
        
        

        
        
        
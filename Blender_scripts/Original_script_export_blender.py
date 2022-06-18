# Timecode Cuepoint corresponding to current mocap.
# Find it into "Cues.txt" file
# 
cue = 16.184


#Export CSV
csv_file=".\\Data\\Adri_Azure_04.csv"
file=open(csv_file, 'w')


# Frames per second
fps = 60;

import mathutils
import bpy
from bpy_extras.object_utils import world_to_camera_view

#import bge --> Blender herarchy

sce = bpy.context.scene
ob = bpy.context.object
cam = bpy.data.objects['KinectCam']

# Position and orientation of the Kinect reference frame
KinectMatrix = [ 1.0000,  0.0000,  0.0000, 0.0000], [ 0.0000, -0.2588, -0.9659, 0.82],[-0.0000,  0.9659, -0.2588, 0.570],[ 0.0000,  0.0000,  0.0000, 1.0000]
KinectMatrix = (mathutils.Matrix(KinectMatrix))
KinectMatrix = KinectMatrix.inverted()





#Skeleton BaseName
#skeletonName = "XaviSkeleton002:"
skeletonName = "AdrianSkeleton003:"

# Select only the joints of interest
def filterJoint(boneName):
    result = ""
    if (boneName == skeletonName+"RightArm"): return "RightShoulder"
    if (boneName == skeletonName+"RightForeArm"): return "RightElbow"
    if (boneName == skeletonName+"RightHand"): return "RightHand"
    if (boneName == skeletonName+"LeftArm"): return "LeftShoulder"
    if (boneName == skeletonName+"LeftForeArm"): return "LeftElbow"
    if (boneName == skeletonName+"LeftHand"): return "LeftHand"
    return result

# Create logfile header
header = "timecode"
# 3D section
for pbone in ob.pose.bones:
    found_bone =  filterJoint(pbone.name)
    if (found_bone != ""):
        header = header + "," + found_bone 
        header = header + "," + found_bone
        header = header + "," + found_bone 
        
                
# 2D Section
for pbone in ob.pose.bones:
    found_bone =  filterJoint(pbone.name)
    if (found_bone != ""):
        header = header + "," + found_bone + "_2D"
        header = header + "," + found_bone+ "_2D"
file.writelines(header + "\n")

# Create log entries
for frame in range(sce.frame_start, sce.frame_start+10):#sce.frame_end+1):
    sce.frame_set(frame)
    #Time code
    entry =  ("%.3f" % (cue + (frame / fps)));
    #3D Section
    for pbone in ob.pose.bones:

        found_bone =  filterJoint(pbone.name)
        if (found_bone != ""):
            pos = cam.matrix_world.inverted() @ ob.matrix_world @ pbone.head
            #pos = KinectMatrix @ ob.matrix_world @ pbone.head
            entry = entry + "," + ("%.3f" % (-pos[0]))
            entry = entry + "," + ("%.3f" % pos[1])
            entry = entry + "," + ("%.3f" % pos[2])
    #2D Section        
    for pbone in ob.pose.bones:
        #timecode
        found_bone =  filterJoint(pbone.name)
        if (found_bone != ""):
            pos2d = world_to_camera_view(bpy.context.scene, cam, ob.matrix_world @ pbone.head) 
            entry = entry + "," + ("%.3f" % (pos2d[0] * 1920))
            entry = entry + "," + ("%.3f" % ((1-pos2d[1]) * 1080))
    file.writelines(entry + "\n")
file.close()
print ("done")
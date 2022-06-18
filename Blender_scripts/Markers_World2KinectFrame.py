import mathutils
import bpy
from bpy_extras.object_utils import world_to_camera_view
import numpy as np


sce = bpy.context.scene
cam = bpy.data.objects['KinectCam']
ob = bpy.context.object


#IMPORTANT---> As the world frame is located in marker number 1, the location of the camera must be referenced from this "new" origin

print(ob.matrix_world)
print(type(ob.matrix_world))

    
#World frame located at marker number. Markers are defined according to the world frame
#Vector(x,y,z) --> Take into account that Y is the depth in this scene
marker_coordinates_world_frame = [
    mathutils.Vector((0.0, -0.178, 0.0)),
    mathutils.Vector((0.0, 0.0, 0.0)),
    mathutils.Vector((0.0, 0.189, 0.0)),
    mathutils.Vector((0.175, -0.208, 0.0)),
    mathutils.Vector((0.239, -0.037, 0.0)),
    mathutils.Vector((0.304, 0.115, 0.0)),
    mathutils.Vector((0.324, -0.278, 0.0)),
    mathutils.Vector((0.435, -0.164, 0.0)),
    mathutils.Vector((0.535, -0.06, 0.0)),
    mathutils.Vector((-0.178, -0.208, 0.0)),
    mathutils.Vector((-0.238, -0.038, 0.0)),
    mathutils.Vector((-0.303, 0.127, 0.0 )),
    mathutils.Vector((-0.325, -0.279, 0.0)),
    mathutils.Vector((-0.435, -0.138, 0.0 )),
    mathutils.Vector((-0.539, -0.038,0.0 ))
]



#print(marker_coordinates_world_frame)
#We just need to transform markers defined from world frame to Kinect frame
marker_coordinates_kinect_frame = []
for marker in marker_coordinates_world_frame: 
    marker_coordinates_kinect_frame.append(cam.matrix_world @ marker)

#The depth values should be inverted as they're not in the kinect orientation. 
print(marker_coordinates_kinect_frame)


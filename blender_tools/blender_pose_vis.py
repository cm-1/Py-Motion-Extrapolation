import bpy
import numpy as np
from numpy.typing import NDArray
import mathutils

# NOTE: The below code assumes poses are object-to-world and in row-order.
# It also assumes there is only one scene, and the keyframe-replacing function
# will DELETE ALL PRIOR KEYFRAMES!

# IN GENERAL, PLEASE READ THE CODE BEFORE RUNNING!

poses_py_list = [

]

poses = np.repeat([np.eye(4)], 33, axis=0) #np.array(poses_py_list)
poses[:,0,3] = np.arange(len(poses))

# poses[:,3,:3] *= 0.1

def createMatrixWorld(matrix: NDArray, rotCorrect: bool):
    # Matrix needs to be converted into a mathutils matrix or else Blender
    # interprets it as transposed. Which is confusing, because both numpy
    # and mathutils seem to use row major? Anyway, one could also just
    # transpose the numpy matrix but this seems "cleaner" and less likely
    # to break with future updates.
    matToSet = mathutils.Matrix(matrix)
    if rotCorrect:
        # Because Blender uses an "unconventional" Z-is-up coordinate
        # system, we might need to rotate our objects by -90 on X before
        # applying our transofmrations.
        matToSet = matToSet @ mathutils.Matrix.Rotation(-np.pi / 2, 4, 'X')
    return matToSet

def setAsKeyframesForObj(obj, matrices: NDArray, rotCorrect: bool = False):
    scene0 = bpy.data.scenes[0]

    # Clear all existing animations from the object
    obj.animation_data_clear()

    frameNum = scene0.frame_start

    for frameMat in matrices:
        if frameNum > scene0.frame_end:
            break
        bpy.context.scene.frame_set(frameNum)

        obj.matrix_world = createMatrixWorld(frameMat, rotCorrect)
        # In previous versions of Blender, using "rotation_euler" here seemed
        # to work, but now that doesn't work properly? Easy fix, at least... 
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frameNum)
        obj.keyframe_insert(data_path="location", frame=frameNum)
        frameNum += 1

    # Return Blender to 1st frame
    bpy.context.scene.frame_set(scene0.frame_start) 
    return

def duplicateObjects(obj, matrices: NDArray, rotCorrect: bool = False):
    all_copies = []
    for i, mat in enumerate(matrices):
        objCopy = obj.copy()
        objCopy.data = obj.data.copy()
        objCopy.name = obj.name + "_copy_" + str ( i )
        
        # We transpose first, since we're assuming that the poses passed in are
        # object-to-world matrices in row-major form.
        objCopy.matrix_world = createMatrixWorld(mat, rotCorrect)
        all_copies.append(objCopy)

        bpy.context.scene.collection.objects.link(objCopy)

    # See https://blender.stackexchange.com/questions/13986/how-to-join-objects-with-python
    # In a later version of Blender, only this worked: https://old.reddit.com/r/blenderhelp/comments/1dxpal9/how_to_join_meshes_by_material_efficiently/
    # Code that worked in an earlier blender version made a copy of the context,
    # but https://blender.stackexchange.com/questions/150989/blender-python-selections-and-context
    # suggests that copying the context might not be a good idea anyway?
    print("Joining objects...")
    
    bpy.ops.object.select_all(action='DESELECT') # Deselect all other objects first.
    for obj_copy in all_copies:
        obj_copy.select_set(True)
    bpy.context.view_layer.objects.active = all_copies[0]
    
    bpy.ops.object.join()
    obj.select_set(True)
    print("Done joining objects!")
    return

setAsKeyframesForObj(bpy.context.selected_objects[0], poses, True)
#duplicateObjects(bpy.context.selected_objects[0], poses)

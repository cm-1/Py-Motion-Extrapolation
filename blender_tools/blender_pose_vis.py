import bpy
import numpy as np


# NOTE: The below code assumes poses are object-to-world and in row-order.
# It also assumes there is only one scene, and the keyframe-replacing function
# will DELETE ALL PRIOR KEYFRAMES!

# IN GENERAL, PLEASE READ THE CODE BEFORE RUNNING!

poses_py_list = [

]

poses = np.repeat([np.eye(4)], 33, axis=0) #np.array(poses_py_list)
poses[:,0,3] = np.arange(len(poses))

# poses[:,3,:3] *= 0.1

def setAsKeyframesForObj(obj, matrices):
    scene0 = bpy.data.scenes[0]

    # Clear all existing animations from the object
    obj.animation_data_clear()

    frameNum = scene0.frame_start

    for frameMat in matrices:
        if frameNum > scene0.frame_end:
            break
        bpy.context.scene.frame_set(frameNum)

        # We transpose first, since we're assuming that the poses passed in are
        # object-to-world matrices in row-major form.
        matToSet = np.copy(frameMat.T)

        obj.matrix_world = matToSet
        obj.keyframe_insert(data_path="rotation_euler", index = -1)
        obj.keyframe_insert(data_path="location", index = -1)
        frameNum += 1

    # Return Blender to 1st frame
    bpy.context.scene.frame_set(scene0.frame_start) 
    return

def duplicateObjects(obj, matrices):
    all_copies = []
    for i, mat in enumerate(matrices):
        objCopy = obj.copy()
        objCopy.data = obj.data.copy()
        objCopy.name = obj.name + "_copy_" + str ( i )
        
        # We transpose first, since we're assuming that the poses passed in are
        # object-to-world matrices in row-major form.
        objCopy.matrix_world = np.copy(mat.T)
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

#setAsKeyframesForObj(bpy.context.selected_objects[0], poses)
duplicateObjects(bpy.context.selected_objects[0], poses[:35])

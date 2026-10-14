import os
import bpy
import mathutils
import numpy as np
import json

# def AddChildEmpty(root_obj, child_name='child', type='SPHERE', size=0.001):
#     bpy.ops.object.empty_add(type=type, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
#     child_obj = bpy.context.object
#     child_obj.name = child_name
#     child_obj.empty_display_size = size
    
#     bpy.ops.object.select_all(action='DESELECT')

#     bpy.context.view_layer.objects.active = root_obj
#     child_obj.select_set(True)
#     bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)

#     return child_obj


# def load_markers(fn):
    
#     markers = np.load(fn)

#     frame_cnt = len(markers)
    
#     start_frame = bpy.context.scene.frame_current

#     markers_cnt = len(markers[0])

#     print(f'> Frames: {frame_cnt} | Markers per Frame: {markers_cnt}')

#     bpy.ops.object.empty_add(type='ARROWS', radius=1, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))

#     root_obj = bpy.context.object
#     root_obj.empty_display_size = 0.1
#     root_obj.name = 'root'

#     marker_objs = []

#     # Create objects
#     for marker in range(markers_cnt):
#         marker_objs.append(AddChildEmpty(root_obj, f'marker_{marker}'))

#     for frame_idx in range(frame_cnt):
#         for marker_idx in range(markers_cnt):
#             print(markers[0][0])
#             if markers[frame_idx][0][0] == -1000:
#                 print(f'Invalid marker_idx {marker_idx} at frame_idx {frame_idx}')
#                 continue
#             if marker_idx > len(marker_objs)-1:
#                 print(f'No marker objects for indices >= {marker_idx}')
#                 break
#             marker_objs[marker_idx].location = markers[frame_idx][marker_idx]
#             marker_objs[marker_idx].keyframe_insert("location", frame=frame_idx+start_frame)


# def create_patch_empties(patch_coordinates):
#     frame_cnt = len(patch_coordinates)
#     start_frame = bpy.context.scene.frame_current
    
#     bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
#     root_obj = bpy.context.object
#     root_obj.name = 'patches'
#     root_obj.empty_display_size = 0.1

#     patches_obj = {}

#     for patch in patch_coordinates[0]:
#         patches_obj[patch] = AddChildEmpty(root_obj, patch, 'ARROWS', 0.02)
#         AddChildEmpty(patches_obj[patch], f'{patch}_offset', 'CIRCLE', 0.01)

#     for frame_idx in range(frame_cnt):
#         for patch in patch_coordinates[frame_idx]:
#             if len(patch_coordinates[frame_idx][patch][0]) != 0:
#                 loc = mathutils.Vector(patch_coordinates[frame_idx][patch][0][0])
#                 conf_loc = patch_coordinates[frame_idx][patch][0][1] if frame_idx > 1 else 1.0
                
#                 blended_loc = patches_obj[patch].location + (loc - patches_obj[patch].location)
#                 patches_obj[patch].location = blended_loc
#                 patches_obj[patch].keyframe_insert("location", frame=frame_idx+start_frame)
#             else:
#                 print(f'patch {patch} --> NONE LOC')
                
#             if len(patch_coordinates[frame_idx][patch][1]) != 0:  
#                 rot = patch_coordinates[frame_idx][patch][1][0]
#                 conf_rot = patch_coordinates[frame_idx][patch][1][1] if frame_idx > 1 else 1.0
                
#                 new_rot = np.asarray(rot)
#                 old_rot = np.asarray(patches_obj[patch].matrix_world.to_3x3())
                
#                 blended_rot = old_rot + (new_rot - old_rot) * conf_rot
                
#                 patches_obj[patch].matrix_world = mathutils.Matrix(blended_rot).to_4x4()
#                 patches_obj[patch].keyframe_insert("rotation_euler", frame=frame_idx+start_frame)
#             else:
#                 print(f'patch {patch} --> NONE ROT')
#         patches_obj[patch].select_set(True)
                
#         # print(f'> Frame {frame_idx} Patch {patch}\n\tLocation: {loc}\n\tRotation: {rot}\n')
#     prev_type = bpy.context.area.type
                    
#     bpy.context.area.type = 'GRAPH_EDITOR'
#     bpy.ops.graph.select_all(action='SELECT')
#     bpy.ops.graph.euler_filter()
#     bpy.ops.graph.gaussian_smooth(factor=1.0, sigma=1.0, filter_width=3)
#     bpy.ops.graph.select_all(action='DESELECT')
#     bpy.context.area.type = prev_type
        
#     # bpy.ops.object.select_all(action='DESELECT')

#     return patches_obj

def CreateHierarchy(root_obj, data, parent = None):
    name = data['joint']
    b = root_obj.data.edit_bones.new(name)
    b.length = 0.01
    if parent:
        b.parent = parent
    #b.use_connect = True
    
    for c in data['children']:
        CreateHierarchy(root_obj, c, b) 

def load_hierarchy(hierarchy):
    arms = []
        
    for arm in all_hierarchy:
        bpy.ops.object.armature_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        root_obj = bpy.context.object
        root_obj.name = arm['arm_name']
        arms.append(root_obj)
        
        root_obj.show_in_front = True
        root_obj.data.show_names = True
        root_obj.data.show_axes = True
        #root_obj.data.display_type = 'STICK'
        
        bpy.ops.object.mode_set(mode='EDIT')
        root_obj.data.edit_bones.remove(root_obj.data.edit_bones[0])
        
        CreateHierarchy(root_obj, arm['hierarchy'])
        
        bpy.ops.object.mode_set(mode='OBJECT')
    return arms

def find_joint_in_hierarchy(e, joint_name):
    if e['joint'] == joint_name:
        return e
    for c in e['children']:
        j = find_joint_in_hierarchy(c, joint_name)
        if j:
            return j
    return None

def apply_patch_objects(obj, patch_objs, patch_coordinates, hierarchy):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='POSE')
    
    constraints = {}
    
    for pb in obj.pose.bones:
        if pb == obj.data.bones[0]:
            continue
        
        joint = find_joint_in_hierarchy(hierarchy, pb.name)
            
        if joint == None:
            continue
            
        if len(joint['influences']) > 0:
            for patch_name, weight in joint['influences'].items(): 
                l_const = pb.constraints.new("COPY_LOCATION")
                l_const.target = patch_objs[patch_name]
                l_const.influence = weight
                
                r_const = pb.constraints.new("COPY_ROTATION")
                r_const.target = patch_objs[patch_name]
                r_const.influence = weight
    bpy.ops.object.mode_set(mode='OBJECT')
        

if __name__ == '__main__':
    #markers_fn = fn = 'P:/Projects/Sol/Prototypes/sol-dextercap/Out/synth_06/post_process/all_markers_3d_positions_interpolated.npy'
    #load_markers(markers_fn)
    
    patch_coordinates_fn = 'P:/Projects/Sol/Prototypes/sol-dextercap/Out/synth_06/triangulation/all_patches_coordinates.json'
    
    with open(patch_coordinates_fn) as f:
        patch_coordinates = json.load(f)
        
    hierarchy_fn = 'P:/Projects/Sol/Prototypes/sol-dextercap/Out/synth_05/patches/all_armatures.json'
    with open(hierarchy_fn) as f: 
        all_hierarchy = json.load(f)
        
    patch_objs = create_patch_empties(patch_coordinates)
    #arms = load_hierarchy(all_hierarchy)
    #for i in range(len(arms)):
    #    apply_patch_objects(arms[i], patch_objs,patch_coordinates, all_hierarchy[i]['hierarchy'])
    

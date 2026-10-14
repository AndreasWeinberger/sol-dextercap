import bpy
import mathutils
import numpy as np

bone_name = 'thumb_02_r'

bone_world_matrix = bpy.data.objects['root'].matrix_world @ bpy.data.armatures[0].bones[bone_name].matrix_local

labels = [b.name for b in bpy.context.selected_objects[0:len(bpy.context.selected_objects)-1]]

output_matrices = []

#combined_offset_matrix = mathutils.Matrix([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

for l in labels:
    bl_matrix = bpy.data.objects[l].matrix_world
    offset_matrix = bpy.data.objects[l].matrix_world.inverted() @ bone_world_matrix
    #combined_offset_matrix += offset_matrix
    out = bl_matrix @ offset_matrix
    output_matrices.append(out)
    print(f'label {l}\noffset_matrix{offset_matrix}')
    print(f'\bone_matrix: {bone_world_matrix}')
    print(f'\block_matrix: {bl_matrix}')
    
#combined_offset_matrix *= 1.0 / len(labels)
    
combined_output_matrix = mathutils.Matrix([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    
for a in output_matrices:
    combined_output_matrix += a
    
combined_output_matrix *= 1.0 / len(output_matrices)

bpy.context.object.matrix_world = combined_output_matrix
bpy.context.object.scale = (1,1,1)

#print(combined_offset_matrix)
import bpy
import json


with open("P:\Projects\Sol\Prototypes\sol-dextercap\Out\synth_06\patches/all_patch_definitions.json") as f:
    patches_definition = json.load(f)
    
marker_defs = patches_definition['marker_defs']
blocks = patches_definition['blocks']

ob = bpy.context.object

assert len(marker_defs) == len(ob.children), f"Count mismatch: marker_defs {len(marker_defs)} but {len(ob.children)} children"

done_blocks = set()

for marker_idx, marker_obj in enumerate(ob.children):
    if marker_obj.location[0] != -1000:
        marker_def = marker_defs[marker_idx]
        for label in marker_def:
            block = blocks[label[0]]
            corners = block['markers']
            patch = block['patch']
            
            for c in corners:
                ob.children[c].select = True
            break
    break
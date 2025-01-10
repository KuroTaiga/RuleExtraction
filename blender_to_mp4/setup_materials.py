import bpy
import sys
import os

argv = sys.argv
try:
    # Find the custom output folder argument (after "--")
    output_folder_index = argv.index("--") + 1
    output_folder = argv[output_folder_index]
except (ValueError, IndexError):
    # Default to the .blend file's directory if no folder is provided
    output_folder = "Blender_With_Material"
blend_file_path = bpy.data.filepath
blend_file_name = os.path.basename(blend_file_path)
blend_name = os.path.splitext(blend_file_name)[0]  
blend_name = blend_name.replace(" ", "_")
output_path = os.path.join(output_folder,blend_file_name)

def create_material(name, color):
    """Create a new material"""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    output = nodes.new('ShaderNodeOutputMaterial')
    
    principled.inputs['Base Color'].default_value = color
    principled.inputs['Roughness'].default_value = 0.7
    
    mat.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    return mat

def assign_material_to_parts(obj, part_names, material, material_index):
    """Assign material to specified body parts"""
    bpy.context.view_layer.objects.active = obj
    
    # Enter edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Select vertices
    for vg in obj.vertex_groups:
        base_name = vg.name.replace("mixamorig:", "")
        if base_name in part_names:
            for v in obj.data.vertices:
                for g in v.groups:
                    if g.group == vg.index:
                        v.select = True
    
    # Assign material
    bpy.ops.object.mode_set(mode='EDIT')
    obj.active_material_index = material_index
    bpy.ops.object.material_slot_assign()
    bpy.ops.object.mode_set(mode='OBJECT')

def setup_materials():
    """Set up all materials for the character"""
    # Find the body model
    body = None
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and 'Man Body' in obj.name:
            body = obj
            break
    
    if not body:
        print("Body model not found")
        return
    
    # Clear existing materials
    while len(body.data.materials) > 0:
        body.data.materials.pop()
    
    # Create new materials
    materials = {
        'skin': create_material("Skin", (0.8, 0.6, 0.5, 1)),    # Skin color
        'shirt': create_material("Shirt", (0.2, 0.3, 0.8, 1)),  # Blue
        'pants': create_material("Pants", (0.2, 0.2, 0.2, 1)),  # Black
        'shoes': create_material("Shoes", (0.1, 0.1, 0.1, 1))   # Dark black
    }
    
    # Add materials to object
    for mat in materials.values():
        body.data.materials.append(mat)
    
    # Define vertex group names for each body part
    parts = {
        'skin_head': [
            'Head', 'Neck',
            'LeftHand', 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3',
            'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3',
            'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3',
            'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3',
            'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3',
            'RightHand', 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3',
            'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3',
            'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3',
            'RightHandRing1', 'RightHandRing2', 'RightHandRing3',
            'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3'
        ],
        'shirt': [
            'Spine', 'Spine1', 'Spine2',
            'LeftShoulder', 'LeftArm', 'LeftForeArm',
            'RightShoulder', 'RightArm', 'RightForeArm'
        ],
        'pants': [
            'LeftUpLeg', 'LeftLeg',
            'RightUpLeg', 'RightLeg'
        ],
        'shoes': [
            'LeftFoot', 'LeftToeBase',
            'RightFoot', 'RightToeBase'
        ]
    }
    
    # Assign materials
    assign_material_to_parts(body, parts['skin_head'], materials['skin'], 0)
    assign_material_to_parts(body, parts['shirt'], materials['shirt'], 1)
    assign_material_to_parts(body, parts['pants'], materials['pants'], 2)
    assign_material_to_parts(body, parts['shoes'], materials['shoes'], 3)
    
    print("Material assignment completed!")
    
    # Verify assignments
    print("\nCurrent materials:")
    for i, mat in enumerate(body.data.materials):
        if mat.use_nodes:
            color = mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value
            print(f"[{i}] {mat.name}: RGB = {[round(c, 3) for c in color[:3]]}")
    bpy.ops.wm.save_as_mainfile(filepath=output_path)
# Run main function
setup_materials()
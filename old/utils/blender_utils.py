import bpy
import os
import json
import shutil

out_mesh_thickness = .7  # This was manually chosen for our objects, and the Photon printer (SLA we used)

def blender_to_stl_set(in_blend, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    bpy.ops.wm.open_mainfile(filepath=in_blend)
    context = bpy.context
    scene = context.scene
    viewlayer = context.view_layer
    obs = [o for o in scene.objects if o.type == 'MESH']
    bpy.ops.object.select_all(action='DESELECT')

    for ob in obs:
        viewlayer.objects.active = ob
        ob.select_set(True)
        obj_path = os.path.join(out_dir, ob.name + '.stl')
        bpy.ops.export_mesh.stl(
            filepath=str(obj_path),
            use_selection=True)
        ob.select_set(False)

def blenders_to_obj(blender_dir):
    blender_path_list = [os.path.join(blender_dir, f) for f in os.listdir(blender_dir) if f.endswith('.blend')]
    for blend in blender_path_list:
        bpy.ops.wm.open_mainfile(filepath=blend)
        bpy.ops.export_scene.obj(filepath=blend.replace('.blend', '.obj'))

def clear_scene():
    to_del = [obj.name for obj in bpy.context.scene.objects]
    print(to_del)
    for name in to_del:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects[name].select_set(True)  # Blender 2.8x
        bpy.ops.object.delete()

def blender_post_process(results_dir):
    stls = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('.stl')]
    process_dir = os.path.join(results_dir, 'processed')
    if not os.path.isdir(process_dir):
        os.makedirs(process_dir)
    for stl_path in stls:
        out_path = os.path.join(process_dir, os.path.basename(stl_path))
        clear_scene()
        if 'carved' in os.path.basename(stl_path):
            bpy.ops.import_mesh.stl(filepath=stl_path)
            bpy.ops.object.modifier_add(type='SOLIDIFY')
            bpy.context.object.modifiers["Solidify"].thickness = out_mesh_thickness
            bpy.ops.export_mesh.stl(filepath=os.path.join(out_path))
        else:
            shutil.copy2(stl_path, out_path)


if __name__ == '__main__':
    # blend_input = './resources/RUJUM_4.blend'
    # blender_to_stl_set(blend_input, blend_input.replace('.blend', ''))

    blender_post_process('./results/RUJUM_4')
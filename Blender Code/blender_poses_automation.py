import bpy
import math
import random 
import numpy as np
## List of camera positions (x, y, z) and rotations (in degrees)

# Parameters
num_cameras = 2  # Number of cameras to place around the object

camera_positions = []


############# For Center Percpective depth map

#location = [(1, 0 ,5)]

#location = [(0, 2 ,5)]
#rotation = [(0,0,0)]

############# For top-down grid

location = [
(-2.25, 2.25, 5), (-1.75, 2.25, 5), (-1.25, 2.25, 5), (-0.75, 2.25, 5), (-0.25, 2.25, 5), 
(0.25, 2.25, 5), (0.75, 2.25, 5), (1.25, 2.25, 5), (1.75, 2.25, 5), (2.25, 2.25, 5),

(-2.25, 1.75, 5), (-1.75, 1.75, 5), (-1.25, 1.75, 5), (-0.75, 1.75, 5), (-0.25, 1.75, 5), 
(0.25, 1.75, 5), (0.75, 1.75, 5), (1.25, 1.75, 5), (1.75, 1.75, 5), (2.25, 1.75, 5),

(-2.25, 1.25, 5), (-1.75, 1.25, 5), (-1.25, 1.25, 5), (-0.75, 1.25, 5), (-0.25, 1.25, 5), 
(0.25, 1.25, 5), (0.75, 1.25, 5), (1.25, 1.25, 5), (1.75, 1.25, 5), (2.25, 1.25, 5),

(-2.25, 0.75, 5), (-1.75, 0.75, 5), (-1.25, 0.75, 5), (-0.75, 0.75, 5), (-0.25, 0.75, 5), 
(0.25, 0.75, 5), (0.75, 0.75, 5), (1.25, 0.75, 5), (1.75, 0.75, 5), (2.25, 0.75, 5),

(-2.25, 0.25, 5), (-1.75, 0.25, 5), (-1.25, 0.25, 5), (-0.75, 0.25, 5), (-0.25, 0.25, 5), 
(0.25, 0.25, 5), (0.75, 0.25, 5), (1.25, 0.25, 5), (1.75, 0.25, 5), (2.25, 0.25, 5),

(-2.25, -0.25, 5), (-1.75, -0.25, 5), (-1.25, -0.25, 5), (-0.75, -0.25, 5), (-0.25, -0.25, 5), 
(0.25, -0.25, 5), (0.75, -0.25, 5), (1.25, -0.25, 5), (1.75, -0.25, 5), (2.25, -0.25, 5),

(-2.25, -0.75, 5), (-1.75, -0.75, 5), (-1.25, -0.75, 5), (-0.75, -0.75, 5), (-0.25, -0.75, 5), 
(0.25, -0.75, 5), (0.75, -0.75, 5), (1.25, -0.75, 5), (1.75, -0.75, 5), (2.25, -0.75, 5),

(-2.25, -1.25, 5), (-1.75, -1.25, 5), (-1.25, -1.25, 5), (-0.75, -1.25, 5), (-0.25, -1.25, 5), 
(0.25, -1.25, 5), (0.75, -1.25, 5), (1.25, -1.25, 5), (1.75, -1.25, 5), (2.25, -1.25, 5),

(-2.25, -1.75, 5), (-1.75, -1.75, 5), (-1.25, -1.75, 5), (-0.75, -1.75, 5), (-0.25, -1.75, 5), 
(0.25, -1.75, 5), (0.75, -1.75, 5), (1.25, -1.75, 5), (1.75, -1.75, 5), (2.25, -1.75, 5),

(-2.25, -2.25, 5), (-1.75, -2.25, 5), (-1.25, -2.25, 5), (-0.75, -2.25, 5), (-0.25, -2.25, 5), 
(0.25, -2.25, 5), (0.75, -2.25, 5), (1.25, -2.25, 5), (1.75, -2.25, 5), (2.25, -2.25, 5)
]

rotation = [(0,0,0)] * 100


#default was 13.9
bpy.data.materials["Corn_Shd"].node_tree.nodes["Principled BSDF"].inputs[3].default_value = 1.1


# Get the source object
source_object1 = bpy.data.objects.get("Corn_AgeMedium_A")
source_object2 = bpy.data.objects.get("Corn_AgeYoung_A")
source_object3 = bpy.data.objects.get("Corn_AgeMature_A")

source_list = [source_object1, source_object2, source_object3]
random_source = random.randint(0, 2)

density = 3 # leave 10 meters between each col
damage = 2 #how many damage corn in a col 



location_range = (-7, 7)
source_object = source_list[random_source]

if source_object is None:
    print(f"Object Corn_Shd not found.")
    
else:
    x_start, x_end = -7, 7
    y_start, y_end = 7, -7

    x_step = .7 * density # steps will determine the density
    y_step = -0.7 # steps will determine the density
    
    i = 0
    for y in np.arange(y_start, y_end + -1, y_step):
        random_damage_corns = random.sample(list(np.arange(-7, 7, x_step)), damage)
        
        for x in np.arange(x_start, x_end + 1, x_step):
            
            
            random_source = random.randint(0, 2)
            source_object = source_list[random_source]
            
            # Copy the object and its data
            new_obj = source_object.copy()
            new_obj.data = source_object.data.copy()
            new_obj.name = f"Corn_copy_{i}"

            # Randomize location, rotation, and scale
            new_obj.location = (
                x,
                y,
                0
            )

                
            if x in random_damage_corns:
                new_obj.rotation_euler = (
                    math.radians(random.randint(-70, 70)),
                    math.radians(random.randint(-70, 70)),
                    0
                )
            else:
                            
                new_obj.rotation_euler = (
                        0,
                        0,
                        0
                    )
                

            
            #random_scale = random.uniform(*scale_range)
            #new_obj.scale = (random_scale, random_scale, random_scale)

            # Link the object to the current collection
            bpy.context.collection.objects.link(new_obj)
            i += 1
            # Add a Subdivision Surface modifier with increasing complexity
            '''
            if i > 0:
                mod = new_obj.modifiers.new(name="Subsurf", type='SUBSURF')
                mod.levels = min(subdivision_levels[1], subdivision_levels[0] + i)
                mod.render_levels = mod.levels
            '''
            # Optionally parent to the original for hierarchy
            #new_obj.parent = source_object if i % 2 == 0 else None

bpy.ops.object.light_add(type='SUN')
sun = bpy.context.object
sun.name = "SunLight"
sun.data.energy = 5.0  # Adjust this value for intensity (e.g., 10.0 for sunny, 2.0 for cloudy)
bpy.context.object.data.use_shadow = False

clip_start = 1 #default 1 - 2
clip_end = 100 #default 100 - 5.1
slice_spacing = 0.0 #default 0.03
## Loop to generate camera positions around the object
for i in range(num_cameras):
    # Append camera position and rotation
    camera_positions.append(
        {
            "location": location[i],  # Camera position default i instead of 0
            "rotation": rotation[i],  # Rotate to face the object x tilt, y roll, z yaw/compass
            "fov": 45,  # Field of view
            "clip_start": clip_start,  # Near clipping plane
            "clip_end": clip_end,  # Far clipping plane
            "aspect_ratio": (1, 1),  # Aspect ratio
        }
    )
    clip_start += slice_spacing
    clip_end += slice_spacing
        



# Loop through the camera positions and create cameras
for i, cam_data in enumerate(camera_positions):
   # Clear existing cameras
   bpy.ops.object.select_by_type(type='CAMERA')
   bpy.ops.object.delete()

   # Create a new scene or use the existing one
   scene = bpy.context.scene
    
   # Enable depth rendering in the compositor
   scene.use_nodes = True
   tree = scene.node_tree
   nodes = tree.nodes

   # Clear default nodes
   for node in nodes:
       nodes.remove(node)

   # Create input render layer node
   render_layers = nodes.new(type='CompositorNodeRLayers')

   # Create a normalize node to scale depth values
   normalize_node = nodes.new(type='CompositorNodeNormalize')
   normalize_node.inputs[0].default_value = 1  # Ensure normalization is applied


   # Create output file node for depth
   depth_file_output = nodes.new(type='CompositorNodeOutputFile')
   depth_file_output.base_path = r"C:\Users\cgpc2\Desktop\Blender\Mohamed\test_random\Depth\\"
   
   depth_file_output.file_slots[0].path = "depth####"
   depth_file_output.format.file_format = 'OPEN_EXR'  # Save depth map in OpenEXR format
   depth_file_output.format.color_depth = '32'  # Use 32-bit float for high precision


   #depth_file_output.format.file_format = 'TIFF'  # Save depth map as TIFF
   #depth_file_output.format.color_depth = '16'  # Use 16-bit depth
   #depth_file_output.format.color_mode = 'BW'  # Use 16-bit depth
   #depth_file_output.format.tiff_codec = 'NONE'  # Disable compression


   #tree.links.new(render_layers.outputs['Depth'], normalize_node.inputs['Value'])
   #tree.links.new(normalize_node.outputs['Value'], depth_file_output.inputs['Image'])
   tree.links.new(render_layers.outputs['Depth'], depth_file_output.inputs['Image'])
   
   # Create a new camera
   bpy.ops.object.camera_add(location=cam_data["location"])
   camera = bpy.context.object
   camera.name = f"Camera_{i+1}"
   
   # Set camera rotation (convert degrees to radians)
   camera.rotation_euler = (
       math.radians(cam_data["rotation"][0]),
       math.radians(cam_data["rotation"][1]),
       math.radians(cam_data["rotation"][2]),
   )
   
   # Apply this rotation matrix to the camera
   #camera.rotation_mode = 'QUATERNION'  # Using quaternion for rotation
   #camera.rotation_quaternion = rotation_matrix.to_quaternion()  # Apply rotation
   
   # Set the field of view (FOV) for the camera
   camera.data.angle = math.radians(cam_data["fov"])  # Convert FOV from degrees to radians
   
       # Set the aspect ratio of the render
   aspect_ratio = cam_data["aspect_ratio"]
   scene.render.resolution_x = 1024  # Set base resolution
   scene.render.resolution_y = int(1024 * (aspect_ratio[1] / aspect_ratio[0]))  # Maintain aspect ratio
   
   # Set near and far clipping planes
   camera.data.clip_start = cam_data["clip_start"]
   camera.data.clip_end = cam_data["clip_end"]

   # Set the camera as the active camera for rendering
   scene.camera = camera
   
   # Render the scene and save the image
   render_path = r"C:\Users\cgpc2\Desktop\Blender\Mohamed\test_random\Images\\"+str(i+1)+".png"
   scene.render.filepath = render_path
   bpy.ops.render.render(write_still=True)
   
   # Render the depth map
   depth_file_output.file_slots[0].path = f"depth_camera_{i+1}_####"
   bpy.ops.render.render(write_still=True)

print("Rendering completed for all cameras.")

"""
References: https://github.com/federicoarenasl/Data-Generation-with-Blender/blob/master/Resources/main_script.py
"""

import bpy
import json
import os
import math
import numpy as np
import math as m
import random

class Render:
    def __init__(self):
        self.scene = bpy.data.scenes['Scene']
        self.scene.use_nodes = False
        # Define the objects to be used in the scene
        self.camera = bpy.data.objects['Camera']
        self.axis = bpy.data.objects['Empty']
        self.sun = bpy.data.objects['Sun']
        self.create_objects() # Create list of bpy.data.objects from bpy.data.objects[1] to bpy.data.objects[N]
        self.xpix = 1000
        self.ypix = 1000
        self.percentage = 100
        self.samples = 250

        
        self.images_filepath = './solution/generated_images/images'
        self.labels_filepath = './solution/generated_images/labels'
        self.coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.category_id_map = {}
        self.annotation_id = 1


        if not os.path.exists(self.images_filepath):
            os.makedirs(self.images_filepath)

        if not os.path.exists(self.labels_filepath):
            os.makedirs(self.labels_filepath)

    def get_all_coordinates(self):
        '''
        This function takes no input and outputs the complete string with the coordinates
        of all the objects in view in the current image
        '''
        main_text_coordinates = '' # Initialize the variable where we'll store the coordinates
        for i, objct in enumerate(self.objects): # Loop through all of the objects
            if objct.name in ['Empty', 'Camera', 'Sun', 'Plane', 'Cube', 'TextureField']:
                continue
            b_box = self.find_bounding_box(objct)
            if b_box:
                text_coordinates = self.format_coordinates(b_box, i, objct.name)
                main_text_coordinates = main_text_coordinates + text_coordinates
            else:
                pass

        return main_text_coordinates

    def format_coordinates(self, coordinates, classe, label):
        ## Change coordinates reference frame
        x1 = (coordinates[0][0])
        x2 = (coordinates[1][0])
        y1 = (1 - coordinates[1][1])
        y2 = (1 - coordinates[0][1])
        txt_coordinates = f"{str(classe)}, {label}, {x1 * self.xpix}, {y1 * self.ypix}, {x2 * self.xpix}, {y2 * self.ypix}\n"
        return txt_coordinates

    def find_bounding_box(self, obj):
        matrix = self.camera.matrix_world.normalized().inverted()
        """ Create a new mesh data block, using the inverse transform matrix to undo any transformations. """
        try:
            mesh = obj.to_mesh(preserve_all_data_layers=True)
        except:
            print(obj.name)
        mesh.transform(obj.matrix_world)
        mesh.transform(matrix)

        """ Get the world coordinates for the camera frame bounding box, before any transformations. """
        frame = [-v for v in self.camera.data.view_frame(scene=self.scene)[:3]]

        lx = []
        ly = []

        for v in mesh.vertices:
            co_local = v.co
            z = -co_local.z

            if z <= 0.0:
                """ Vertex is behind the camera; ignore it. """
                continue
            else:
                """ Perspective division """
                frame = [(v / (v.z / z)) for v in frame]

            min_x, max_x = frame[1].x, frame[2].x
            min_y, max_y = frame[0].y, frame[1].y

            x = (co_local.x - min_x) / (max_x - min_x)
            y = (co_local.y - min_y) / (max_y - min_y)

            lx.append(x)
            ly.append(y)


        """ Image is not in view if all the mesh verts were ignored """
        if not lx or not ly:
            return None

        min_x = np.clip(min(lx), 0.0, 1.0)
        min_y = np.clip(min(ly), 0.0, 1.0)
        max_x = np.clip(max(lx), 0.0, 1.0)
        max_y = np.clip(max(ly), 0.0, 1.0)

        """ Image is not in view if both bounding points exist on the same side """
        if min_x == max_x or min_y == max_y:
            return None

        """ Figure out the rendered image size """
        render = self.scene.render
        fac = render.resolution_percentage * 0.01
        dim_x = render.resolution_x * fac
        dim_y = render.resolution_y * fac
        
        ## Verify there's no coordinates equal to zero
        coord_list = [min_x, min_y, max_x, max_y]
        if min(coord_list) == 0.0:
            indexmin = coord_list.index(min(coord_list))
            coord_list[indexmin] = coord_list[indexmin] + 0.0000001

        return (min_x, min_y), (max_x, max_y)
    
    def collect_coco_data(self):
        image_id = self.counter
        image_filename = f"{image_id}.png"
        self.coco_data["images"].append({
            "id": image_id,
            "width": self.xpix,
            "height": self.ypix,
            "file_name": image_filename
        })

        text_coordinates = self.get_all_coordinates().split('\n')[:-1]
        for line in text_coordinates:
            parts = line.split(', ')
            class_id, label, x1, y1, x2, y2 = parts
            if label not in self.category_id_map:
                self.category_id_map[label] = len(self.category_id_map) + 1
                self.coco_data["categories"].append({
                    "id": self.category_id_map[label],
                    "name": label,
                    "supercategory": "none"
                })
            bbox = [float(x1), float(y1), float(x2) - float(x1), float(y2) - float(y1)]
            self.coco_data["annotations"].append({
                "id": self.annotation_id,
                "image_id": image_id,
                "category_id": self.category_id_map[label],
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })
            self.annotation_id += 1

    def generate_coco_json(self):
        output_path = os.path.join(self.labels_filepath, "coco_annotations.json")
        with open(output_path, 'w') as f:
            json.dump(self.coco_data, f, indent=4)

    def main_rendering_loop(self, rotation_step):
        self.counter = 0
        full_circle = math.pi * 2
        num_images = int(360 / rotation_step)

        for i in range(10):
            self.shuffle_objects_with_margin(
                margin=1,
                x_range=(-3, 3),
                y_range=(-3, 3),
                z_range=(0, 0)
                )
            self.sun.data.energy = random.randint(1, 10)  # Brightness of the light
            self.sun.data.angle = np.random.uniform(0.1, math.pi)  # Width of the light

            for i in range(num_images):
                self.hide_some_objects()
                self.random_rotate_objects()
                self.sun.location = (random.randint(0, 10), random.randint(0, 10), random.randint(0, 10))
                self.sun.rotation_euler = (math.radians(45), 0, math.radians(-45))
                angle = full_circle * (i / num_images)
                self.axis.rotation_euler = (0, 0, angle)
                self.render_blender()
                self.collect_coco_data()
        self.generate_coco_json()
                
    def render_blender(self):
        self.counter += 1
        bpy.context.scene.cycles.samples = self.samples
        self.scene.render.resolution_x = self.xpix
        self.scene.render.resolution_y = self.ypix
        self.scene.render.resolution_percentage = self.percentage
        self.scene.render.filepath = os.path.join(self.images_filepath, f"{self.counter}.png")
        bpy.ops.render.render(write_still=True)

    
    # Function to check if a new position is too close to existing objects
    def is_too_close(self, obj, new_position, margin):
        for other_obj in bpy.data.objects:
            if other_obj != obj and 'can' in other_obj.name:  # Check for your specific object naming
                distance = (other_obj.location - new_position).length
                if distance < margin:
                    return True
        return False

    # Function to shuffle the locations of existing objects with a margin
    def shuffle_objects_with_margin(self, margin, x_range, y_range, z_range):
        random.shuffle(self.objects)  # Shuffle the list of objects
        
        for obj in self.objects:
            # Try to find a new location for each object
            for attempt in range(100):  # Avoid infinite loop
                new_position = bpy.context.scene.cursor.location.copy()
                new_position.x = random.uniform(*x_range)
                new_position.y = random.uniform(*y_range)
                new_position.z = random.uniform(*z_range)
                
                # If the new position is not too close to other objects, move the object there
                if not self.is_too_close(obj, new_position, margin):
                    obj.location = new_position
                    break  # Break out of the loop if a valid location was found

    def hide_some_objects(self, probability=0.5):
        for obj in self.objects:
            obj.hide_render = False  # Make all objects visible by default
            do_hide = random.random() > probability  # Randomly decide whether to hide the object
            if obj.name not in ['Empty', 'Camera', 'Sun', 'Plane', 'Cube', 'TextureField'] and do_hide:
                obj.hide_render = True

    def random_rotate_objects(self):
        for obj in self.objects:
            if obj.name not in ['Empty', 'Camera', 'Sun', 'Plane', 'Cube', 'TextureField']:
                obj.rotation_euler[2] = random.uniform(0, 2 * math.pi)

    def create_objects(self):  # This function creates a list of all the <bpy.data.objects>
        self.objects = []
        for i, obj in enumerate(bpy.data.objects, start=1):
            if obj.name in ['Empty', 'Camera', 'Sun', 'Plane', 'Cube', 'TextureField']:
                continue

            print(f"Object {i}: {obj.name}")
            self.objects.append(obj)

if __name__ == '__main__':
    r = Render()
    r.main_rendering_loop(rotation_step=10)
# """
# StandardArena with walls and domain randomization support
# File: arena/arena.py
# """
# from dm_control import mjcf
# import numpy as np


# class StandardArena(object):
#     def __init__(self, randomize=True) -> None:
#         """
#         Initializes the StandardArena with floor, walls, and lights.
        
#         Args:
#             randomize: If True, randomize floor and wall colors
#         """
#         self._mjcf_model = mjcf.RootElement()
#         self._mjcf_model.option.timestep = 0.002
#         self._mjcf_model.option.flag.warmstart = "enable"
#         self._mjcf_model.option.integrator = "implicitfast"
#         self._mjcf_model.option.solver = "CG"
        
#         # Randomize colors if enabled
#         if randomize:
#             floor_rgb1, floor_rgb2 = self._random_floor_colors()
#             wall_rgba = self._random_wall_color()
#         else:
#             floor_rgb1 = [0.2, 0.3, 0.4]
#             floor_rgb2 = [0.3, 0.4, 0.5]
#             wall_rgba = [0.8, 0.8, 0.8, 1.0]
        
#         # Floor texture
#         chequered = self._mjcf_model.asset.add(
#             "texture",
#             type="2d",
#             builtin="checker",
#             width=300,
#             height=300,
#             rgb1=floor_rgb1,
#             rgb2=floor_rgb2,
#         )
#         grid = self._mjcf_model.asset.add(
#             "material",
#             name="grid",
#             texture=chequered,
#             texrepeat=[5, 5],
#             reflectance=0.2,
#         )
        
#         # Floor plane
#         self._mjcf_model.worldbody.add(
#             "geom", 
#             type="plane", 
#             size=[3, 3, 0.1], 
#             material=grid
#         )
        
#         # Add 4 walls to create indoor environment
#         wall_height = 1.5
#         wall_thickness = 0.1
#         room_size = 3.0
        
#         # Wall material
#         wall_mat = self._mjcf_model.asset.add(
#             "material",
#             name="wall_material",
#             rgba=wall_rgba,
#             reflectance=0.1
#         )
        
#         # North wall (y = room_size)
#         self._mjcf_model.worldbody.add(
#             "geom",
#             type="box",
#             size=[room_size, wall_thickness, wall_height],
#             pos=[0, room_size, wall_height],
#             material=wall_mat,
#             contype="1",
#             conaffinity="1"
#         )
        
#         # South wall (y = -room_size)
#         self._mjcf_model.worldbody.add(
#             "geom",
#             type="box",
#             size=[room_size, wall_thickness, wall_height],
#             pos=[0, -room_size, wall_height],
#             material=wall_mat,
#             contype="1",
#             conaffinity="1"
#         )
        
#         # East wall (x = room_size)
#         self._mjcf_model.worldbody.add(
#             "geom",
#             type="box",
#             size=[wall_thickness, room_size, wall_height],
#             pos=[room_size, 0, wall_height],
#             material=wall_mat,
#             contype="1",
#             conaffinity="1"
#         )
        
#         # West wall (x = -room_size)
#         self._mjcf_model.worldbody.add(
#             "geom",
#             type="box",
#             size=[wall_thickness, room_size, wall_height],
#             pos=[-room_size, 0, wall_height],
#             material=wall_mat,
#             contype="1",
#             conaffinity="1"
#         )
        
#         # Lights
#         for x in [-2, 2]:
#             self._mjcf_model.worldbody.add(
#                 "light", 
#                 pos=[x, -1, 3], 
#                 dir=[-x, 1, -2]
#             )
    
#     def _random_floor_colors(self):
#         """Generate random floor checker colors"""
#         # Random base color
#         base_h = np.random.uniform(0, 360)
#         base_s = np.random.uniform(0.2, 0.5)
#         base_v = np.random.uniform(0.3, 0.6)
        
#         rgb1 = self._hsv_to_rgb(base_h, base_s, base_v)
#         rgb2 = self._hsv_to_rgb(base_h + 20, base_s + 0.1, base_v + 0.1)
        
#         return list(rgb1), list(rgb2)
    
#     def _random_wall_color(self):
#         """Generate random wall color"""
#         # Neutral-ish walls
#         r = np.random.uniform(0.6, 0.9)
#         g = np.random.uniform(0.6, 0.9)
#         b = np.random.uniform(0.6, 0.9)
#         return [r, g, b, 1.0]
    
#     def _hsv_to_rgb(self, h, s, v):
#         """Convert HSV to RGB"""
#         h = h / 360.0
#         c = v * s
#         x = c * (1 - abs((h * 6) % 2 - 1))
#         m = v - c
        
#         if h < 1/6:
#             r, g, b = c, x, 0
#         elif h < 2/6:
#             r, g, b = x, c, 0
#         elif h < 3/6:
#             r, g, b = 0, c, x
#         elif h < 4/6:
#             r, g, b = 0, x, c
#         elif h < 5/6:
#             r, g, b = x, 0, c
#         else:
#             r, g, b = c, 0, x
        
#         return r + m, g + m, b + m
    
#     def attach(self, child, pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0]) -> mjcf.Element:
#         """
#         Attaches a child element to the MJCF model at a specified position and orientation.
        
#         Args:
#             child: The child element to attach.
#             pos: The position of the child element.
#             quat: The orientation of the child element.
            
#         Returns:
#             The frame of the attached child element.
#         """
#         frame = self._mjcf_model.attach(child)
#         frame.pos = pos
#         frame.quat = quat
#         return frame
    
#     def attach_free(self, child, pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0]) -> mjcf.Element:
#         """
#         Attaches a child element to the MJCF model with a free joint.
        
#         Args:
#             child: The child element to attach.
#             pos: Position of the child element.
#             quat: Orientation of the child element.
            
#         Returns:
#             The frame of the attached child element.
#         """
#         frame = self.attach(child)
#         frame.add('freejoint')
#         frame.pos = pos
#         frame.quat = quat
#         return frame
    
#     @property
#     def mjcf_model(self) -> mjcf.RootElement:
#         """
#         Returns the MJCF model for the StandardArena object.
        
#         Returns:
#             The MJCF model.
#         """
#         return self._mjcf_model
"""
StandardArena with walls and domain randomization support
File: arena/arena.py
"""
from dm_control import mjcf
import numpy as np


class StandardArena(object):
    def __init__(self, randomize=True) -> None:
        """
        Initializes the StandardArena with floor, walls, and lights.
        
        Args:
            randomize: If True, randomize floor and wall colors
        """
        self._mjcf_model = mjcf.RootElement()
        self._mjcf_model.option.timestep = 0.002
        self._mjcf_model.option.flag.warmstart = "enable"
        self._mjcf_model.option.integrator = "implicitfast"
        self._mjcf_model.option.solver = "CG"
        
        # Randomize colors if enabled
        if randomize:
            # Floor màu đơn random
            r = np.random.uniform(0.2, 0.8)
            g = np.random.uniform(0.2, 0.8)
            b = np.random.uniform(0.2, 0.8)
            floor_rgb = [r, g, b, 1.0]
            wall_rgba = self._random_wall_color()
        else:
            floor_rgb = [0.5, 0.5, 0.5, 1.0]
            wall_rgba = [0.8, 0.8, 0.8, 1.0]
        
        # Floor material (màu đơn, không đổ bóng)
        floor_mat = self._mjcf_model.asset.add(
            "material",
            name="floor_mat",
            rgba=floor_rgb,
            reflectance=0.0,
            specular=0.0
        )
        
        # Floor plane
        self._mjcf_model.worldbody.add(
            "geom",
            type="plane",
            size=[3, 3, 0.1],
            material=floor_mat
        )
        
        # Add 4 walls to create indoor environment
        wall_height = 0.5
        wall_thickness = 0.1
        room_size = 3.0
        
        # Wall material
        wall_mat = self._mjcf_model.asset.add(
            "material",
            name="wall_material",
            rgba=wall_rgba,
            reflectance=0.1
        )
        
        # North wall (y = room_size)
        self._mjcf_model.worldbody.add(
            "geom",
            type="box",
            size=[room_size, wall_thickness, wall_height],
            pos=[0, room_size, wall_height],
            material=wall_mat,
            contype="1",
            conaffinity="1"
        )
        
        # South wall (y = -room_size)
        self._mjcf_model.worldbody.add(
            "geom",
            type="box",
            size=[room_size, wall_thickness, wall_height],
            pos=[0, -room_size, wall_height],
            material=wall_mat,
            contype="1",
            conaffinity="1"
        )
        
        # East wall (x = room_size)
        self._mjcf_model.worldbody.add(
            "geom",
            type="box",
            size=[wall_thickness, room_size, wall_height],
            pos=[room_size, 0, wall_height],
            material=wall_mat,
            contype="1",
            conaffinity="1"
        )
        
        # West wall (x = -room_size)
        self._mjcf_model.worldbody.add(
            "geom",
            type="box",
            size=[wall_thickness, room_size, wall_height],
            pos=[-room_size, 0, wall_height],
            material=wall_mat,
            contype="1",
            conaffinity="1"
        )
        
        # Lights
        for x in [-2, 2]:
            self._mjcf_model.worldbody.add(
                "light", 
                pos=[x, -1, 3], 
                dir=[-x, 1, -2]
            )
    
    def _random_wall_color(self):
        """Generate random wall color"""
        r = np.random.uniform(0.6, 0.9)
        g = np.random.uniform(0.6, 0.9)
        b = np.random.uniform(0.6, 0.9)
        return [r, g, b, 1.0]
    
    def attach(self, child, pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0]) -> mjcf.Element:
        frame = self._mjcf_model.attach(child)
        frame.pos = pos
        frame.quat = quat
        return frame
    
    def attach_free(self, child, pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0]) -> mjcf.Element:
        frame = self.attach(child)
        frame.add('freejoint')
        frame.pos = pos
        frame.quat = quat
        return frame
    
    @property
    def mjcf_model(self) -> mjcf.RootElement:
        return self._mjcf_model

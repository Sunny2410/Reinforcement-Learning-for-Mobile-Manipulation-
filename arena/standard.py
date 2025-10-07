from dm_control import mjcf

class StandardArena(object):
    def __init__(self) -> None:
        self._mjcf_model = mjcf.RootElement()
        self._mjcf_model.option.timestep = 0.002
        self._mjcf_model.option.flag.warmstart = "enable"

        # Floor
        chequered = self._mjcf_model.asset.add(
            "texture", type="2d", builtin="checker",
            width=300, height=300,
            rgb1=[0.2, 0.3, 0.4], rgb2=[0.3, 0.4, 0.5]
        )
        grid = self._mjcf_model.asset.add(
            "material", name="grid", texture=chequered,
            texrepeat=[5, 5], reflectance=0.2
        )
        self._mjcf_model.worldbody.add(
            "geom", type="plane", size=[2, 2, 0.1], material=grid
        )

        # Walls
        wall_mat = self._mjcf_model.asset.add(
            "material", name="wall_material",
            rgba=[0.9, 0.9, 0.9, 1.0], reflectance=0.2
        )
        wall_height = 1.5
        wall_thickness = 0.1
        room_size = 2.0
        z_offset = 0.001

        def wall(x, y, sx, sy):
            self._mjcf_model.worldbody.add(
                "geom",
                type="box",
                size=[sx, sy, wall_height/2],
                pos=[x, y, wall_height/2 + z_offset],
                material=wall_mat,
                contype="1",
                conaffinity="1",
            )

        wall(0,  room_size + wall_thickness, room_size, wall_thickness)
        wall(0, -room_size - wall_thickness, room_size, wall_thickness)
        wall( room_size + wall_thickness, 0, wall_thickness, room_size)
        wall(-room_size - wall_thickness, 0, wall_thickness, room_size)

        # Lights
        for x in [-2, 2]:
            self._mjcf_model.worldbody.add(
                "light", pos=[x, -1, 3], dir=[-x, 1, -2]
            )
        self._mjcf_model.worldbody.add(
            "light", pos=[0, 0, 4], dir=[0, 0, -1]
        )

        # Optional: tweak shadow map
        self._mjcf_model.visual.map.shadowclip = 0.01

    def attach(self, child, pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0]) -> mjcf.Element:
        """Attach a child element to the MJCF model."""
        frame = self._mjcf_model.attach(child)
        frame.pos = pos
        frame.quat = quat
        return frame

    def attach_free(self, child, pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0]) -> mjcf.Element:
        """Attach a child element to the MJCF model with a free joint."""
        frame = self.attach(child)
        frame.add('freejoint')
        frame.pos = pos
        frame.quat = quat
        return frame

    @property
    def mjcf_model(self) -> mjcf.RootElement:
        """Return the MJCF model."""
        return self._mjcf_model

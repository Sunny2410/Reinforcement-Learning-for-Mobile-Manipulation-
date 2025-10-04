"""
Primitive class with domain randomization, soft contact, composite, and flexcomp support
File: props/primitive.py
"""
from dm_control import mjcf
import numpy as np


class Primitive:
    """
    Base class for primitive objects in MuJoCo.
    Supports:
      - Rigid objects (box, sphere, cylinder)
      - Soft contacts (solimp / solref)
      - Composite objects (cloth, deformable mesh)
      - Flexcomp (soft/deformable block)
      - Domain randomization: size, rgba, mass, friction, stiffness/damping
      - Centered composite spawn
    """

    def __init__(
        self,
        type="box",               
        size=None,
        rgba=None,
        mass=None,
        friction=None,
        soft=False,               
        solimp=None,
        solref=None,
        composite_config=None,    
        mesh=None,                
        **kwargs
    ):
        self._mjcf_model = mjcf.RootElement()
        self._geom = None
        self._all_geoms = []

        # Default geom_kwargs
        geom_kwargs = kwargs.copy()
        if size is not None:
            geom_kwargs['size'] = size
        else:
            if 'size' not in geom_kwargs:
                geom_kwargs['size'] = [0.02, 0.02, 0.02]
        if rgba is not None:
            geom_kwargs['rgba'] = rgba
        else:
            if 'rgba' not in geom_kwargs:
                geom_kwargs['rgba'] = [1, 0, 0, 1]
        if friction is not None:
            geom_kwargs['friction'] = friction
        if soft:
            geom_kwargs['solimp'] = solimp if solimp is not None else [0.9, 0.95, 0.001]
            geom_kwargs['solref'] = solref if solref is not None else [0.01, 1.0]

        # ----- Composite (cloth / deformable) -----
        if type == "composite":
            if composite_config is None:
                raise ValueError("composite_config required for type='composite'")
            comp = composite_config
            count = comp.get("count", [1,1,1])
            spacing = comp.get("spacing", 0.01)
            geom_type = comp.get("geom_type", "capsule")
            geom_size = comp.get("geom_size", [0.01])
            rgba = comp.get("geom_rgba", [1,0,0,1])
            mass = comp.get("mass", 0.001)

            # Offset để composite centered tại origin
            center_offset = [
                -0.5 * (count[0]-1) * spacing,
                -0.5 * (count[1]-1) * spacing,
                -0.5 * (count[2]-1) * spacing
            ]

            for i in range(count[0]):
                for j in range(count[1]):
                    for k in range(count[2]):
                        pos = [
                            i * spacing + center_offset[0],
                            j * spacing + center_offset[1],
                            k * spacing + center_offset[2]
                        ]
                        g = self._mjcf_model.worldbody.add(
                            "geom",
                            type=geom_type,
                            size=geom_size,
                            rgba=rgba,
                            mass=mass,
                            pos=pos
                        )
                        self._all_geoms.append(g)

        # ----- FlexComp (soft/deformable block) -----
        elif type == "flexcomp":
            if composite_config is None:
                raise ValueError("composite_config required for type='flexcomp'")
            comp = composite_config
            count = comp.get("count", [2,2,2])
            spacing = comp.get("spacing", [0.1,0.1,0.1])
            pos0 = comp.get("pos", [0,0,0])
            radius = comp.get("radius", 0.01)
            rgba = comp.get("rgba", [0.68,0.53,0.38,1])
            mass = max(comp.get("mass", 0.25), 1e-5)  # đảm bảo mass ≥ mjMINVAL
            condim = comp.get("condim", 3)
            solref = comp.get("solref", [0.01,1])
            solimp = comp.get("solimp", [0.95,0.99,0.0001])
            damping = comp.get("damping", 1)
            young = comp.get("young", 6e6)
            poisson = comp.get("poisson", 0.2)
            thickness = comp.get("thickness", 8e-3)
            elastic2d = comp.get("elastic2d", "bend")
            elastic_damp = comp.get("elastic_damp", 1e-5)
            dim = comp.get("dim", 2)

            # Tạo flexcomp
            flex = self._mjcf_model.worldbody.add(
                "flexcomp",
                type="box",
                count="{} {} {}".format(*count),
                spacing="{} {} {}".format(*spacing),
                pos="{} {} {}".format(*pos0),
                radius=str(radius),
                rgba="{} {} {} {}".format(*rgba),
                mass=str(mass),
                dim=str(dim)
            )
            flex.add(
                "contact",
                condim=str(condim),
                solref="{} {}".format(*solref),
                solimp="{} {} {}".format(*solimp),
                selfcollide="none"
            )
            flex.add(
                "edge",
                equality="true",
                damping=str(damping)
            )
            flex.add(
                "elasticity",
                young=str(young),
                poisson=str(poisson),
                thickness=str(thickness),
                elastic2d=elastic2d,
                damping=str(elastic_damp)
            )


        # ----- Mesh -----
        elif type == "mesh":
            if mesh is None:
                raise ValueError("mesh filename required for type='mesh'")
            self._geom = self._mjcf_model.worldbody.add(
                "geom",
                type="mesh",
                mesh=mesh,
                mass=mass if mass is not None else 0.1,
                **geom_kwargs
            )
            self._all_geoms.append(self._geom)

        # ----- Standard rigid geom -----
        else:
            self._geom = self._mjcf_model.worldbody.add(
                "geom",
                type=type,
                mass=mass if mass is not None else 0.1,
                **geom_kwargs
            )
            self._all_geoms.append(self._geom)

        # Add inertia if mass and size are given (for rigid box/sphere/cylinder)
        if type in ["box", "sphere", "cylinder"] and mass is not None:
            s = geom_kwargs['size']
            if isinstance(s, (list, tuple, np.ndarray)) and len(s) >= 3:
                ixx = mass * (s[1]**2 + s[2]**2) / 12
                iyy = mass * (s[0]**2 + s[2]**2) / 12
                izz = mass * (s[0]**2 + s[1]**2) / 12
                diag_inertia = [ixx, iyy, izz]
            else:
                i = 0.4 * mass * s[0]**2
                diag_inertia = [i, i, i]
            self._mjcf_model.worldbody.add(
                "inertial",
                pos=[0, 0, 0],
                mass=mass,
                diaginertia=diag_inertia
            )

    @property
    def geom(self):
        """Return first geom element (main)"""
        return self._geom

    @property
    def mjcf_model(self):
        """Return MJCF model"""
        return self._mjcf_model

    @property
    def all_geoms(self):
        """Return list of all geoms (for composite/flexcomp objects)"""
        return self._all_geoms

    def update_properties(self, rgba=None, friction=None, solimp=None, solref=None):
        """
        Update properties for domain randomization after creation.
        Applies to all geoms (composite, flexcomp, or single geom).
        """
        for g in self._all_geoms:
            if rgba is not None:
                g.rgba = rgba
            if friction is not None:
                g.friction = friction
            if solimp is not None:
                g.solimp = solimp
            if solref is not None:
                g.solref = solref

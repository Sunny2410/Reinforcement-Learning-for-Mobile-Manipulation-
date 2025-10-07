"""
Primitive object with domain randomization support
File: props/primitive.py
FIXED: inertial must be added to body, not worldbody
"""
from dm_control import mjcf
import numpy as np


class Primitive(object):
    """
    A primitive object with domain randomization for box type.
    Automatically randomizes size, position, color, and mass.
    """

    def __init__(self, type="box", randomize=True, **kwargs):
        """
        Initialize the Primitive object with optional randomization.

        Args:
            type: Geometry type (default: "box")
            randomize: If True, randomize size, rgba, and mass
            **kwargs: Additional keyword arguments for configuring the primitive
        """
        self._mjcf_model = mjcf.RootElement()
        
        # Apply randomization if enabled
        if randomize:
            geom_kwargs = self._apply_randomization(kwargs)
        else:
            geom_kwargs = kwargs.copy()
            # Set defaults if not provided
            if 'size' not in geom_kwargs:
                geom_kwargs['size'] = [0.02, 0.02, 0.02]
            if 'rgba' not in geom_kwargs:
                geom_kwargs['rgba'] = [1, 0, 0, 1]
            if 'mass' not in geom_kwargs:
                geom_kwargs['mass'] = 0.03
        
        # Add collision properties
        geom_kwargs['contype'] = 2
        geom_kwargs['condim'] = 6
        geom_kwargs['priority'] = 1
        
        # Extract mass for inertial calculation
        mass = geom_kwargs.get('mass', 0.03)
        size = geom_kwargs.get('size', [0.02, 0.02, 0.02])
        
        # Create a body first (required for inertial)
        self._body = self._mjcf_model.worldbody.add('body')
        
        # Add geom to the body
        self._geom = self._body.add(
            "geom", 
            type=type,
            **geom_kwargs
        )
        
        # # Add inertial properties to the body (not worldbody!)
        # if type == "box" and len(size) >= 3:
        #     # Calculate inertia for box
        #     ixx = mass * (size[1]**2 + size[2]**2) / 12
        #     iyy = mass * (size[0]**2 + size[2]**2) / 12
        #     izz = mass * (size[0]**2 + size[1]**2) / 12
            
        #     self._body.add(
        #         "inertial",
        #         pos=[0, 0, 0],
        #         mass=mass,
        #         diaginertia=[ixx, iyy, izz]
        #     )

    def _apply_randomization(self, kwargs):
        """
        Apply domain randomization to object properties.

        Args:
            kwargs: Original keyword arguments

        Returns:
            Modified kwargs with randomized values
        """
        geom_kwargs = kwargs.copy()
        
        # Randomize size (between 0.02 and 0.05)
        if 'size' not in geom_kwargs:
            base_size = 0.02
            geom_kwargs['size'] = [np.random.uniform(0.02, 0.05)] * 3
        else:
            # Scale existing size slightly within range
            original_size = geom_kwargs['size']
            if isinstance(original_size, (list, tuple, np.ndarray)):
                geom_kwargs['size'] = [np.clip(s * np.random.uniform(0.8, 1.2), 0.02, 0.05) for s in original_size]
            else:
                geom_kwargs['size'] = [np.clip(original_size * np.random.uniform(0.8, 1.2), 0.02, 0.05)] * 3
        
        # Randomize RGBA (varied colors)
        if 'rgba' not in geom_kwargs:
            r = np.random.uniform(0.2, 1.0)
            g = np.random.uniform(0.2, 1.0)
            b = np.random.uniform(0.2, 1.0)
            geom_kwargs['rgba'] = [r, g, b, 1.0]
        
        # Randomize mass (between 0.01 and 0.05)
        if 'mass' not in geom_kwargs:
            geom_kwargs['mass'] = np.random.uniform(0.01, 0.05)
        else:
            # Clip existing mass to range
            base_mass = geom_kwargs['mass']
            geom_kwargs['mass'] = float(np.clip(base_mass * np.random.uniform(0.8, 1.2), 0.01, 0.05))
        
        # Randomize friction if not specified
        if 'friction' not in geom_kwargs:
            geom_kwargs['friction'] = [np.random.uniform(0.5, 1.5)] * 3
        
        return geom_kwargs

    @property
    def geom(self):
        """Returns the primitive's geom, e.g., to change color or friction."""
        return self._geom
    
    @property
    def body(self):
        """Returns the primitive's body element."""
        return self._body
    
    @property
    def mjcf_model(self):
        """Returns the primitive's mjcf model."""
        return self._mjcf_model
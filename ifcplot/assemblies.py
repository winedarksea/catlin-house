"""
Wall and roof assembly specifications for IFC models and detail drawings.

These dataclasses define common construction assemblies with dimensions
that can be used both for IFC model generation and matplotlib detail plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from .units import inch


@dataclass(frozen=True)
class WallLayerSpec:
    """Specification for a single wall layer."""
    name: str
    thickness_in: float
    material: str = "generic"
    r_value_per_inch: float = 0.0  # R-value per inch if insulating
    
    @property
    def thickness_m(self) -> float:
        return inch(self.thickness_in)


@dataclass
class ExteriorWallAssembly:
    """
    Base class for exterior wall assemblies with standard layers.
    
    Layers are defined from interior to exterior. The coordinate system
    places x=0 at a reference point (typically exterior face of cladding
    or structural face) and negative x toward interior.
    """
    drywall_in: float = 0.625
    stud_depth_in: float = 3.5
    sheathing_in: float = 0.625
    membrane_in: float = 0.0
    polyiso_in: float = 0.0
    eps_in: float = 0.0
    furring_in: float = 0.0
    cladding_in: float = 0.5
    
    # Reference point: exterior face of cladding at x = x_clad_exterior
    x_clad_exterior: float = 1.0
    
    def total_thickness_in(self) -> float:
        return (
            self.drywall_in +
            self.stud_depth_in +
            self.sheathing_in +
            self.membrane_in +
            self.polyiso_in +
            self.eps_in +
            self.furring_in +
            self.cladding_in
        )
    
    def layer_coords(self) -> Dict[str, Tuple[float, float]]:
        """
        Returns dict of layer name -> (x_interior, x_exterior) in inches.
        Coordinates are relative to exterior face of cladding at x_clad_exterior.
        """
        x = self.x_clad_exterior
        coords = {}
        
        x0 = x - self.cladding_in
        coords["cladding"] = (x0, x)
        x = x0
        
        x0 = x - self.furring_in
        coords["furring"] = (x0, x)
        x = x0
        
        x0 = x - self.eps_in
        coords["eps"] = (x0, x)
        x = x0
        
        x0 = x - self.polyiso_in
        coords["polyiso"] = (x0, x)
        x = x0
        
        x0 = x - self.membrane_in
        coords["membrane"] = (x0, x)
        x = x0
        
        x0 = x - self.sheathing_in
        coords["sheathing"] = (x0, x)
        x = x0
        
        x0 = x - self.stud_depth_in
        coords["stud"] = (x0, x)
        x = x0
        
        x0 = x - self.drywall_in
        coords["drywall"] = (x0, x)
        
        return coords
    
    def to_ifc_params(self) -> dict:
        """Export assembly as a dict suitable for IFC property set storage."""
        return {
            "drywall_in": self.drywall_in,
            "stud_depth_in": self.stud_depth_in,
            "sheathing_in": self.sheathing_in,
            "membrane_in": self.membrane_in,
            "polyiso_in": self.polyiso_in,
            "eps_in": self.eps_in,
            "furring_in": self.furring_in,
            "cladding_in": self.cladding_in,
        }


@dataclass
class RoofAssembly:
    """
    Roof assembly specification for warm roof with exterior insulation.
    
    Layers are defined from joist centerline outward (positive offset).
    """
    joist_depth_in: float = 11.875  # 11-7/8" I-joist
    sheathing_in: float = 0.75  # 3/4" Struct 1
    membrane_in: float = 0.25  # Liquid membrane
    polyiso_in: float = 2.0
    eps_in: float = 4.0
    roofing_membrane_in: float = 0.25
    furring_in: float = 0.75
    roofing_in: float = 0.5  # Standing seam
    
    pitch_rise_over_run: float = 4.0 / 12.0
    overhang_in: float = 16.0
    
    def total_buildup_in(self) -> float:
        """Total thickness above joist top flange."""
        return (
            self.sheathing_in +
            self.membrane_in +
            self.polyiso_in +
            self.eps_in +
            self.roofing_membrane_in +
            self.furring_in +
            self.roofing_in
        )
    
    def layer_offsets(self, from_joist_center: bool = True) -> Dict[str, Tuple[float, float]]:
        """
        Returns dict of layer name -> (bottom_offset, top_offset) from joist centerline.
        
        If from_joist_center is True, offsets are from joist centerline (mid-depth).
        If False, offsets are from top of joist.
        """
        base = self.joist_depth_in / 2 if from_joist_center else 0.0
        offsets = {}
        
        bot = base
        top = bot + self.sheathing_in
        offsets["sheathing"] = (bot, top)
        
        bot = top
        top = bot + self.membrane_in
        offsets["membrane"] = (bot, top)
        
        bot = top
        top = bot + self.polyiso_in
        offsets["polyiso"] = (bot, top)
        
        bot = top
        top = bot + self.eps_in
        offsets["eps"] = (bot, top)
        
        bot = top
        top = bot + self.roofing_membrane_in
        offsets["roofing_membrane"] = (bot, top)
        
        bot = top
        top = bot + self.furring_in
        offsets["furring"] = (bot, top)
        
        bot = top
        top = bot + self.roofing_in
        offsets["roofing"] = (bot, top)
        
        return offsets
    
    def to_ifc_params(self) -> dict:
        """Export assembly as a dict suitable for IFC property set storage."""
        return {
            "ijoist_depth_in": self.joist_depth_in,
            "sheathing_in": self.sheathing_in,
            "membrane_in": self.membrane_in,
            "polyiso_in": self.polyiso_in,
            "eps_in": self.eps_in,
            "roofing_membrane_in": self.roofing_membrane_in,
            "furring_in": self.furring_in,
            "metal_roof_in": self.roofing_in,
            "pitch_rise_over_run": self.pitch_rise_over_run,
            "overhang_in": self.overhang_in,
        }


@dataclass
class ICFFoundationAssembly:
    """ICF (Insulated Concrete Form) foundation wall assembly."""
    core_in: float = 8.0  # Concrete core
    eps_in: float = 2.5  # EPS foam on each side
    coating_in: float = 0.5  # Protective coating above grade
    
    above_grade_in: float = 22.0  # Height exposed above grade
    frost_depth_in: float = 42.0  # Depth below grade to footing
    
    @property
    def total_width_in(self) -> float:
        return self.core_in + 2 * self.eps_in
    
    def to_ifc_params(self) -> dict:
        return {
            "core_in": self.core_in,
            "eps_in": self.eps_in,
            "coating_in": self.coating_in,
            "total_in": self.total_width_in,
            "above_grade_in": self.above_grade_in,
            "frost_depth_in": self.frost_depth_in,
        }


# Pre-defined assemblies for common configurations
HOUSE_WALL_2X4_WITH_CI = ExteriorWallAssembly(
    drywall_in=0.625,
    stud_depth_in=3.5,
    sheathing_in=0.625,
    membrane_in=0.0,
    polyiso_in=2.0,
    eps_in=2.0,
    furring_in=0.5,
    cladding_in=0.5,
)

HOUSE_WALL_2X6_WITH_ZIPR = ExteriorWallAssembly(
    drywall_in=0.625,
    stud_depth_in=5.5,
    sheathing_in=1.5,  # Zip-R
    membrane_in=0.0,
    polyiso_in=0.0,
    eps_in=0.0,
    furring_in=0.375,  # Rainscreen
    cladding_in=0.5,
)

GARAGE_WALL = ExteriorWallAssembly(
    drywall_in=0.625,
    stud_depth_in=5.5,
    sheathing_in=1.5,  # Zip-R
    membrane_in=0.0,
    polyiso_in=0.0,
    eps_in=0.0,
    furring_in=0.375,
    cladding_in=0.5,
)

HOUSE_ROOF = RoofAssembly(
    joist_depth_in=11.875,
    sheathing_in=0.75,
    membrane_in=0.25,
    polyiso_in=2.0,
    eps_in=4.0,
    roofing_membrane_in=0.25,
    furring_in=0.75,
    roofing_in=0.5,
    pitch_rise_over_run=4.0 / 12.0,
    overhang_in=16.0,
)

GARAGE_ICF = ICFFoundationAssembly(
    core_in=8.0,
    eps_in=2.5,
    above_grade_in=22.0,
    frost_depth_in=42.0,
)

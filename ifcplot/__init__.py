"""
ifcPlot - IFC-first house modeling utilities.

A Python library for creating IFC (Industry Foundation Classes) building models
with a focus on residential construction. Designed to be LLM coding agent friendly
with modular, testable components.

Key features:
- IFC model generation using ifcopenshell
- Wall assembly specifications for 2D detail generation
- Unit conversion utilities for imperial/metric
- Integration with matplotlib for detail drawings
"""

from .units import M_PER_FT, M_PER_IN, ft, ft_in, inch, m_to_ft, m_to_in
from .ifc_utils import (
    IfcContexts,
    add_building,
    add_prism_from_profile,
    add_slab,
    add_storey,
    add_trade_groups,
    add_wall_between_points,
    assign_to_group,
    ensure_pset,
    init_ifc_project,
    set_pset_json,
    translation_matrix,
)
from .assemblies import (
    ExteriorWallAssembly,
    ICFFoundationAssembly,
    RoofAssembly,
    WallLayerSpec,
    HOUSE_WALL_2X4_WITH_CI,
    HOUSE_WALL_2X6_WITH_ZIPR,
    GARAGE_WALL,
    HOUSE_ROOF,
    GARAGE_ICF,
)

__version__ = "0.1.0"
__all__ = [
    # Units
    "M_PER_FT",
    "M_PER_IN",
    "ft",
    "ft_in",
    "inch",
    "m_to_ft",
    "m_to_in",
    # IFC utilities
    "IfcContexts",
    "add_building",
    "add_prism_from_profile",
    "add_slab",
    "add_storey",
    "add_trade_groups",
    "add_wall_between_points",
    "assign_to_group",
    "ensure_pset",
    "init_ifc_project",
    "set_pset_json",
    "translation_matrix",
    # Assembly specifications
    "ExteriorWallAssembly",
    "ICFFoundationAssembly",
    "RoofAssembly",
    "WallLayerSpec",
    "HOUSE_WALL_2X4_WITH_CI",
    "HOUSE_WALL_2X6_WITH_ZIPR",
    "GARAGE_WALL",
    "HOUSE_ROOF",
    "GARAGE_ICF",
]


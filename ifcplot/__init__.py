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
    add_prism_from_profile_with_voids,
    add_rect_member_between_points,
    add_slab,
    add_storey,
    add_trade_groups,
    add_wall_between_points,
    assign_to_group,
    ensure_pset,
    init_ifc_project,
    placement_matrix,
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
from .detail_utils import (
    COLORS,
    HATCHES,
    MATERIAL_COLORS,
    LayerMap,
    ExteriorWoodWallAssembly,
    BasementConcreteWallAssembly,
    load_markdown_notes,
    _rect,
    _poly,
    _leader,
    _dim_h,
    _dim_v,
    _offset_segment,
    _quad_from_segment,
    _round_rect,
    _pipe,
    _dotted_pipe,
    _rebar_L,
    _rebar_h,
    _screw,
    _batt_insulation,
    _stud_pattern,
    _path_from_steps,
    _thick_polyline,
    _flashing,
    _lumber,
    _french_drain,
    _slab_assembly,
    _translate_layers,
    _wrap_notes,
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
    "add_prism_from_profile_with_voids",
    "add_rect_member_between_points",
    "add_slab",
    "add_storey",
    "add_trade_groups",
    "add_wall_between_points",
    "assign_to_group",
    "ensure_pset",
    "init_ifc_project",
    "placement_matrix",
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
    # Detail utilities
    "COLORS",
    "HATCHES",
    "MATERIAL_COLORS",
    "LayerMap",
    "ExteriorWoodWallAssembly",
    "BasementConcreteWallAssembly",
    "load_markdown_notes",
]

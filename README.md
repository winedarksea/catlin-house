# ifcPlot

IFC-first house modeling utilities for residential construction.

A Python library for creating IFC (Industry Foundation Classes) building models with a focus on North American residential construction. Designed to be LLM coding agent friendly with modular, testable components.

## Features

- **IFC Model Generation**: Build complete IFC models using ifcopenshell with support for walls, slabs, roofs, and building elements
- **Wall Assembly Specifications**: Define and reuse wall assemblies (ICF, wood frame with continuous insulation) with layer-by-layer specifications
- **2D Detail Generation**: Generate matplotlib-based construction detail drawings that read parameters directly from IFC property sets
- **Unit Conversion**: Seamless imperial/metric conversion utilities for North American building standards
- **Trade Groups**: Organize elements by trade (concrete, framing, HVAC, plumbing, drywall, cladding)

## Installation

```bash
# Install from source (development mode)
pip install -e .

# Or install dependencies directly
pip install numpy matplotlib ifcopenshell
```

## Project Structure

```
catlin-house/
├── ifcplot/                    # Reusable library package
│   ├── __init__.py             # Package exports
│   ├── assemblies.py           # Wall and roof assembly specifications
│   ├── ifc_utils.py            # IFC creation utilities
│   ├── units.py                # Unit conversion helpers
│   └── catlin_house.py         # House-specific definitions
│
├── catlin_house/               # Specific implementation
│   ├── build_catlin_house_ifc.py    # Build the complete IFC model
│   ├── garage_wall_detail_side_ifc.py  # Garage wall section detail
│   ├── roof_wall_eave_detail_ifc.py    # Roof-wall eave detail
│   └── basement_floor_plan_ifc.py      # Basement floor plan
│
├── detail_utils.py             # Shared matplotlib helpers (colors, hatches)
├── house_description.md        # Full project specification
├── pyproject.toml              # Package configuration
└── requirements.txt            # Dependencies
```

## Quick Start

### 1. Build the IFC Model

```bash
cd catlin-house
PYTHONPATH="." python catlin_house/build_catlin_house_ifc.py
```

This creates `out/catlin_house.ifc` with the complete model including parameter property sets.

### 2. Generate Detail Drawings

```bash
# Garage wall section
PYTHONPATH="." python catlin_house/garage_wall_detail_side_ifc.py

# Roof-wall eave detail
PYTHONPATH="." python catlin_house/roof_wall_eave_detail_ifc.py

# Basement floor plan
PYTHONPATH="." python catlin_house/basement_floor_plan_ifc.py
```

Each script reads parameters from the IFC file and generates a PNG detail drawing.

## Usage

### Define Wall Assemblies

```python
from ifcplot import ExteriorWallAssembly, inch

# Define a 2x6 wall with continuous insulation
wall = ExteriorWallAssembly(
    drywall_in=0.625,
    stud_depth_in=5.5,      # 2x6 studs
    sheathing_in=1.5,        # Zip-R sheathing
    polyiso_in=2.0,          # Continuous insulation
    furring_in=0.375,        # Rainscreen gap
    cladding_in=0.5,         # Standing seam
)

# Get layer coordinates for detail drawing
coords = wall.layer_coords()
print(f"Stud cavity: {coords['stud']}")  # (interior_x, exterior_x) in inches
```

### Create IFC Elements

```python
from ifcplot import init_ifc_project, add_building, add_storey, add_wall_between_points, ft

# Initialize project
ifc, contexts = init_ifc_project("My House")
building = add_building(ifc, contexts, "House")
storey = add_storey(ifc, building, "First Floor", 0.0)

# Add a wall
wall = add_wall_between_points(
    ifc, contexts, storey,
    name="North Wall",
    start=(0, ft(36)),
    end=(ft(36), ft(36)),
    height=ft(9),
    thickness=inch(8.75),
)
```

### Use Pre-defined Assemblies

```python
from ifcplot import HOUSE_ROOF, GARAGE_ICF

# Get roof layer offsets for detail drawing
offsets = HOUSE_ROOF.layer_offsets()
print(f"EPS insulation: {offsets['eps']}")  # (bottom, top) from joist center

# ICF foundation parameters
print(f"ICF total width: {GARAGE_ICF.total_width_in} inches")
```

## IFC-First Workflow

The key principle is that **IFC is the single source of truth**:

1. **Model parameters** are stored in IFC property sets as JSON
2. **Detail drawings** read parameters from the IFC file
3. **Changes propagate automatically** - update the IFC model once, regenerate all details

Example property set stored in IFC:
```json
{
  "ijoist_depth_in": 11.875,
  "polyiso_in": 2.0,
  "eps_in": 4.0,
  "pitch_rise_over_run": 0.333,
  "overhang_in": 16.0
}
```

## Design Goals

- **Single source of truth**: Update dimensions in one place, all details update
- **LLM-friendly**: Small, modular files that fit in context windows
- **North American standards**: Imperial units (2x4s, 8' panels) with metric support
- **Colored details**: More color than traditional architectural details for clarity
- **Trade organization**: IFC groups for concrete, framing, HVAC, plumbing, etc.

## Dependencies

- **numpy** >= 1.20.0 - Geometric calculations
- **matplotlib** >= 3.5.0 - 2D detail drawings
- **ifcopenshell** >= 0.7.0 - IFC file handling

Optional:
- **adjustText** - Automatic label positioning
- **Bonsai BIM** - 3D visualization in Blender

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

This project is designed to be extended for your own house projects. Fork the repository, modify `catlin_house/` for your specific design, and use `ifcplot/` utilities for common operations.

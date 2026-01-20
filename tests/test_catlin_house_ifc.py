from __future__ import annotations

import json

import ifcopenshell
import ifcopenshell.util.element
import ifcopenshell.util.placement
import pytest

from ifcplot.catlin_house import CatlinHouseSpec, build_catlin_house_ifc
from ifcplot.units import ft, inch


def test_build_catlin_house_ifc_has_centerline_wall_and_joists(tmp_path) -> None:
    out_path = tmp_path / "catlin_house.ifc"
    build_catlin_house_ifc(out_path=out_path)

    f = ifcopenshell.open(str(out_path))

    centerline_walls = [w for w in f.by_type("IfcWall") if w.Name and "House Centerline Wall" in w.Name]
    assert len(centerline_walls) == 3

    spec = CatlinHouseSpec()
    house_size_in = spec.house_size_ft * 12.0
    expected_positions = int(round(house_size_in / spec.framing_spacing_in)) + 1
    expected_pair_count = expected_positions * 2

    second_floor_joists = [m for m in f.by_type("IfcBeam") if m.Name and "House Second Floor Joist" in m.Name]
    assert len(second_floor_joists) == expected_pair_count

    attic_floor_joists = [m for m in f.by_type("IfcBeam") if m.Name and "House Attic Floor Joist" in m.Name]
    assert len(attic_floor_joists) == expected_pair_count

    roof_joists = [m for m in f.by_type("IfcBeam") if m.Name and m.Name.startswith("House Roof Joist")]
    assert len(roof_joists) == expected_pair_count

    # Spot-check roof joist slope (4:12) via placement Z axis.
    sample = next(m for m in roof_joists if m.Name == "House Roof Joist E-01")
    m = ifcopenshell.util.placement.get_local_placement(sample.ObjectPlacement)
    z_axis = m[:3, 2]
    slope = abs(float(z_axis[2]) / float(z_axis[0]))  # rise/run magnitude in XZ plane
    assert slope == pytest.approx(spec.roof_pitch_rise_over_run, rel=1e-6)


def test_roof_has_detail_pset_json(tmp_path) -> None:
    out_path = tmp_path / "catlin_house.ifc"
    build_catlin_house_ifc(out_path=out_path)

    f = ifcopenshell.open(str(out_path))
    roof = next(r for r in f.by_type("IfcRoof") if r.Name and r.Name.startswith("House Roof"))
    psets = ifcopenshell.util.element.get_psets(roof)
    raw = psets.get("Pset_ifcPlot_DetailParams", {}).get("ParamsJSON")
    assert raw

    params = json.loads(raw)
    assert "roof" in params
    assert "wall" in params
    assert params["roof"]["pitch_rise_over_run"] == CatlinHouseSpec().roof_pitch_rise_over_run


def test_house_has_basement_construction_pset_json(tmp_path) -> None:
    out_path = tmp_path / "catlin_house.ifc"
    build_catlin_house_ifc(out_path=out_path)

    f = ifcopenshell.open(str(out_path))
    house = next(b for b in f.by_type("IfcBuilding") if b.Name == "House")
    psets = ifcopenshell.util.element.get_psets(house)
    raw = psets.get("Pset_ifcPlot_BasementConstruction", {}).get("ParamsJSON")
    assert raw

    params = json.loads(raw)
    assert {"foundation", "slab"} <= set(params)
    spec = CatlinHouseSpec()
    assert params["foundation"]["wall_thickness_in"] == spec.basement_wall_thickness_in
    assert params["slab"]["slab_thickness_in"] == spec.basement_slab_thickness_in


def test_house_has_basement_to_framed_wall_detail_pset_json(tmp_path) -> None:
    out_path = tmp_path / "catlin_house.ifc"
    build_catlin_house_ifc(out_path=out_path)

    f = ifcopenshell.open(str(out_path))
    house = next(b for b in f.by_type("IfcBuilding") if b.Name == "House")
    psets = ifcopenshell.util.element.get_psets(house)
    raw = psets.get("Pset_ifcPlot_BasementToFramedWallDetail", {}).get("ParamsJSON")
    assert raw

    params = json.loads(raw)
    assert {"wall", "basement_exterior", "sill", "detail_view"} <= set(params)
    assert {"stud_depth_in", "sheathing_in", "polyiso_in", "eps_in"} <= set(params["wall"])
    assert params["basement_exterior"]["xps_layers"] >= 1


def test_house_has_sauna_shower_detail_pset_json(tmp_path) -> None:
    out_path = tmp_path / "catlin_house.ifc"
    build_catlin_house_ifc(out_path=out_path)

    f = ifcopenshell.open(str(out_path))
    house = next(b for b in f.by_type("IfcBuilding") if b.Name == "House")
    psets = ifcopenshell.util.element.get_psets(house)
    raw = psets.get("Pset_ifcPlot_SaunaShowerDetail", {}).get("ParamsJSON")
    assert raw

    params = json.loads(raw)
    assert {"finish", "structure", "adjacent_wall", "benches", "heater", "shower"} <= set(params)
    assert params["structure"]["clear_height_in"] == CatlinHouseSpec().basement_clear_height_ft * 12.0


def test_house_elements_have_surface_styles(tmp_path) -> None:
    out_path = tmp_path / "catlin_house.ifc"
    build_catlin_house_ifc(out_path=out_path)

    f = ifcopenshell.open(str(out_path))

    # Ensure we export multiple named surface styles (not just the wood framing).
    style_names = {s.Name for s in f.by_type("IfcSurfaceStyle") if s.Name}
    assert {"Concrete", "Sheathing/OSB", "Polyiso", "EPS", "XPS", "Membrane", "Framing Wood", "Standing Seam Metal"} <= style_names

    # Spot-check that representative elements actually reference a surface style (Bonsai otherwise shows default gray).
    sheathing = next(w for w in f.by_type("IfcWall") if w.Name == "House Main Exterior Sheathing 1")
    item = sheathing.Representation.Representations[0].Items[0]
    assert item.StyledByItem

    xps = next(w for w in f.by_type("IfcWall") if w.Name and w.Name.startswith("House Basement Exterior XPS L1"))
    xps_item = xps.Representation.Representations[0].Items[0]
    assert xps_item.StyledByItem


def test_house_has_basement_exterior_xps_layers(tmp_path) -> None:
    out_path = tmp_path / "catlin_house.ifc"
    build_catlin_house_ifc(out_path=out_path)

    f = ifcopenshell.open(str(out_path))
    xps_walls = [w for w in f.by_type("IfcWall") if w.Name and w.Name.startswith("House Basement Exterior XPS")]
    # 4 perimeter segments, 2 layers each.
    assert len(xps_walls) == 8


def test_roof_and_gables_have_expected_absolute_placements(tmp_path) -> None:
    out_path = tmp_path / "catlin_house.ifc"
    build_catlin_house_ifc(out_path=out_path)

    f = ifcopenshell.open(str(out_path))
    garage_roof = next(r for r in f.by_type("IfcRoof") if r.Name == "Garage Roof (placeholder prism)")
    house_roof = next(r for r in f.by_type("IfcRoof") if r.Name == "House Roof Sheathing (W)")
    gable = next(w for w in f.by_type("IfcWall") if w.Name == "House Attic Gable South Stud Wall")

    m_garage = ifcopenshell.util.placement.get_local_placement(garage_roof.ObjectPlacement)
    m_house = ifcopenshell.util.placement.get_local_placement(house_roof.ObjectPlacement)
    m_gable = ifcopenshell.util.placement.get_local_placement(gable.ObjectPlacement)

    # Garage is offset 12' north of the house's north wall and aligned on the west wall.
    # Roof starts at the west overhang (16") and extrudes east-west after the 90° rotation.
    assert float(m_garage[0, 3]) == pytest.approx(-inch(CatlinHouseSpec().garage_overhang_in) * 1000.0, abs=1e-6)
    spec = CatlinHouseSpec()
    assert float(m_garage[1, 3]) == pytest.approx(ft(spec.house_size_ft + 12.0) * 1000.0, abs=1e-6)

    z_axis = m_garage[:3, 2]
    assert tuple(map(float, z_axis)) == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)
    # House roof + gables are placed at the attic elevation (18').
    assert float(m_house[2, 3]) == pytest.approx(ft(18.0) * 1000.0, abs=1e-6)
    assert float(m_gable[2, 3]) == pytest.approx(ft(18.0) * 1000.0, abs=1e-6)


def test_project_has_bom_pset_json(tmp_path) -> None:
    out_path = tmp_path / "catlin_house.ifc"
    build_catlin_house_ifc(out_path=out_path)

    f = ifcopenshell.open(str(out_path))
    project = f.by_type("IfcProject")[0]
    psets = ifcopenshell.util.element.get_psets(project)
    raw = psets.get("Pset_ifcPlot_BOM", {}).get("SummaryJSON")
    assert raw

    bom = json.loads(raw)
    assert bom["studs"]["total"] > 0
    assert bom["joists"]["total"] > 0


def test_joists_have_nonzero_swept_solid_section(tmp_path) -> None:
    out_path = tmp_path / "catlin_house.ifc"
    build_catlin_house_ifc(out_path=out_path)

    f = ifcopenshell.open(str(out_path))
    joist = next(e for e in f.by_type("IfcBeam") if e.Name == "House Second Floor Joist W-01")
    rep = joist.Representation.Representations[0]
    assert rep.RepresentationIdentifier == "Body"
    assert rep.RepresentationType == "SweptSolid"
    solid = rep.Items[0]
    assert solid.is_a("IfcExtrudedAreaSolid")
    profile = solid.SweptArea
    assert profile.is_a("IfcRectangleProfileDef")
    # Project length units are millimeters; section dims should be O(10..1000), not ~0.
    assert float(profile.XDim) > 1.0
    assert float(profile.YDim) > 1.0


def test_studs_decompose_stud_wall_containers(tmp_path) -> None:
    out_path = tmp_path / "catlin_house.ifc"
    build_catlin_house_ifc(out_path=out_path)

    f = ifcopenshell.open(str(out_path))
    wall = next(w for w in f.by_type("IfcWall") if w.Name == "House Main Stud Wall 1")
    assert wall.IsDecomposedBy
    rel = wall.IsDecomposedBy[0]
    parts = rel.RelatedObjects
    assert any(p.is_a("IfcMember") for p in parts)


def test_joists_decompose_floor_framing_assembly(tmp_path) -> None:
    out_path = tmp_path / "catlin_house.ifc"
    build_catlin_house_ifc(out_path=out_path)

    f = ifcopenshell.open(str(out_path))
    asm = next(a for a in f.by_type("IfcElementAssembly") if a.Name == "House Second Floor Framing")
    assert asm.IsDecomposedBy
    parts = asm.IsDecomposedBy[0].RelatedObjects
    assert any(p.is_a("IfcBeam") and p.Name and p.Name.startswith("House Second Floor Joist") for p in parts)

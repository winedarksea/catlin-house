from __future__ import annotations

import json

import ifcopenshell
import ifcopenshell.util.element
import ifcopenshell.util.placement
import pytest

from ifcplot.catlin_house import CatlinHouseSpec, build_catlin_house_ifc


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


def test_house_elements_have_surface_styles(tmp_path) -> None:
    out_path = tmp_path / "catlin_house.ifc"
    build_catlin_house_ifc(out_path=out_path)

    f = ifcopenshell.open(str(out_path))

    # Ensure we export multiple named surface styles (not just the wood framing).
    style_names = {s.Name for s in f.by_type("IfcSurfaceStyle") if s.Name}
    assert {"Concrete", "Sheathing/OSB", "Polyiso", "EPS", "Membrane", "Framing Wood", "Standing Seam Metal"} <= style_names

    # Spot-check that representative elements actually reference a surface style (Bonsai otherwise shows default gray).
    sheathing = next(w for w in f.by_type("IfcWall") if w.Name == "House Main Exterior Sheathing 1")
    item = sheathing.Representation.Representations[0].Items[0]
    assert item.StyledByItem

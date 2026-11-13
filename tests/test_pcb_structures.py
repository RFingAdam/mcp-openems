"""Regression tests for the PCB-structure calculators and script generators.

None of these need openEMS installed: they check the analytical numbers, the
geometry data, and the text of the generated simulation script.
"""
from __future__ import annotations

import math
import re

import pytest

from mcp_openems.server import PCBStructureDesigner

C0 = 299792458.0


@pytest.fixture
def designer():
    return PCBStructureDesigner()


def _via(designer, **overrides):
    kwargs = dict(frequency_ghz=5.0, drill_mm=0.3, pad_mm=0.6, dielectric_height_mm=1.6, substrate_er=4.2)
    kwargs.update(overrides)
    result = designer.create_via_transition(**kwargs)
    assert result["success"] is True, result
    return result


# ---------------------------------------------------------------------------
# Defect 8: microstrip delay was reported 1000x too large (ps/m, not ps/mm)
# ---------------------------------------------------------------------------

def test_microstrip_delay_is_ps_per_mm(designer):
    calc = designer.create_microstrip_trace(
        frequency_ghz=2.4, trace_width_mm=3.0, trace_length_mm=20,
        dielectric_height_mm=1.6, substrate_er=4.2,
    )["design"]["calculated"]

    er_eff = calc["effective_er"]
    expected = math.sqrt(er_eff) / C0 * 1e9  # s/m -> ps/mm
    assert 3.0 < er_eff < 3.4
    assert calc["delay_ps_per_mm"] == pytest.approx(expected, rel=1e-3)
    assert 5.5 < calc["delay_ps_per_mm"] < 6.5


# ---------------------------------------------------------------------------
# Defect 9: Z_via inconsistent with its own L and C; feed width hard-coded
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("drill,pad,h", [(0.3, 0.6, 1.6), (0.2, 0.45, 0.2), (0.25, 0.5, 0.8)])
def test_via_impedance_consistent_with_reported_l_and_c(designer, drill, pad, h):
    calc = _via(designer, drill_mm=drill, pad_mm=pad, dielectric_height_mm=h)["design"]["calculated"]

    l_h = calc["inductance_ph"] * 1e-12
    c_f = calc["capacitance_ff"] * 1e-15
    assert calc["z_via_ohms"] == pytest.approx(math.sqrt(l_h / c_f), rel=0.01)
    f_res = 1 / (2 * math.pi * math.sqrt(l_h * c_f)) / 1e9
    assert calc["resonance_ghz"] == pytest.approx(f_res, rel=0.01)


def test_via_feed_width_is_derived_from_target_impedance(designer):
    thin = _via(designer, dielectric_height_mm=0.4)["design"]
    thick = _via(designer, dielectric_height_mm=1.6)["design"]

    # A 50-ohm feed on a thicker stackup must be wider; the old code returned
    # max(drill, 0.2) mm regardless of stackup.
    assert thick["dimensions"]["feed_trace_width_mm"] > 2 * thin["dimensions"]["feed_trace_width_mm"]

    # Cross-check with the repo's own microstrip calculator at the feed height.
    for design in (thin, thick):
        dims = design["dimensions"]
        ms = designer.create_microstrip_trace(
            frequency_ghz=5.0, trace_width_mm=dims["feed_trace_width_mm"], trace_length_mm=5,
            dielectric_height_mm=dims["feed_dielectric_height_mm"], substrate_er=4.2,
        )["design"]["calculated"]
        assert ms["z0_ohms"] == pytest.approx(50.0, abs=0.5)
        assert design["calculated"]["feed_z0_ohms"] == pytest.approx(50.0, abs=0.5)


def test_via_feed_width_accepts_user_value_and_target(designer):
    user = _via(designer, feed_trace_width_mm=0.37)["design"]
    assert user["dimensions"]["feed_trace_width_mm"] == pytest.approx(0.37)

    z75 = _via(designer, feed_target_z0_ohms=75.0)["design"]
    z50 = _via(designer)["design"]
    assert z75["dimensions"]["feed_trace_width_mm"] < z50["dimensions"]["feed_trace_width_mm"]
    assert z75["calculated"]["feed_z0_ohms"] == pytest.approx(75.0, abs=0.75)


# ---------------------------------------------------------------------------
# Defect 10: shorted bottom pad / no antipad; S-params; geometry check
# ---------------------------------------------------------------------------

def _ground_planes(design):
    # Pre-fix geometry had no "role" key; its ground was named "ground_plane_bottom".
    return [
        p for p in design["geometry"]["primitives"]
        if p["type"] == "box" and (p.get("role") == "ground" or "ground" in p["name"])
    ]


@pytest.mark.parametrize("planes", [1, 2, 3])
def test_via_every_reference_plane_has_antipad_and_no_short(designer, planes):
    design = _via(designer, reference_planes=planes)["design"]
    grounds = _ground_planes(design)
    assert len(grounds) == planes

    drill_r = design["parameters"]["drill_mm"] / 2
    for plane in grounds:
        cutouts = plane.get("cutouts", [])
        assert len(cutouts) == 1
        assert cutouts[0]["center"] == [0, 0]
        assert cutouts[0]["radius"] > drill_r

    assert PCBStructureDesigner.check_signal_ground_clearance(design["geometry"]["primitives"]) == []

    script = designer.generate_openems_script(design["id"])
    assert script["success"] is True, script
    # One antipad clearance cut per reference plane in the generated script.
    assert len(re.findall(r"# Antipad clearance in ref_plane_\d+", script["script"])) == planes
    compile(script["script"], "<via_script>", "exec")


def test_via_signal_layers_do_not_coincide_with_ground(designer):
    """The old geometry put the bottom pad and feed inside the ground box."""
    design = _via(designer)["design"]
    signal_z = {0.0, design["parameters"]["dielectric_height_mm"]}
    for plane in _ground_planes(design):
        z0, z1 = sorted((plane["start"][2], plane["stop"][2]))
        for z in signal_z:
            assert not (z0 <= z <= z1), f"signal layer z={z} lies inside {plane['name']} [{z0}, {z1}]"


def test_generate_script_refuses_signal_touching_ground(designer):
    design = _via(designer)["design"]
    # Remove the antipad: the barrel now passes through solid copper.
    for plane in _ground_planes(design):
        plane["cutouts"] = []

    result = designer.generate_openems_script(design["id"])
    assert result["success"] is False
    assert "via_barrel" in result["error"]
    assert result["geometry_errors"]


def test_clearance_check_flags_pad_on_ground_layer():
    primitives = [
        {"type": "box", "name": "gnd", "role": "ground", "material": "PEC",
         "start": [-5, -5, -0.035], "stop": [5, 5, 0], "cutouts": []},
        {"type": "disc", "name": "pad_bottom", "role": "signal", "material": "PEC",
         "center": [0, 0, 0], "radius": 0.3},
    ]
    errors = PCBStructureDesigner.check_signal_ground_clearance(primitives)
    assert errors and "pad_bottom" in errors[0]

    # Same pad, but inside an antipad of larger radius: no short.
    primitives[0]["cutouts"] = [{"shape": "circle", "center": [0, 0], "radius": 0.5}]
    assert PCBStructureDesigner.check_signal_ground_clearance(primitives) == []


def test_via_script_sparams_from_incident_and_reflected_waves(designer):
    design = _via(designer)["design"]
    script = designer.generate_openems_script(design["id"])["script"]

    assert "s11 = port1.uf_ref / port1.uf_inc" in script
    assert "s21 = port2.uf_ref / port1.uf_inc" in script
    assert script.count("CalcPort(Sim_Path, f, ref_impedance=z_ref)") == 2
    assert "uf_tot" not in script
    # The old script labelled port-1 input impedance as the via impedance.
    assert "Via impedance (from S11)" not in script


def test_via_ports_span_reference_plane_to_signal(designer):
    design = _via(designer, reference_planes=2)["design"]
    h = design["parameters"]["dielectric_height_mm"]
    grounds = _ground_planes(design)
    plane_faces = {round(z, 6) for g in grounds for z in (g["start"][2], g["stop"][2])}

    ports = design["geometry"]["ports"]
    assert len(ports) == 2
    for port, signal_z in zip(ports, (0.0, h), strict=True):
        z_start, z_stop = port["start"][2], port["stop"][2]
        assert round(z_start, 6) in plane_faces, port
        assert z_stop == pytest.approx(signal_z), port
        assert port["impedance"] == pytest.approx(50.0)


@pytest.mark.parametrize("kind", ["microstrip", "coupled", "via"])
def test_pcb_scripts_are_valid_python(designer, kind):
    if kind == "microstrip":
        did = designer.create_microstrip_trace(2.4, 3.0, 20, 1.6)["design_id"]
    elif kind == "coupled":
        did = designer.create_coupled_lines(2.4, 0.2, 0.2, 20, 0.2)["design_id"]
    else:
        did = _via(designer)["design_id"]
    script = designer.generate_openems_script(did)
    assert script["success"] is True
    compile(script["script"], f"<{kind}>", "exec")

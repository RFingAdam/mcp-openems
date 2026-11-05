"""Regression tests for the analytical models behind coupled lines, horn and monopole.

None of these need openEMS installed.
"""
from __future__ import annotations

import math

import pytest

from mcp_openems.server import AntennaDesigner, PCBStructureDesigner

C0 = 299792458.0


@pytest.fixture
def pcb():
    return PCBStructureDesigner()


@pytest.fixture
def ant():
    return AntennaDesigner()


def _coupled(pcb, w, s, h, er):
    result = pcb.create_coupled_lines(
        frequency_ghz=5.0, trace_width_mm=w, spacing_mm=s, trace_length_mm=30.0,
        dielectric_height_mm=h, substrate_er=er,
    )
    assert result["success"] is True, result
    return result["design"]["calculated"]


# ---------------------------------------------------------------------------
# (d) Coupled microstrip: real even/odd-mode model, consistent reporting
# ---------------------------------------------------------------------------

def test_coupled_lines_names_the_model_actually_used(pcb):
    calc = _coupled(pcb, 3.0, 0.5, 1.6, 4.4)

    assert "Kirschning-Jansen" in calc["model"]
    assert "static" in calc["model"]
    assert calc["model_warnings"] == []


@pytest.mark.parametrize(("w", "s", "h", "er"), [(3.0, 0.5, 1.6, 4.4), (0.3, 0.15, 0.2, 4.2), (1.0, 0.1, 1.0, 4.2)])
def test_coupled_lines_mode_quantities_are_consistent(pcb, w, s, h, er):
    calc = _coupled(pcb, w, s, h, er)
    ze, zo = calc["z_even_ohms"], calc["z_odd_ohms"]

    assert ze > calc["z0_single_ohms"] > zo
    assert calc["effective_er_even"] > calc["effective_er_single"] > calc["effective_er_odd"]
    assert calc["z_diff_ohms"] == pytest.approx(2 * zo, abs=0.011)
    assert calc["z_common_ohms"] == pytest.approx(ze / 2, abs=0.011)
    assert calc["coupling_coefficient"] == pytest.approx((ze - zo) / (ze + zo), abs=2e-4)


def test_coupled_lines_weak_coupling_limit(pcb):
    """At S/h = 10 the lines are effectively uncoupled: Z0e ~ Z0o ~ Z0 and Z0e*Z0o ~ Z0^2."""
    calc = _coupled(pcb, 1.0, 10.0, 1.0, 4.2)
    z0 = calc["z0_single_ohms"]

    assert calc["z_even_ohms"] == pytest.approx(z0, rel=0.01)
    assert calc["z_odd_ohms"] == pytest.approx(z0, rel=0.01)
    assert calc["z_even_ohms"] * calc["z_odd_ohms"] == pytest.approx(z0 ** 2, rel=0.01)
    assert calc["coupling_coefficient"] < 0.01


def test_coupled_lines_agrees_with_independent_garg_bahl_model(pcb):
    """W = 3.0, S = 0.5, h = 1.6 mm, er = 4.4.

    An independent quasi-static Garg-Bahl capacitance model (the coupled
    microstrip analysis in this suite's qucs-s package) gives Z0e = 61.67 ohm,
    Z0o = 37.07 ohm, eps_e = 3.561, eps_o = 2.883. Kirschning-Jansen must agree
    within the few-percent spread between the two closed forms.
    """
    calc = _coupled(pcb, 3.0, 0.5, 1.6, 4.4)

    assert calc["z_even_ohms"] == pytest.approx(61.67, rel=0.03)
    assert calc["z_odd_ohms"] == pytest.approx(37.07, rel=0.03)
    assert calc["effective_er_even"] == pytest.approx(3.561, rel=0.03)
    assert calc["effective_er_odd"] == pytest.approx(2.883, rel=0.03)
    assert calc["z_diff_ohms"] == pytest.approx(2 * 37.07, rel=0.03)


def test_coupled_lines_flags_out_of_range_geometry(pcb):
    calc = _coupled(pcb, 1.0, 0.05, 1.0, 4.2)

    assert any("S/h" in w for w in calc["model_warnings"])


# ---------------------------------------------------------------------------
# (e) Horn: reported gain follows from the generated aperture
# ---------------------------------------------------------------------------

def _aperture_gain_dbi(calc, dims, f_ghz):
    lam_mm = C0 / (f_ghz * 1e9) * 1e3
    area = dims["aperture_width_mm"] * dims["aperture_height_mm"]
    return 10 * math.log10(calc["aperture_efficiency"] * 4 * math.pi * area / lam_mm ** 2)


@pytest.mark.parametrize("target", [12.0, 15.0, 20.0])
def test_horn_reported_gain_matches_its_own_aperture(ant, target):
    design = ant.create_horn_antenna(frequency_ghz=10.0, gain_dbi=target)["design"]
    calc, dims = design["calculated"], design["dimensions"]

    assert calc["estimated_gain_dbi"] == pytest.approx(_aperture_gain_dbi(calc, dims, 10.0), abs=0.02)
    assert calc["estimated_gain_dbi"] == pytest.approx(target, abs=0.02)
    assert calc["aperture_efficiency"] == 0.51


def test_horn_gain_for_the_old_default_aperture(ant):
    """The old default (15 dBi at 10 GHz) produced a 67.25 x 47.08 mm aperture
    but reported 15 dBi. With e_ap = 0.51 that aperture gives
    10*log10(0.51 * 4*pi * 67.25*47.08 / 29.979^2) = 13.54 dBi."""
    design = ant.create_horn_antenna(
        frequency_ghz=10.0, aperture_width_mm=67.25, aperture_height_mm=47.08
    )["design"]

    assert design["calculated"]["estimated_gain_dbi"] == pytest.approx(13.54, abs=0.02)
    assert "Balanis" in design["calculated"]["aperture_efficiency_basis"]


def test_horn_aperture_efficiency_override(ant):
    d50 = ant.create_horn_antenna(frequency_ghz=10.0, aperture_width_mm=67.25, aperture_height_mm=47.08,
                                  aperture_efficiency=0.5)["design"]["calculated"]
    assert d50["estimated_gain_dbi"] == pytest.approx(13.54 + 10 * math.log10(0.5 / 0.51), abs=0.02)
    assert d50["aperture_efficiency_basis"] == "user value"


# ---------------------------------------------------------------------------
# (f) Monopole: infinite-ground assumption is explicit; small ground is flagged
# ---------------------------------------------------------------------------

def test_monopole_on_small_ground_plane_reports_no_gain(ant):
    calc = ant.create_monopole_antenna(frequency_ghz=1.0, ground_plane_mm=100)["design"]["calculated"]

    assert calc["ground_plane_wavelengths"] == pytest.approx(0.33, abs=0.01)
    assert calc["small_ground_plane"] is True
    assert calc["directivity_dbi"] is None
    assert "not estimated for finite ground" in calc["gain_status"]
    assert calc["directivity_dbi_infinite_ground"] == 5.15
    assert "infinite" in calc["ground_plane_assumption"]


def test_monopole_on_large_ground_plane_keeps_caveat(ant):
    calc = ant.create_monopole_antenna(frequency_ghz=2.4, ground_plane_mm=300)["design"]["calculated"]

    assert calc["small_ground_plane"] is False
    assert calc["directivity_dbi"] == 5.15
    assert "infinite-ground value" in calc["gain_status"]


def test_compare_designs_does_not_rank_unestimated_gain(ant):
    mono = ant.create_monopole_antenna(frequency_ghz=1.0, ground_plane_mm=100)["design_id"]
    dip = ant.create_dipole_antenna(frequency_ghz=1.0)["design_id"]

    result = ant.compare_designs([mono, dip])
    assert result["success"] is True
    text = " ".join(result.get("recommendations", []))
    assert "Monopole" not in text.split("Most compact")[0]

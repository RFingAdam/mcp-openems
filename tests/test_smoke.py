"""Smoke tests for mcp-openems.

These tests don't run an actual FDTD simulation — they just verify the
MCP server module is importable, the tool surface is wired correctly,
and the analytical design calculators (which work without openEMS being
installed) give sensible numbers.

The full FDTD validation lives in `examples/microstrip_50ohm.py` and
runs as a documentation example, not in CI.
"""
from __future__ import annotations

import pytest

import mcp_openems


def test_package_importable():
    """The package imports without optional openEMS dependency."""
    assert mcp_openems is not None


def test_server_module_importable():
    """The MCP server module imports."""
    from mcp_openems import server
    assert server.server is not None


def test_antenna_designer_creates_patch():
    """Patch-antenna calculator returns sane dimensions for 2.4 GHz on FR4."""
    from mcp_openems.server import AntennaDesigner

    d = AntennaDesigner()
    result = d.create_patch_antenna(
        frequency_ghz=2.4,
        substrate_er=4.4,
        substrate_height_mm=1.6,
        name="test_patch_2g4",
    )
    assert result["success"] is True

    dims = result["design"]["dimensions"]
    # Patch length for 2.4 GHz on FR4 is ~28-30 mm by the transmission-line
    # model. Width is typically ~38 mm. Loose bounds — smoke check.
    assert 20 <= dims["patch_length_mm"] <= 40
    assert 30 <= dims["patch_width_mm"] <= 50


def test_antenna_designer_creates_dipole():
    """Dipole length for 100 MHz should be ~1.4 m (half-wavelength)."""
    from mcp_openems.server import AntennaDesigner

    d = AntennaDesigner()
    result = d.create_dipole_antenna(
        frequency_ghz=0.1,
        name="test_dipole_100m",
    )
    assert result["success"] is True

    dims = result["design"]["dimensions"]
    # Half-wavelength at 100 MHz in free space is 1.5 m; thin-wire
    # adjustment brings it to ~1.42 m. Generous bounds.
    assert 1200 <= dims["total_length_mm"] <= 1600


@pytest.mark.skipif(
    not getattr(__import__("mcp_openems.server", fromlist=["OPENEMS_AVAILABLE"]),
                "OPENEMS_AVAILABLE", False),
    reason="openEMS not installed (this is expected in CI)",
)
def test_openems_install_check():
    """If openEMS is installed locally, the install check should report OK."""
    import openEMS

    from mcp_openems.server import AntennaDesigner  # noqa: F401
    assert openEMS is not None

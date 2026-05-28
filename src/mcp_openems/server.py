#!/usr/bin/env python3
"""OpenEMS Electromagnetic Simulation MCP Server.

Standalone MCP server for AI-assisted antenna and RF structure design using
OpenEMS FDTD simulation. Provides design calculators, geometry generation,
and simulation script export.

Usage:
    mcp-openems                    # Run as MCP server
    python -m mcp_openems.server   # Alternative invocation
"""

import asyncio
import json
import math
import sys
from typing import Any
from uuid import uuid4

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Check for OpenEMS availability
OPENEMS_AVAILABLE = False
try:
    import CSXCAD
    import openEMS
    OPENEMS_AVAILABLE = True
except ImportError:
    pass

# Create MCP server instance
server = Server("openems-simulator")


class AntennaDesigner:
    """Calculates antenna dimensions and generates OpenEMS geometry."""

    def __init__(self):
        self.designs: dict[str, dict] = {}
        self.simulations: dict[str, dict] = {}

    def create_patch_antenna(
        self,
        frequency_ghz: float,
        substrate_er: float = 4.4,
        substrate_height_mm: float = 1.6,
        name: str = "Patch Antenna",
    ) -> dict:
        """Design a rectangular patch antenna using transmission line model.

        Args:
            frequency_ghz: Center frequency in GHz
            substrate_er: Dielectric constant of substrate
            substrate_height_mm: Substrate thickness in mm
            name: Design name

        Returns:
            Design specification with calculated dimensions
        """
        design_id = str(uuid4())
        c = 299792458  # Speed of light m/s
        f = frequency_ghz * 1e9
        h = substrate_height_mm / 1000  # Convert to meters
        er = substrate_er

        # Calculate patch width (for good radiation efficiency)
        w = (c / (2 * f)) * math.sqrt(2 / (er + 1))
        w_mm = w * 1000

        # Effective permittivity
        er_eff = (er + 1) / 2 + ((er - 1) / 2) * (1 + 12 * h / w) ** -0.5

        # Fringing extension
        delta_l = (
            0.412 * h * ((er_eff + 0.3) * (w / h + 0.264)) /
            ((er_eff - 0.258) * (w / h + 0.8))
        )

        # Effective length and actual length
        l_eff = c / (2 * f * math.sqrt(er_eff))
        l = l_eff - 2 * delta_l
        l_mm = l * 1000

        # Feed inset for 50 ohm match (approximate)
        # R_in = R_edge * cos^2(pi * x / L)
        r_edge = 90 * (er ** 2) / (er - 1) * (l / w) ** 2  # Edge impedance
        feed_inset = (l / math.pi) * math.acos(math.sqrt(50 / r_edge))
        feed_inset_mm = feed_inset * 1000

        # Ground plane size (typically 2x patch dimension)
        ground_mm = max(l_mm, w_mm) * 2 + 20

        # Estimated directivity using cavity model approximation
        # D ≈ 4(kW)² / (π × I₁) where k = 2π/λ and I₁ ≈ 1.0-1.2
        # This gives typical values of 6-8 dBi for rectangular patches
        wavelength = c / f
        k = 2 * math.pi / wavelength
        i1 = 1.0 + 0.1 * (w / l - 1)  # Pattern integral, varies slightly with aspect ratio
        directivity_linear = 4 * (k * w) ** 2 / (math.pi * i1)
        # Account for finite ground plane and substrate losses
        efficiency = 0.9 - 0.05 * (er - 2.2) / 2.2  # Lower efficiency for higher εr
        directivity_dbi = 10 * math.log10(directivity_linear * efficiency)

        design = {
            "id": design_id,
            "name": name,
            "type": "patch",
            "frequency_ghz": frequency_ghz,
            "parameters": {
                "substrate_er": er,
                "substrate_height_mm": substrate_height_mm,
            },
            "dimensions": {
                "patch_length_mm": round(l_mm, 3),
                "patch_width_mm": round(w_mm, 3),
                "feed_inset_mm": round(feed_inset_mm, 3),
                "ground_plane_mm": round(ground_mm, 2),
            },
            "calculated": {
                "effective_er": round(er_eff, 3),
                "fringing_extension_mm": round(delta_l * 1000, 4),
                "edge_impedance_ohms": round(r_edge, 1),
                "estimated_directivity_dbi": round(directivity_dbi, 1),
            },
            "geometry": self._patch_geometry(
                l_mm, w_mm, feed_inset_mm, ground_mm, substrate_height_mm
            ),
        }

        self.designs[design_id] = design
        return {"success": True, "design_id": design_id, "design": design}

    def _patch_geometry(
        self, l_mm: float, w_mm: float, feed_mm: float, ground_mm: float, h_mm: float
    ) -> dict:
        """Generate OpenEMS-compatible geometry specification."""
        return {
            "unit": "mm",
            "primitives": [
                {
                    "type": "box",
                    "name": "ground_plane",
                    "material": "PEC",
                    "start": [-ground_mm / 2, -ground_mm / 2, 0],
                    "stop": [ground_mm / 2, ground_mm / 2, 0],
                },
                {
                    "type": "box",
                    "name": "substrate",
                    "material": "substrate",
                    "start": [-ground_mm / 2, -ground_mm / 2, 0],
                    "stop": [ground_mm / 2, ground_mm / 2, h_mm],
                },
                {
                    "type": "box",
                    "name": "patch",
                    "material": "PEC",
                    "start": [-l_mm / 2, -w_mm / 2, h_mm],
                    "stop": [l_mm / 2, w_mm / 2, h_mm],
                },
            ],
            "ports": [
                {
                    "type": "lumped",
                    "position": [-l_mm / 2 + feed_mm, 0, h_mm / 2],
                    "direction": "z",
                    "impedance": 50,
                }
            ],
        }

    def create_dipole_antenna(
        self,
        frequency_ghz: float,
        wire_radius_mm: float = 1.0,
        name: str = "Dipole Antenna",
    ) -> dict:
        """Design a half-wave dipole antenna.

        Args:
            frequency_ghz: Center frequency in GHz
            wire_radius_mm: Wire radius in mm
            name: Design name

        Returns:
            Design specification with calculated dimensions
        """
        design_id = str(uuid4())
        c = 299792458
        f = frequency_ghz * 1e9
        wavelength = c / f
        wavelength_mm = wavelength * 1000

        # Half-wave dipole with end-effect shortening (typically 2-5%)
        shortening = 0.95
        total_length_mm = (wavelength_mm / 2) * shortening
        arm_length_mm = total_length_mm / 2
        gap_mm = 2.0  # Feed gap

        # Theoretical impedance (72 + j42.5 ohms for infinitely thin)
        # Adjust for wire thickness
        a = wire_radius_mm / 1000
        omega = 2 * math.pi * (total_length_mm / 1000) / (2 * a)
        z_real = 73.1  # Simplified, actual varies with thickness
        z_imag = 42.5 * (1 - 0.1 * math.log10(omega))

        design = {
            "id": design_id,
            "name": name,
            "type": "dipole",
            "frequency_ghz": frequency_ghz,
            "parameters": {
                "wire_radius_mm": wire_radius_mm,
            },
            "dimensions": {
                "total_length_mm": round(total_length_mm, 2),
                "arm_length_mm": round(arm_length_mm, 2),
                "gap_mm": gap_mm,
                "wavelength_mm": round(wavelength_mm, 2),
            },
            "calculated": {
                "impedance_real_ohms": round(z_real, 1),
                "impedance_imag_ohms": round(z_imag, 1),
                "directivity_dbi": 2.15,  # Theoretical half-wave dipole
                "half_power_beamwidth_deg": 78,
            },
            "geometry": {
                "unit": "mm",
                "primitives": [
                    {
                        "type": "cylinder",
                        "name": "arm1",
                        "material": "PEC",
                        "start": [0, 0, gap_mm / 2],
                        "stop": [0, 0, gap_mm / 2 + arm_length_mm],
                        "radius": wire_radius_mm,
                    },
                    {
                        "type": "cylinder",
                        "name": "arm2",
                        "material": "PEC",
                        "start": [0, 0, -gap_mm / 2],
                        "stop": [0, 0, -gap_mm / 2 - arm_length_mm],
                        "radius": wire_radius_mm,
                    },
                ],
                "ports": [
                    {
                        "type": "lumped",
                        "position": [0, 0, 0],
                        "direction": "z",
                        "impedance": 73,
                    }
                ],
            },
        }

        self.designs[design_id] = design
        return {"success": True, "design_id": design_id, "design": design}

    def create_horn_antenna(
        self,
        frequency_ghz: float,
        gain_dbi: float = 15,
        name: str = "Horn Antenna",
    ) -> dict:
        """Design a pyramidal horn antenna for specified gain.

        Args:
            frequency_ghz: Center frequency in GHz
            gain_dbi: Target gain in dBi (typically 10-25)
            name: Design name

        Returns:
            Design specification with calculated dimensions
        """
        design_id = str(uuid4())
        c = 299792458
        f = frequency_ghz * 1e9
        wavelength = c / f
        wavelength_mm = wavelength * 1000

        # WR-xx waveguide dimensions (approximate for frequency)
        # Using standard waveguide sizing
        a_wg = wavelength_mm * 0.7  # Waveguide width
        b_wg = a_wg / 2  # Waveguide height

        # Calculate aperture dimensions from gain
        # G = 10 * log10(4 * pi * A_e / lambda^2)
        # A_e = aperture efficiency * A_physical (efficiency ~ 0.5)
        g_linear = 10 ** (gain_dbi / 10)
        a_physical = g_linear * (wavelength ** 2) / (4 * math.pi * 0.5)
        a_physical_mm = math.sqrt(a_physical) * 1000

        # Aperture dimensions (pyramidal horn)
        a_horn = a_physical_mm  # E-plane aperture
        b_horn = a_physical_mm * 0.7  # H-plane aperture

        # Horn length (optimum horn has specific length for max efficiency)
        r_e = (a_horn ** 2) / (3 * wavelength_mm)  # E-plane slant length
        r_h = (b_horn ** 2) / (2 * wavelength_mm)  # H-plane slant length
        horn_length = max(r_e, r_h) * 0.8

        design = {
            "id": design_id,
            "name": name,
            "type": "horn",
            "frequency_ghz": frequency_ghz,
            "parameters": {
                "target_gain_dbi": gain_dbi,
            },
            "dimensions": {
                "waveguide_width_mm": round(a_wg, 2),
                "waveguide_height_mm": round(b_wg, 2),
                "aperture_width_mm": round(a_horn, 2),
                "aperture_height_mm": round(b_horn, 2),
                "horn_length_mm": round(horn_length, 2),
                "wavelength_mm": round(wavelength_mm, 2),
            },
            "calculated": {
                "estimated_gain_dbi": gain_dbi,
                "aperture_efficiency": 0.5,
                "e_plane_beamwidth_deg": round(56 * wavelength_mm / a_horn, 1),
                "h_plane_beamwidth_deg": round(67 * wavelength_mm / b_horn, 1),
            },
            "geometry": {
                "unit": "mm",
                "description": "Pyramidal horn - requires CSG or mesh for accurate modeling",
                "primitives": [
                    {
                        "type": "horn",
                        "name": "horn_body",
                        "material": "PEC",
                        "waveguide_start": [0, 0, 0],
                        "waveguide_width": a_wg,
                        "waveguide_height": b_wg,
                        "aperture_width": a_horn,
                        "aperture_height": b_horn,
                        "length": horn_length,
                    }
                ],
                "ports": [
                    {
                        "type": "waveguide",
                        "position": [0, 0, 0],
                        "a": a_wg,
                        "b": b_wg,
                    }
                ],
            },
        }

        self.designs[design_id] = design
        return {"success": True, "design_id": design_id, "design": design}

    def create_monopole_antenna(
        self,
        frequency_ghz: float,
        ground_plane_mm: float = 100,
        wire_radius_mm: float = 1.0,
        name: str = "Monopole Antenna",
    ) -> dict:
        """Design a quarter-wave monopole antenna.

        Args:
            frequency_ghz: Center frequency in GHz
            ground_plane_mm: Ground plane diameter in mm
            wire_radius_mm: Wire radius in mm
            name: Design name

        Returns:
            Design specification
        """
        design_id = str(uuid4())
        c = 299792458
        f = frequency_ghz * 1e9
        wavelength = c / f
        wavelength_mm = wavelength * 1000

        # Quarter-wave with shortening
        height_mm = (wavelength_mm / 4) * 0.95

        # Monopole impedance is half of dipole
        z_real = 36.5

        design = {
            "id": design_id,
            "name": name,
            "type": "monopole",
            "frequency_ghz": frequency_ghz,
            "parameters": {
                "ground_plane_mm": ground_plane_mm,
                "wire_radius_mm": wire_radius_mm,
            },
            "dimensions": {
                "height_mm": round(height_mm, 2),
                "ground_plane_diameter_mm": ground_plane_mm,
                "wavelength_mm": round(wavelength_mm, 2),
            },
            "calculated": {
                "impedance_ohms": z_real,
                "directivity_dbi": 5.15,  # 3 dB over dipole due to ground plane
                "ground_plane_wavelengths": round(ground_plane_mm / wavelength_mm, 2),
            },
            "geometry": {
                "unit": "mm",
                "primitives": [
                    {
                        "type": "cylinder",
                        "name": "ground_plane",
                        "material": "PEC",
                        "start": [0, 0, -0.5],
                        "stop": [0, 0, 0],
                        "radius": ground_plane_mm / 2,
                    },
                    {
                        "type": "cylinder",
                        "name": "monopole",
                        "material": "PEC",
                        "start": [0, 0, 0],
                        "stop": [0, 0, height_mm],
                        "radius": wire_radius_mm,
                    },
                ],
                "ports": [
                    {
                        "type": "lumped",
                        "position": [0, 0, 0.5],
                        "direction": "z",
                        "impedance": 36.5,
                    }
                ],
            },
        }

        self.designs[design_id] = design
        return {"success": True, "design_id": design_id, "design": design}

    def create_helix_antenna(
        self,
        frequency_ghz: float,
        turns: int = 10,
        name: str = "Helix Antenna",
    ) -> dict:
        """Design an axial-mode helical antenna for circular polarization.

        Args:
            frequency_ghz: Center frequency in GHz
            turns: Number of turns
            name: Design name

        Returns:
            Design specification
        """
        design_id = str(uuid4())
        c = 299792458
        f = frequency_ghz * 1e9
        wavelength = c / f
        wavelength_mm = wavelength * 1000

        # Axial mode: circumference ~ 1 wavelength
        circumference_mm = wavelength_mm
        diameter_mm = circumference_mm / math.pi

        # Pitch angle typically 12-14 degrees for axial mode
        pitch_angle = 13
        spacing_mm = circumference_mm * math.tan(math.radians(pitch_angle))

        # Total length
        total_length_mm = turns * spacing_mm

        # Ground plane typically 0.75-1.0 wavelength diameter
        ground_mm = wavelength_mm * 0.8

        # Gain approximation for axial mode helix
        gain_dbi = 10.8 + 10 * math.log10(
            (circumference_mm / wavelength_mm) ** 2 * turns * spacing_mm / wavelength_mm
        )

        # Impedance
        z_real = 140 * circumference_mm / wavelength_mm

        design = {
            "id": design_id,
            "name": name,
            "type": "helix",
            "frequency_ghz": frequency_ghz,
            "parameters": {
                "turns": turns,
            },
            "dimensions": {
                "diameter_mm": round(diameter_mm, 2),
                "circumference_mm": round(circumference_mm, 2),
                "spacing_mm": round(spacing_mm, 2),
                "total_length_mm": round(total_length_mm, 2),
                "ground_plane_mm": round(ground_mm, 2),
                "pitch_angle_deg": pitch_angle,
                "wavelength_mm": round(wavelength_mm, 2),
            },
            "calculated": {
                "estimated_gain_dbi": round(gain_dbi, 1),
                "impedance_ohms": round(z_real, 1),
                "polarization": "RHCP",  # Right-hand circular
                "axial_ratio_db": 1.0,  # Typical for well-designed helix
            },
            "geometry": {
                "unit": "mm",
                "primitives": [
                    {
                        "type": "disc",
                        "name": "ground_plane",
                        "material": "PEC",
                        "center": [0, 0, 0],
                        "radius": ground_mm / 2,
                    },
                    {
                        "type": "helix",
                        "name": "helix_wire",
                        "material": "PEC",
                        "center": [0, 0, 0],
                        "radius": diameter_mm / 2,
                        "pitch": spacing_mm,
                        "turns": turns,
                        "wire_radius": 1.0,
                    },
                ],
                "ports": [
                    {
                        "type": "lumped",
                        "position": [diameter_mm / 2, 0, 0],
                        "direction": "z",
                        "impedance": round(z_real, 0),
                    }
                ],
            },
        }

        self.designs[design_id] = design
        return {"success": True, "design_id": design_id, "design": design}

    def generate_openems_script(self, design_id: str) -> dict:
        """Generate Python script for OpenEMS simulation.

        Args:
            design_id: ID of the design

        Returns:
            OpenEMS Python script as string
        """
        if design_id not in self.designs:
            return {"success": False, "error": f"Design not found: {design_id}"}

        design = self.designs[design_id]
        design_type = design["type"]

        # Generate appropriate script based on design type
        if design_type == "patch":
            script = self._generate_patch_script(design)
        elif design_type == "dipole":
            script = self._generate_dipole_script(design)
        elif design_type == "monopole":
            script = self._generate_monopole_script(design)
        elif design_type == "helix":
            script = self._generate_helix_script(design)
        elif design_type == "horn":
            script = self._generate_horn_script(design)
        else:
            script = self._generate_generic_script(design)

        return {
            "success": True,
            "design_id": design_id,
            "script_type": "openems_python",
            "script": script,
            "instructions": [
                "1. Save this script as simulate_antenna.py",
                "2. Install OpenEMS: pip install CSXCAD openEMS",
                "3. Run: python simulate_antenna.py",
                "4. View results in ParaView or OpenEMS AppCSXCAD",
            ],
        }

    def _generate_patch_script(self, design: dict) -> str:
        """Generate OpenEMS script for patch antenna."""
        dims = design["dimensions"]
        params = design["parameters"]

        return f'''#!/usr/bin/env python3
"""OpenEMS Patch Antenna Simulation
Design: {design["name"]}
Frequency: {design["frequency_ghz"]} GHz
Generated by mcp-openems
"""

import os
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

# Design parameters
f0 = {design["frequency_ghz"]}e9  # Center frequency
fc = f0 * 0.5  # 20 dB bandwidth
substrate_er = {params["substrate_er"]}
substrate_h = {params["substrate_height_mm"]}  # mm

# Calculated dimensions
patch_l = {dims["patch_length_mm"]}  # mm
patch_w = {dims["patch_width_mm"]}  # mm
feed_inset = {dims["feed_inset_mm"]}  # mm
ground_size = {dims["ground_plane_mm"]}  # mm

# Simulation setup
unit = 1e-3  # mm
FDTD = openEMS(EndCriteria=1e-4)
FDTD.SetGaussExcite(f0, fc)
FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8'])

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)

# Materials
substrate = CSX.AddMaterial('substrate', epsilon=substrate_er)
metal = CSX.AddMetal('PEC')

# Geometry
# Ground plane
metal.AddBox(
    start=[-ground_size/2, -ground_size/2, 0],
    stop=[ground_size/2, ground_size/2, 0],
    priority=10
)

# Substrate
substrate.AddBox(
    start=[-ground_size/2, -ground_size/2, 0],
    stop=[ground_size/2, ground_size/2, substrate_h],
    priority=1
)

# Patch
metal.AddBox(
    start=[-patch_l/2, -patch_w/2, substrate_h],
    stop=[patch_l/2, patch_w/2, substrate_h],
    priority=10
)

# Feed port (lumped port)
feed_pos = -patch_l/2 + feed_inset
port = FDTD.AddLumpedPort(
    1, 50,
    start=[feed_pos, -0.5, 0],
    stop=[feed_pos, 0.5, substrate_h],
    p_dir='z',
    excite=1
)

# Mesh
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)

# Mesh resolution
resolution = C0 / (f0 + fc) / unit / 20
mesh.AddLine('x', np.concatenate([
    np.arange(-ground_size/2 - 20, -patch_l/2, resolution),
    np.linspace(-patch_l/2, patch_l/2, 15),
    np.arange(patch_l/2, ground_size/2 + 20, resolution)
]))
mesh.AddLine('y', np.concatenate([
    np.arange(-ground_size/2 - 20, -patch_w/2, resolution),
    np.linspace(-patch_w/2, patch_w/2, 15),
    np.arange(patch_w/2, ground_size/2 + 20, resolution)
]))
mesh.AddLine('z', np.concatenate([
    np.arange(-10, 0, resolution),
    np.linspace(0, substrate_h, 5),
    np.arange(substrate_h, 30, resolution)
]))

# NF2FF box for far-field
nf2ff = FDTD.CreateNF2FFBox()

# Run simulation
import tempfile
Sim_Path = os.path.join(tempfile.gettempdir(), 'openems_patch')
if os.path.exists(Sim_Path):
    import shutil
    shutil.rmtree(Sim_Path)
os.makedirs(Sim_Path)

FDTD.Run(Sim_Path, verbose=3, cleanup=True)

# Post-processing
f = np.linspace(f0 - fc, f0 + fc, 201)
port.CalcPort(Sim_Path, f)
s11 = port.uf_ref / port.uf_inc
s11_dB = 20 * np.log10(np.abs(s11))

# Find resonance
idx = np.argmin(s11_dB)
print(f"Resonant frequency: {{f[idx]/1e9:.3f}} GHz")
print(f"S11 at resonance: {{s11_dB[idx]:.1f}} dB")

# Save results
import matplotlib.pyplot as plt
plt.figure()
plt.plot(f/1e9, s11_dB)
plt.xlabel('Frequency (GHz)')
plt.ylabel('S11 (dB)')
plt.title('Patch Antenna S11')
plt.grid(True)
plt.savefig('patch_s11.png', dpi=150)
print("S11 plot saved to patch_s11.png")
'''

    def _generate_dipole_script(self, design: dict) -> str:
        """Generate OpenEMS script for dipole antenna."""
        dims = design["dimensions"]

        return f'''#!/usr/bin/env python3
"""OpenEMS Dipole Antenna Simulation
Design: {design["name"]}
Frequency: {design["frequency_ghz"]} GHz
Generated by mcp-openems
"""

import os
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

# Design parameters
f0 = {design["frequency_ghz"]}e9
fc = f0 * 0.3

# Dimensions
arm_length = {dims["arm_length_mm"]}  # mm
gap = {dims["gap_mm"]}  # mm
wire_radius = {design["parameters"]["wire_radius_mm"]}  # mm

unit = 1e-3
FDTD = openEMS(EndCriteria=1e-4)
FDTD.SetGaussExcite(f0, fc)
FDTD.SetBoundaryCond(['PML_8'] * 6)

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)

metal = CSX.AddMetal('PEC')

# Dipole arms as cylinders
metal.AddCylinder(
    start=[0, 0, gap/2],
    stop=[0, 0, gap/2 + arm_length],
    radius=wire_radius,
    priority=10
)
metal.AddCylinder(
    start=[0, 0, -gap/2],
    stop=[0, 0, -gap/2 - arm_length],
    radius=wire_radius,
    priority=10
)

# Feed port
port = FDTD.AddLumpedPort(
    1, 73,
    start=[-wire_radius, -wire_radius, -gap/2],
    stop=[wire_radius, wire_radius, gap/2],
    p_dir='z',
    excite=1
)

# Mesh
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)
resolution = C0 / (f0 + fc) / unit / 20

total_len = arm_length * 2 + gap
sim_box = total_len * 2

mesh.AddLine('x', np.arange(-sim_box/2, sim_box/2, resolution))
mesh.AddLine('y', np.arange(-sim_box/2, sim_box/2, resolution))
mesh.AddLine('z', np.concatenate([
    np.arange(-sim_box/2, -arm_length - gap/2, resolution),
    np.linspace(-arm_length - gap/2, arm_length + gap/2, 30),
    np.arange(arm_length + gap/2, sim_box/2, resolution)
]))

nf2ff = FDTD.CreateNF2FFBox()

import tempfile
Sim_Path = os.path.join(tempfile.gettempdir(), 'openems_dipole')
if os.path.exists(Sim_Path):
    import shutil
    shutil.rmtree(Sim_Path)
os.makedirs(Sim_Path)

FDTD.Run(Sim_Path, verbose=3, cleanup=True)

# Post-processing
f = np.linspace(f0 - fc, f0 + fc, 201)
port.CalcPort(Sim_Path, f)
s11 = port.uf_ref / port.uf_inc
s11_dB = 20 * np.log10(np.abs(s11))

idx = np.argmin(s11_dB)
print(f"Resonant frequency: {{f[idx]/1e9:.3f}} GHz")
print(f"S11 at resonance: {{s11_dB[idx]:.1f}} dB")

import matplotlib.pyplot as plt
plt.figure()
plt.plot(f/1e9, s11_dB)
plt.xlabel('Frequency (GHz)')
plt.ylabel('S11 (dB)')
plt.title('Dipole Antenna S11')
plt.grid(True)
plt.savefig('dipole_s11.png', dpi=150)
print("S11 plot saved to dipole_s11.png")
'''

    def _generate_monopole_script(self, design: dict) -> str:
        """Generate OpenEMS script for monopole antenna."""
        dims = design["dimensions"]
        params = design["parameters"]

        return f'''#!/usr/bin/env python3
"""OpenEMS Monopole Antenna Simulation
Design: {design["name"]}
Frequency: {design["frequency_ghz"]} GHz
Generated by mcp-openems
"""

import os
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

f0 = {design["frequency_ghz"]}e9
fc = f0 * 0.3

height = {dims["height_mm"]}
ground_r = {params["ground_plane_mm"]} / 2
wire_r = {params["wire_radius_mm"]}

unit = 1e-3
FDTD = openEMS(EndCriteria=1e-4)
FDTD.SetGaussExcite(f0, fc)
FDTD.SetBoundaryCond(['PML_8'] * 6)

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)

metal = CSX.AddMetal('PEC')

# Ground plane
metal.AddCylinder(
    start=[0, 0, -0.5],
    stop=[0, 0, 0],
    radius=ground_r,
    priority=10
)

# Monopole
metal.AddCylinder(
    start=[0, 0, 0],
    stop=[0, 0, height],
    radius=wire_r,
    priority=10
)

# Feed
port = FDTD.AddLumpedPort(
    1, 36.5,
    start=[-wire_r, -wire_r, 0],
    stop=[wire_r, wire_r, 1],
    p_dir='z',
    excite=1
)

mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)
resolution = C0 / (f0 + fc) / unit / 20

sim_box = max(ground_r * 2, height * 4)
mesh.AddLine('x', np.arange(-sim_box/2, sim_box/2, resolution))
mesh.AddLine('y', np.arange(-sim_box/2, sim_box/2, resolution))
mesh.AddLine('z', np.concatenate([
    np.arange(-20, 0, resolution),
    np.linspace(0, height, 20),
    np.arange(height, sim_box/2, resolution)
]))

nf2ff = FDTD.CreateNF2FFBox()

import tempfile
Sim_Path = os.path.join(tempfile.gettempdir(), 'openems_monopole')
if os.path.exists(Sim_Path):
    import shutil
    shutil.rmtree(Sim_Path)
os.makedirs(Sim_Path)

FDTD.Run(Sim_Path, verbose=3, cleanup=True)

f = np.linspace(f0 - fc, f0 + fc, 201)
port.CalcPort(Sim_Path, f)
s11 = port.uf_ref / port.uf_inc
s11_dB = 20 * np.log10(np.abs(s11))

idx = np.argmin(s11_dB)
print(f"Resonant frequency: {{f[idx]/1e9:.3f}} GHz")
print(f"S11 at resonance: {{s11_dB[idx]:.1f}} dB")

import matplotlib.pyplot as plt
plt.figure()
plt.plot(f/1e9, s11_dB)
plt.xlabel('Frequency (GHz)')
plt.ylabel('S11 (dB)')
plt.grid(True)
plt.savefig('monopole_s11.png', dpi=150)
'''

    def _generate_helix_script(self, design: dict) -> str:
        """Generate OpenEMS script for helical antenna."""
        dims = design["dimensions"]
        params = design["parameters"]

        return f'''#!/usr/bin/env python3
"""OpenEMS Helical Antenna Simulation
Design: {design["name"]}
Frequency: {design["frequency_ghz"]} GHz
Turns: {params["turns"]}
Generated by mcp-openems
"""

import os
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

f0 = {design["frequency_ghz"]}e9
fc = f0 * 0.3

# Helix dimensions
radius = {dims["diameter_mm"]} / 2  # mm
pitch = {dims["spacing_mm"]}  # mm (turn spacing)
turns = {params["turns"]}
wire_r = 1.0  # mm
ground_r = {dims["ground_plane_mm"]} / 2  # mm

unit = 1e-3
FDTD = openEMS(EndCriteria=1e-4)
FDTD.SetGaussExcite(f0, fc)
FDTD.SetBoundaryCond(['PML_8'] * 6)

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)

metal = CSX.AddMetal('PEC')

# Ground plane (circular disc approximated as polygon)
n_sides = 32
theta = np.linspace(0, 2*np.pi, n_sides+1)[:-1]
ground_points = np.column_stack([
    ground_r * np.cos(theta),
    ground_r * np.sin(theta),
    np.zeros(n_sides)
])
metal.AddPolygon(ground_points.T, 'z', elevation=0, priority=10)

# Create helix as series of short wire segments
n_segments_per_turn = 20
total_segments = turns * n_segments_per_turn
t = np.linspace(0, turns * 2 * np.pi, total_segments + 1)
z = np.linspace(0, turns * pitch, total_segments + 1)

for i in range(total_segments):
    x0, y0, z0 = radius * np.cos(t[i]), radius * np.sin(t[i]), z[i]
    x1, y1, z1 = radius * np.cos(t[i+1]), radius * np.sin(t[i+1]), z[i+1]
    metal.AddCylinder(
        start=[x0, y0, z0],
        stop=[x1, y1, z1],
        radius=wire_r,
        priority=10
    )

# Feed port at helix start
port = FDTD.AddLumpedPort(
    1, {design["calculated"]["impedance_ohms"]},
    start=[radius - wire_r, -wire_r, 0],
    stop=[radius + wire_r, wire_r, 3],
    p_dir='z',
    excite=1
)

# Mesh
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)
resolution = C0 / (f0 + fc) / unit / 15

total_height = turns * pitch
sim_box = max(ground_r * 3, total_height * 1.5)

mesh.AddLine('x', np.arange(-sim_box, sim_box, resolution))
mesh.AddLine('y', np.arange(-sim_box, sim_box, resolution))
mesh.AddLine('z', np.concatenate([
    np.arange(-20, 0, resolution),
    np.linspace(0, total_height, int(total_height/resolution)*2),
    np.arange(total_height, sim_box, resolution)
]))

nf2ff = FDTD.CreateNF2FFBox()

import tempfile
Sim_Path = os.path.join(tempfile.gettempdir(), 'openems_helix')
if os.path.exists(Sim_Path):
    import shutil
    shutil.rmtree(Sim_Path)
os.makedirs(Sim_Path)

FDTD.Run(Sim_Path, verbose=3, cleanup=True)

# Post-processing
f = np.linspace(f0 - fc, f0 + fc, 201)
port.CalcPort(Sim_Path, f)
s11 = port.uf_ref / port.uf_inc
s11_dB = 20 * np.log10(np.abs(s11))

idx = np.argmin(s11_dB)
print(f"Resonant frequency: {{f[idx]/1e9:.3f}} GHz")
print(f"S11 at resonance: {{s11_dB[idx]:.1f}} dB")
print(f"Expected gain: {design["calculated"]["estimated_gain_dbi"]} dBi")
print(f"Polarization: {design["calculated"]["polarization"]}")

import matplotlib.pyplot as plt
plt.figure()
plt.plot(f/1e9, s11_dB)
plt.xlabel('Frequency (GHz)')
plt.ylabel('S11 (dB)')
plt.title('Helical Antenna S11')
plt.grid(True)
plt.axhline(-10, color='r', linestyle='--', label='-10 dB')
plt.legend()
plt.savefig('helix_s11.png', dpi=150)
print("S11 plot saved to helix_s11.png")
'''

    def _generate_horn_script(self, design: dict) -> str:
        """Generate OpenEMS script for horn antenna."""
        dims = design["dimensions"]

        return f'''#!/usr/bin/env python3
"""OpenEMS Horn Antenna Simulation
Design: {design["name"]}
Frequency: {design["frequency_ghz"]} GHz
Target Gain: {design["parameters"]["target_gain_dbi"]} dBi
Generated by mcp-openems
"""

import os
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

f0 = {design["frequency_ghz"]}e9
fc = f0 * 0.4  # Wider bandwidth for horn

# Waveguide dimensions
wg_a = {dims["waveguide_width_mm"]}  # mm (broad dimension)
wg_b = {dims["waveguide_height_mm"]}  # mm (narrow dimension)

# Horn aperture dimensions
ap_a = {dims["aperture_width_mm"]}  # mm
ap_b = {dims["aperture_height_mm"]}  # mm
horn_len = {dims["horn_length_mm"]}  # mm

# Waveguide feed length
wg_len = {dims["wavelength_mm"]} * 2

unit = 1e-3
FDTD = openEMS(EndCriteria=1e-4)
FDTD.SetGaussExcite(f0, fc)
FDTD.SetBoundaryCond(['PML_8'] * 6)

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)

metal = CSX.AddMetal('PEC')

# Create horn as set of 4 tapered plates
# Back of waveguide at z=0, aperture at z=wg_len+horn_len
wall_thickness = 2  # mm

# Bottom plate
metal.AddLinPoly(
    points=np.array([
        [-wg_a/2-wall_thickness, -wg_a/2-wall_thickness],
        [wg_a/2+wall_thickness, wg_a/2+wall_thickness],
        [ap_a/2+wall_thickness, ap_a/2+wall_thickness],
        [-ap_a/2-wall_thickness, -ap_a/2-wall_thickness]
    ]).T,
    norm_dir='y',
    elevation=-wg_b/2-wall_thickness,
    length=wall_thickness,
    priority=10
)

# Waveguide section
metal.AddBox(
    start=[-wg_a/2-wall_thickness, -wg_b/2-wall_thickness, -wg_len],
    stop=[wg_a/2+wall_thickness, wg_b/2+wall_thickness, 0],
    priority=5
)
# Hollow out waveguide
metal.AddBox(
    start=[-wg_a/2, -wg_b/2, -wg_len],
    stop=[wg_a/2, wg_b/2, 0],
    priority=10
)

# Waveguide port (TE10 mode)
port = FDTD.AddRectWaveGuidePort(
    0, 1,
    start=[-wg_a/2, -wg_b/2, -wg_len+1],
    stop=[wg_a/2, wg_b/2, -wg_len+1],
    p_dir='z',
    a=wg_a * unit,
    b=wg_b * unit,
    mode_name='TE10',
    excite=1
)

# Mesh
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)
resolution = C0 / (f0 + fc) / unit / 15

sim_box_xy = ap_a * 2
sim_box_z = wg_len + horn_len + ap_a

mesh.AddLine('x', np.concatenate([
    np.arange(-sim_box_xy/2, -ap_a/2, resolution),
    np.linspace(-ap_a/2, ap_a/2, 30),
    np.arange(ap_a/2, sim_box_xy/2, resolution)
]))
mesh.AddLine('y', np.concatenate([
    np.arange(-sim_box_xy/2, -ap_b/2, resolution),
    np.linspace(-ap_b/2, ap_b/2, 20),
    np.arange(ap_b/2, sim_box_xy/2, resolution)
]))
mesh.AddLine('z', np.concatenate([
    np.arange(-wg_len - 20, -wg_len, resolution),
    np.linspace(-wg_len, wg_len + horn_len, 50),
    np.arange(wg_len + horn_len, sim_box_z, resolution)
]))

nf2ff = FDTD.CreateNF2FFBox()

import tempfile
Sim_Path = os.path.join(tempfile.gettempdir(), 'openems_horn')
if os.path.exists(Sim_Path):
    import shutil
    shutil.rmtree(Sim_Path)
os.makedirs(Sim_Path)

print("Note: Horn simulation requires careful geometry - this is a template.")
print("For accurate results, use AppCSXCAD to verify geometry before running.")

# Uncomment to run:
# FDTD.Run(Sim_Path, verbose=3, cleanup=True)

print(f"Target gain: {design["calculated"]["estimated_gain_dbi"]} dBi")
print(f"E-plane beamwidth: {design["calculated"]["e_plane_beamwidth_deg"]}°")
print(f"H-plane beamwidth: {design["calculated"]["h_plane_beamwidth_deg"]}°")
'''

    def _generate_generic_script(self, design: dict) -> str:
        """Generate basic OpenEMS script template."""
        return f'''#!/usr/bin/env python3
"""OpenEMS Simulation Template
Design: {design["name"]}
Type: {design["type"]}
Frequency: {design["frequency_ghz"]} GHz
Generated by mcp-openems

This is a template - customize the geometry for your specific design.
"""

import os
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

f0 = {design["frequency_ghz"]}e9
fc = f0 * 0.3

unit = 1e-3
FDTD = openEMS(EndCriteria=1e-4)
FDTD.SetGaussExcite(f0, fc)
FDTD.SetBoundaryCond(['PML_8'] * 6)

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)

metal = CSX.AddMetal('PEC')

# TODO: Add geometry from design specification:
# {json.dumps(design.get("geometry", {}), indent=2)}

# TODO: Add mesh and run simulation

print("Template generated - customize geometry before running")
'''

    def list_designs(self) -> dict:
        """List all created designs."""
        designs = []
        for design_id, design in self.designs.items():
            designs.append({
                "id": design_id,
                "name": design["name"],
                "type": design["type"],
                "frequency_ghz": design["frequency_ghz"],
            })
        return {"success": True, "designs": designs, "count": len(designs)}

    def get_design(self, design_id: str) -> dict:
        """Get full details of a design."""
        if design_id not in self.designs:
            return {"success": False, "error": f"Design not found: {design_id}"}
        return {"success": True, "design": self.designs[design_id]}

    def export_design(self, design_id: str, format: str = "json") -> dict:
        """Export design to various formats for visualization.

        Args:
            design_id: ID of the design
            format: Export format (json, svg, ascii)

        Returns:
            Exported design data
        """
        if design_id not in self.designs:
            return {"success": False, "error": f"Design not found: {design_id}"}

        design = self.designs[design_id]

        if format == "json":
            return {
                "success": True,
                "format": "json",
                "data": design,
                "usage": "Import into CAD software or visualization tools",
            }

        elif format == "svg":
            svg = self._generate_svg(design)
            return {
                "success": True,
                "format": "svg",
                "data": svg,
                "usage": "Open in browser or vector graphics editor",
            }

        elif format == "ascii":
            ascii_art = self._generate_ascii(design)
            return {
                "success": True,
                "format": "ascii",
                "data": ascii_art,
                "usage": "Quick terminal visualization",
            }

        else:
            return {"success": False, "error": f"Unknown format: {format}. Use json, svg, or ascii"}

    def _generate_svg(self, design: dict) -> str:
        """Generate SVG visualization of antenna design."""
        design_type = design["type"]
        dims = design["dimensions"]

        if design_type == "patch":
            # Top view of patch antenna
            patch_l = dims["patch_length_mm"]
            patch_w = dims["patch_width_mm"]
            ground = dims["ground_plane_mm"]
            feed = dims.get("feed_inset_mm", 0)

            scale = 400 / ground  # Scale to fit in 400px
            cx, cy = 250, 250  # Center

            svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500">
  <title>{design["name"]} - Top View</title>
  <style>
    .ground {{ fill: #4a5568; stroke: #2d3748; stroke-width: 2; }}
    .patch {{ fill: #f6ad55; stroke: #dd6b20; stroke-width: 2; }}
    .feed {{ fill: #e53e3e; }}
    .label {{ font-family: sans-serif; font-size: 12px; fill: #2d3748; }}
    .dim {{ font-family: monospace; font-size: 10px; fill: #718096; }}
  </style>

  <!-- Ground plane -->
  <rect class="ground" x="{cx - ground*scale/2}" y="{cy - ground*scale/2}"
        width="{ground*scale}" height="{ground*scale}" rx="2"/>

  <!-- Patch -->
  <rect class="patch" x="{cx - patch_l*scale/2}" y="{cy - patch_w*scale/2}"
        width="{patch_l*scale}" height="{patch_w*scale}"/>

  <!-- Feed point -->
  <circle class="feed" cx="{cx - patch_l*scale/2 + feed*scale}" cy="{cy}" r="5"/>

  <!-- Labels -->
  <text class="label" x="250" y="30" text-anchor="middle">{design["name"]}</text>
  <text class="dim" x="250" y="50" text-anchor="middle">{design["frequency_ghz"]} GHz</text>

  <!-- Dimensions -->
  <text class="dim" x="{cx}" y="{cy - patch_w*scale/2 - 10}" text-anchor="middle">L={patch_l:.1f}mm</text>
  <text class="dim" x="{cx + patch_l*scale/2 + 10}" y="{cy}" text-anchor="start">W={patch_w:.1f}mm</text>
  <text class="dim" x="250" y="480" text-anchor="middle">Ground: {ground:.1f}mm × {ground:.1f}mm</text>
</svg>'''

        elif design_type == "dipole":
            total_len = dims["total_length_mm"]
            arm_len = dims["arm_length_mm"]

            scale = 300 / total_len
            cx, cy = 250, 250

            svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500">
  <title>{design["name"]} - Side View</title>
  <style>
    .wire {{ fill: #f6ad55; stroke: #dd6b20; stroke-width: 2; }}
    .feed {{ fill: #e53e3e; stroke: #c53030; stroke-width: 2; }}
    .label {{ font-family: sans-serif; font-size: 12px; fill: #2d3748; }}
    .dim {{ font-family: monospace; font-size: 10px; fill: #718096; }}
  </style>

  <!-- Upper arm -->
  <rect class="wire" x="{cx-5}" y="{cy - arm_len*scale}" width="10" height="{arm_len*scale - 5}"/>

  <!-- Lower arm -->
  <rect class="wire" x="{cx-5}" y="{cy + 5}" width="10" height="{arm_len*scale - 5}"/>

  <!-- Feed gap -->
  <rect class="feed" x="{cx-8}" y="{cy - 5}" width="16" height="10" rx="2"/>

  <!-- Labels -->
  <text class="label" x="250" y="30" text-anchor="middle">{design["name"]}</text>
  <text class="dim" x="250" y="50" text-anchor="middle">{design["frequency_ghz"]} GHz | Z={design["calculated"]["impedance_real_ohms"]}Ω</text>

  <!-- Dimension lines -->
  <line x1="{cx+30}" y1="{cy - arm_len*scale}" x2="{cx+30}" y2="{cy + arm_len*scale}"
        stroke="#718096" stroke-width="1" marker-start="url(#arrow)" marker-end="url(#arrow)"/>
  <text class="dim" x="{cx+40}" y="{cy}" text-anchor="start">{total_len:.1f}mm</text>

  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="5" refY="5" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#718096"/>
    </marker>
  </defs>
</svg>'''

        elif design_type == "helix":
            diameter = dims["diameter_mm"]
            total_len = dims["total_length_mm"]
            turns = design["parameters"]["turns"]

            scale = min(200 / diameter, 300 / total_len)
            cx = 250

            # Generate helix path
            helix_points = []
            for i in range(turns * 20 + 1):
                angle = i * 2 * math.pi / 20
                x = cx + (diameter/2) * scale * math.sin(angle)
                y = 400 - i * (total_len * scale) / (turns * 20)
                helix_points.append(f"{x:.1f},{y:.1f}")

            svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500">
  <title>{design["name"]} - Side View</title>
  <style>
    .helix {{ fill: none; stroke: #f6ad55; stroke-width: 3; stroke-linecap: round; }}
    .ground {{ fill: #4a5568; stroke: #2d3748; stroke-width: 2; }}
    .label {{ font-family: sans-serif; font-size: 12px; fill: #2d3748; }}
    .dim {{ font-family: monospace; font-size: 10px; fill: #718096; }}
  </style>

  <!-- Ground plane -->
  <ellipse class="ground" cx="{cx}" cy="420" rx="{dims["ground_plane_mm"]*scale/2}" ry="20"/>

  <!-- Helix wire -->
  <polyline class="helix" points="{' '.join(helix_points)}"/>

  <!-- Labels -->
  <text class="label" x="250" y="30" text-anchor="middle">{design["name"]}</text>
  <text class="dim" x="250" y="50" text-anchor="middle">{design["frequency_ghz"]} GHz | {turns} turns | {design["calculated"]["polarization"]}</text>
  <text class="dim" x="250" y="70" text-anchor="middle">Gain: {design["calculated"]["estimated_gain_dbi"]} dBi</text>

  <!-- Dimensions -->
  <text class="dim" x="420" y="300" text-anchor="start">⌀{diameter:.1f}mm</text>
  <text class="dim" x="420" y="320" text-anchor="start">H={total_len:.1f}mm</text>
</svg>'''

        else:
            # Generic text-based SVG
            svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 300">
  <title>{design["name"]}</title>
  <style>.label {{ font-family: sans-serif; font-size: 14px; fill: #2d3748; }}</style>
  <text class="label" x="250" y="30" text-anchor="middle" font-weight="bold">{design["name"]}</text>
  <text class="label" x="250" y="60" text-anchor="middle">Type: {design["type"]}</text>
  <text class="label" x="250" y="90" text-anchor="middle">Frequency: {design["frequency_ghz"]} GHz</text>
  <text class="label" x="250" y="130" text-anchor="middle">Dimensions:</text>
  <text class="label" x="250" y="160" text-anchor="middle" font-family="monospace">{json.dumps(dims)}</text>
</svg>'''

        return svg

    def _generate_ascii(self, design: dict) -> str:
        """Generate ASCII art visualization."""
        design_type = design["type"]
        dims = design["dimensions"]
        calc = design["calculated"]

        header = f"""
╔══════════════════════════════════════════════════════════════╗
║  {design["name"]:^58}  ║
║  Type: {design["type"]:10}  Frequency: {design["frequency_ghz"]} GHz{" "*(35-len(str(design["frequency_ghz"])))}║
╠══════════════════════════════════════════════════════════════╣"""

        if design_type == "patch":
            art = f"""
║                        TOP VIEW                              ║
║        ┌─────────────────────────────────────┐               ║
║        │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ Ground        ║
║        │░░░┌───────────────────────┐░░░░░░░░░│               ║
║        │░░░│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│░░░░░░░░░│               ║
║        │░░░│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│░░░░░░░░░│               ║
║        │░░░│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│░░░░░░░░░│ Patch         ║
║        │░░░│▓●▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│░░░░░░░░░│ ● = Feed      ║
║        │░░░│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│░░░░░░░░░│               ║
║        │░░░└───────────────────────┘░░░░░░░░░│               ║
║        │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│               ║
║        └─────────────────────────────────────┘               ║
║                                                              ║
║  Patch:  L={dims["patch_length_mm"]:6.2f}mm  W={dims["patch_width_mm"]:6.2f}mm{" "*21}║
║  Feed:   Inset={dims["feed_inset_mm"]:5.2f}mm  Impedance=50Ω{" "*18}║
║  Ground: {dims["ground_plane_mm"]:6.2f}mm × {dims["ground_plane_mm"]:6.2f}mm{" "*24}║
║  Gain:   {calc["estimated_directivity_dbi"]:5.1f} dBi (estimated){" "*28}║"""

        elif design_type == "dipole":
            art = f"""
║                       SIDE VIEW                              ║
║                          │                                   ║
║                          │  ← Arm 1                          ║
║                          │     {dims["arm_length_mm"]:.1f}mm                        ║
║                          │                                   ║
║                         ┌┴┐ ← Feed gap                       ║
║                         └┬┘   {dims["gap_mm"]}mm                            ║
║                          │                                   ║
║                          │  ← Arm 2                          ║
║                          │     {dims["arm_length_mm"]:.1f}mm                        ║
║                          │                                   ║
║                                                              ║
║  Total Length: {dims["total_length_mm"]:6.2f}mm  (λ/2 × 0.95){" "*17}║
║  Wavelength:   {dims["wavelength_mm"]:6.2f}mm{" "*30}║
║  Impedance:    {calc["impedance_real_ohms"]:5.1f} + j{calc["impedance_imag_ohms"]:.1f} Ω{" "*25}║
║  Gain:         {calc["directivity_dbi"]:5.2f} dBi{" "*31}║"""

        elif design_type == "helix":
            art = f"""
║                       SIDE VIEW                              ║
║                          ╱╲                                  ║
║                         ╱  ╲   ← {design["parameters"]["turns"]} turns                      ║
║                        │    │                                ║
║                        │    │  Diameter: {dims["diameter_mm"]:.1f}mm              ║
║                         ╲  ╱   Pitch: {dims["spacing_mm"]:.1f}mm                 ║
║                          ╲╱                                  ║
║                          ╱╲                                  ║
║                         ╱  ╲                                 ║
║                        │    │  Total Height: {dims["total_length_mm"]:.1f}mm          ║
║                         ╲  ╱                                 ║
║                          ╲╱                                  ║
║                    ══════╧══════  ← Ground Plane             ║
║                    {dims["ground_plane_mm"]:.1f}mm diameter{" "*27}║
║                                                              ║
║  Gain:        {calc["estimated_gain_dbi"]:5.1f} dBi{" "*32}║
║  Impedance:   {calc["impedance_ohms"]:5.1f} Ω{" "*34}║
║  Polarization: {calc["polarization"]}{" "*(43-len(calc["polarization"]))}║"""

        else:
            dims_str = "\n".join([f"║    {k}: {v}{' '*(52-len(k)-len(str(v)))}║"
                                  for k, v in dims.items()])
            art = f"""
║  Dimensions:{" "*49}║
{dims_str}"""

        footer = """
╚══════════════════════════════════════════════════════════════╝"""

        return header + art + footer

    def get_optimization_hints(self, design_id: str) -> dict:
        """Get suggestions for optimizing an antenna design.

        Args:
            design_id: ID of the design

        Returns:
            Optimization suggestions and parameter sensitivities
        """
        if design_id not in self.designs:
            return {"success": False, "error": f"Design not found: {design_id}"}

        design = self.designs[design_id]
        design_type = design["type"]
        hints = []
        parameter_sensitivity = {}

        if design_type == "patch":
            dims = design["dimensions"]
            params = design["parameters"]

            hints = [
                {
                    "category": "Bandwidth",
                    "issue": "Narrow bandwidth is common with patch antennas",
                    "suggestions": [
                        f"Increase substrate height (current: {params['substrate_height_mm']}mm) - doubles BW per doubling of height",
                        f"Use lower εr substrate (current: {params['substrate_er']}) - Rogers RO4003 (εr=3.55) or RT/duroid (εr=2.2)",
                        "Add parasitic elements for stacked patch design",
                        "Use U-slot or E-shaped patch for wideband operation",
                    ],
                },
                {
                    "category": "Impedance Match",
                    "issue": f"Feed inset calculated for 50Ω (current: {dims['feed_inset_mm']:.2f}mm)",
                    "suggestions": [
                        "Fine-tune feed inset ±10% during simulation",
                        "Consider quarter-wave transformer for edge feed",
                        "Aperture-coupled feed provides better bandwidth",
                    ],
                },
                {
                    "category": "Gain",
                    "issue": f"Estimated directivity: {design['calculated']['estimated_directivity_dbi']:.1f} dBi",
                    "suggestions": [
                        "Use antenna array (2x2 gives +6dB, 4x4 gives +12dB)",
                        "Increase ground plane size for slight improvement",
                        "Reduce substrate losses (use lower loss tangent material)",
                    ],
                },
            ]

            parameter_sensitivity = {
                "patch_length": "±1mm changes resonant frequency by ~±50MHz at 2.4GHz",
                "patch_width": "Affects impedance and bandwidth, less critical than length",
                "feed_inset": "±0.5mm changes input impedance by ~±10Ω",
                "substrate_height": "Doubling increases BW ~2x but also surface waves",
            }

        elif design_type == "dipole":
            hints = [
                {
                    "category": "Impedance Match",
                    "issue": "Dipole impedance ~73Ω, not matched to 50Ω",
                    "suggestions": [
                        "Use a balun with impedance transformation",
                        "Use gamma match or T-match for 50Ω",
                        "Folded dipole gives 4x impedance (~300Ω)",
                    ],
                },
                {
                    "category": "Pattern",
                    "issue": "Omnidirectional in H-plane (donut pattern)",
                    "suggestions": [
                        "Add reflector for directional pattern (+3dB gain)",
                        "Create Yagi-Uda array for high directivity",
                        "Use vertical orientation for omnidirectional coverage",
                    ],
                },
                {
                    "category": "Bandwidth",
                    "issue": "~10% bandwidth typical",
                    "suggestions": [
                        "Increase wire diameter (fatter = wider BW)",
                        "Use cage dipole or fan dipole for wider BW",
                        "Biconical dipole for very wide bandwidth",
                    ],
                },
            ]

            parameter_sensitivity = {
                "total_length": "±1% length = ±1% frequency shift",
                "wire_radius": "Thicker wire = lower Q = wider bandwidth",
                "gap": "Minimal effect if << wavelength",
            }

        elif design_type == "helix":
            hints = [
                {
                    "category": "Gain",
                    "issue": f"Current gain: {design['calculated']['estimated_gain_dbi']:.1f} dBi with {design['parameters']['turns']} turns",
                    "suggestions": [
                        "Each additional turn adds ~1dB gain",
                        "Optimal circumference is 1.0-1.2λ",
                        "Pitch angle 12-14° is optimal for axial mode",
                    ],
                },
                {
                    "category": "Circular Polarization",
                    "issue": f"Axial ratio: {design['calculated']['axial_ratio_db']} dB",
                    "suggestions": [
                        "Ensure C/λ is close to 1.0 for best CP",
                        "Maintain consistent pitch throughout",
                        "Ground plane should be ≥0.75λ diameter",
                    ],
                },
                {
                    "category": "Impedance",
                    "issue": f"Characteristic impedance ~{design['calculated']['impedance_ohms']}Ω",
                    "suggestions": [
                        "Use tapered helix or matching section",
                        "Quarter-wave matching stub",
                        "Adjust first turn geometry for better match",
                    ],
                },
            ]

            parameter_sensitivity = {
                "turns": "+1 turn ≈ +1dB gain",
                "circumference": "±10% from λ degrades axial ratio",
                "pitch_angle": "12-14° optimal, outside range reduces gain",
            }

        else:
            hints = [{"category": "General", "suggestions": ["Run full EM simulation for accurate optimization"]}]
            parameter_sensitivity = {}

        return {
            "success": True,
            "design_id": design_id,
            "design_type": design_type,
            "optimization_hints": hints,
            "parameter_sensitivity": parameter_sensitivity,
            "next_steps": [
                "Generate OpenEMS script and run simulation",
                "Perform parameter sweep on sensitive parameters",
                "Compare simulated vs calculated values",
                "Iterate on design based on simulation results",
            ],
        }

    def compare_designs(self, design_ids: list[str]) -> dict:
        """Compare multiple antenna designs.

        Args:
            design_ids: List of design IDs to compare

        Returns:
            Comparison table and recommendations
        """
        designs = []
        for did in design_ids:
            if did not in self.designs:
                return {"success": False, "error": f"Design not found: {did}"}
            designs.append(self.designs[did])

        comparison = {
            "designs": [],
            "comparison_table": {},
        }

        # Build comparison data
        metrics = ["frequency_ghz", "type", "gain_dbi", "impedance_ohms", "polarization", "size_mm"]

        for design in designs:
            entry = {
                "id": design["id"],
                "name": design["name"],
                "frequency_ghz": design["frequency_ghz"],
                "type": design["type"],
            }

            calc = design.get("calculated", {})
            dims = design.get("dimensions", {})

            # Extract common metrics
            if "estimated_directivity_dbi" in calc:
                entry["gain_dbi"] = calc["estimated_directivity_dbi"]
            elif "estimated_gain_dbi" in calc:
                entry["gain_dbi"] = calc["estimated_gain_dbi"]
            elif "directivity_dbi" in calc:
                entry["gain_dbi"] = calc["directivity_dbi"]

            if "impedance_ohms" in calc:
                entry["impedance_ohms"] = calc["impedance_ohms"]
            elif "impedance_real_ohms" in calc:
                entry["impedance_ohms"] = calc["impedance_real_ohms"]

            entry["polarization"] = calc.get("polarization", "Linear")

            # Size metric (largest dimension)
            if "total_length_mm" in dims:
                entry["size_mm"] = dims["total_length_mm"]
            elif "ground_plane_mm" in dims:
                entry["size_mm"] = dims["ground_plane_mm"]
            elif "aperture_width_mm" in dims:
                entry["size_mm"] = dims["aperture_width_mm"]

            comparison["designs"].append(entry)

        # Generate recommendations
        recommendations = []
        if len(designs) >= 2:
            gains = [(d.get("gain_dbi", 0), d["name"]) for d in comparison["designs"]]
            gains.sort(reverse=True)
            recommendations.append(f"Highest gain: {gains[0][1]} ({gains[0][0]:.1f} dBi)")

            sizes = [(d.get("size_mm", 999), d["name"]) for d in comparison["designs"]]
            sizes.sort()
            recommendations.append(f"Most compact: {sizes[0][1]} ({sizes[0][0]:.1f}mm)")

            # Check for circular polarization
            cp_designs = [d["name"] for d in comparison["designs"] if "RHCP" in d.get("polarization", "") or "LHCP" in d.get("polarization", "")]
            if cp_designs:
                recommendations.append(f"Circular polarization: {', '.join(cp_designs)}")

        comparison["recommendations"] = recommendations

        return {"success": True, **comparison}

    def list_antenna_types(self) -> dict:
        """List available antenna types with descriptions."""
        return {
            "success": True,
            "antenna_types": [
                {
                    "type": "patch",
                    "name": "Microstrip Patch Antenna",
                    "description": "Rectangular patch on dielectric substrate",
                    "typical_gain_dbi": "6-9",
                    "polarization": "Linear",
                    "applications": ["WiFi", "GPS", "Satellite", "Mobile"],
                },
                {
                    "type": "dipole",
                    "name": "Half-Wave Dipole",
                    "description": "Classic resonant wire antenna",
                    "typical_gain_dbi": "2.15",
                    "polarization": "Linear",
                    "applications": ["HF/VHF/UHF comms", "Reference antenna"],
                },
                {
                    "type": "monopole",
                    "name": "Quarter-Wave Monopole",
                    "description": "Vertical antenna over ground plane",
                    "typical_gain_dbi": "5.15",
                    "polarization": "Vertical",
                    "applications": ["Mobile", "Vehicle", "Base station"],
                },
                {
                    "type": "horn",
                    "name": "Pyramidal Horn",
                    "description": "Waveguide-fed horn antenna",
                    "typical_gain_dbi": "10-25",
                    "polarization": "Linear",
                    "applications": ["Radar", "Satellite", "Microwave links"],
                },
                {
                    "type": "helix",
                    "name": "Axial-Mode Helix",
                    "description": "Helical antenna for circular polarization",
                    "typical_gain_dbi": "10-15",
                    "polarization": "Circular (RHCP/LHCP)",
                    "applications": ["Satellite", "GPS", "Space comms"],
                },
            ],
        }


class PCBStructureDesigner:
    """Calculates PCB interconnect structures and generates OpenEMS geometry."""

    def __init__(self):
        self.designs: dict[str, dict] = {}

    def create_microstrip_trace(
        self,
        frequency_ghz: float,
        trace_width_mm: float,
        trace_length_mm: float,
        dielectric_height_mm: float,
        substrate_er: float = 4.2,
        name: str = "Microstrip Trace",
    ) -> dict:
        """Design a microstrip trace with analytical impedance calculation.

        Uses Hammerstad-Jensen formula for characteristic impedance.

        Args:
            frequency_ghz: Analysis frequency in GHz
            trace_width_mm: Trace width in mm
            trace_length_mm: Trace length in mm
            dielectric_height_mm: Dielectric thickness in mm
            substrate_er: Substrate dielectric constant (default 4.2 for FR-4)
            name: Design name

        Returns:
            Design specification with calculated impedance and geometry
        """
        design_id = str(uuid4())
        c = 299792458  # Speed of light m/s
        f = frequency_ghz * 1e9
        w = trace_width_mm
        h = dielectric_height_mm
        er = substrate_er

        # Hammerstad-Jensen effective permittivity
        u = w / h
        a_ham = 1 + (1 / 49) * math.log(
            (u ** 4 + (u / 52) ** 2) / (u ** 4 + 0.432)
        ) + (1 / 18.7) * math.log(1 + (u / 18.1) ** 3)
        b_ham = 0.564 * ((er - 0.9) / (er + 3)) ** 0.053
        er_eff = (er + 1) / 2 + ((er - 1) / 2) * (1 + 10 / u) ** (-a_ham * b_ham)

        # Hammerstad characteristic impedance (free-space, then adjust for er_eff)
        f_u = 6 + (2 * math.pi - 6) * math.exp(-(30.666 / u) ** 0.7528)
        z0_air = (377 / (2 * math.pi)) * math.log(f_u / u + math.sqrt(1 + (2 / u) ** 2))
        z0 = z0_air / math.sqrt(er_eff)

        # Wavelength in medium
        wavelength_mm = (c / f) * 1000 / math.sqrt(er_eff)
        electrical_length_deg = 360 * trace_length_mm / wavelength_mm

        # Propagation delay
        delay_ps_per_mm = math.sqrt(er_eff) / (c / 1e9) * 1e3  # ps/mm

        # Simulation box sizing
        margin = max(10 * h, 5 * w, 5)  # Adequate margin for fringing fields
        box_x = trace_length_mm + 2 * margin
        box_y = trace_width_mm + 2 * margin
        box_z_above = max(10 * h, 10)
        ground_thickness = 0.035  # 1 oz copper

        design = {
            "id": design_id,
            "name": name,
            "type": "microstrip_trace",
            "frequency_ghz": frequency_ghz,
            "parameters": {
                "substrate_er": er,
                "dielectric_height_mm": h,
                "trace_width_mm": w,
                "trace_length_mm": trace_length_mm,
            },
            "dimensions": {
                "trace_width_mm": round(w, 4),
                "trace_length_mm": round(trace_length_mm, 3),
                "dielectric_height_mm": round(h, 4),
                "ground_plane_x_mm": round(box_x, 2),
                "ground_plane_y_mm": round(box_y, 2),
            },
            "calculated": {
                "z0_ohms": round(z0, 2),
                "effective_er": round(er_eff, 4),
                "wavelength_mm": round(wavelength_mm, 2),
                "electrical_length_deg": round(electrical_length_deg, 1),
                "delay_ps_per_mm": round(delay_ps_per_mm, 3),
            },
            "geometry": {
                "unit": "mm",
                "primitives": [
                    {
                        "type": "box",
                        "name": "ground_plane",
                        "material": "PEC",
                        "start": [-box_x / 2, -box_y / 2, -ground_thickness],
                        "stop": [box_x / 2, box_y / 2, 0],
                    },
                    {
                        "type": "box",
                        "name": "substrate",
                        "material": "substrate",
                        "start": [-box_x / 2, -box_y / 2, 0],
                        "stop": [box_x / 2, box_y / 2, h],
                    },
                    {
                        "type": "box",
                        "name": "trace",
                        "material": "PEC",
                        "start": [-trace_length_mm / 2, -w / 2, h],
                        "stop": [trace_length_mm / 2, w / 2, h],
                    },
                ],
                "ports": [
                    {
                        "type": "lumped",
                        "number": 1,
                        "position": [-trace_length_mm / 2, 0, h / 2],
                        "direction": "z",
                        "impedance": round(z0, 1),
                        "description": "near end",
                    },
                    {
                        "type": "lumped",
                        "number": 2,
                        "position": [trace_length_mm / 2, 0, h / 2],
                        "direction": "z",
                        "impedance": round(z0, 1),
                        "description": "far end",
                    },
                ],
            },
        }

        self.designs[design_id] = design
        return {"success": True, "design_id": design_id, "design": design}

    def create_coupled_lines(
        self,
        frequency_ghz: float,
        trace_width_mm: float,
        spacing_mm: float,
        trace_length_mm: float,
        dielectric_height_mm: float,
        substrate_er: float = 4.2,
        name: str = "Coupled Lines",
    ) -> dict:
        """Design edge-coupled microstrip lines with impedance analysis.

        Calculates even/odd mode impedances and differential/common impedances.

        Args:
            frequency_ghz: Analysis frequency in GHz
            trace_width_mm: Width of each trace in mm
            spacing_mm: Edge-to-edge spacing between traces in mm
            trace_length_mm: Coupled length in mm
            dielectric_height_mm: Dielectric thickness in mm
            substrate_er: Substrate dielectric constant (default 4.2)
            name: Design name

        Returns:
            Design specification with even/odd/differential/common impedances
        """
        design_id = str(uuid4())
        c = 299792458
        f = frequency_ghz * 1e9
        w = trace_width_mm
        s = spacing_mm
        h = dielectric_height_mm
        er = substrate_er

        # Single-line effective permittivity and Z0 (Hammerstad)
        u = w / h
        a_ham = 1 + (1 / 49) * math.log(
            (u ** 4 + (u / 52) ** 2) / (u ** 4 + 0.432)
        ) + (1 / 18.7) * math.log(1 + (u / 18.1) ** 3)
        b_ham = 0.564 * ((er - 0.9) / (er + 3)) ** 0.053
        er_eff = (er + 1) / 2 + ((er - 1) / 2) * (1 + 10 / u) ** (-a_ham * b_ham)
        f_u = 6 + (2 * math.pi - 6) * math.exp(-(30.666 / u) ** 0.7528)
        z0_single = (377 / (2 * math.pi)) * math.log(
            f_u / u + math.sqrt(1 + (2 / u) ** 2)
        ) / math.sqrt(er_eff)

        # Even/odd mode impedances using Kirschning-Jansen approximation
        # Coupling factor based on s/h ratio
        g = s / h
        coupling_factor = math.exp(-0.627 * g) if g < 10 else 0.0

        # Even mode: fields add constructively -> higher Z
        z_even = z0_single * (1 + coupling_factor * 0.65 * (1 - math.exp(-1.0 / g)))
        # Odd mode: fields cancel partially -> lower Z
        z_odd = z0_single * (1 - coupling_factor * 0.65 * (1 - math.exp(-1.0 / g)))

        # Even/odd effective permittivities (simplified)
        er_even = er_eff * (1 + 0.02 * coupling_factor)
        er_odd = er_eff * (1 - 0.02 * coupling_factor)

        # Differential and common-mode impedances
        z_diff = 2 * z_odd
        z_common = z_even / 2

        # Coupling coefficient
        k_coupling = (z_even - z_odd) / (z_even + z_odd)

        # Wavelengths
        wavelength_even_mm = (c / f) * 1000 / math.sqrt(er_even)
        wavelength_odd_mm = (c / f) * 1000 / math.sqrt(er_odd)

        # Simulation box
        margin = max(10 * h, 5 * w, 5)
        total_width = 2 * w + s
        box_x = trace_length_mm + 2 * margin
        box_y = total_width + 2 * margin
        box_z_above = max(10 * h, 10)
        ground_thickness = 0.035

        # Trace Y positions (centered)
        trace1_y_center = -(w + s) / 2
        trace2_y_center = (w + s) / 2

        design = {
            "id": design_id,
            "name": name,
            "type": "coupled_lines",
            "frequency_ghz": frequency_ghz,
            "parameters": {
                "substrate_er": er,
                "dielectric_height_mm": h,
                "trace_width_mm": w,
                "spacing_mm": s,
                "trace_length_mm": trace_length_mm,
            },
            "dimensions": {
                "trace_width_mm": round(w, 4),
                "spacing_mm": round(s, 4),
                "trace_length_mm": round(trace_length_mm, 3),
                "dielectric_height_mm": round(h, 4),
                "total_pair_width_mm": round(total_width, 4),
                "ground_plane_x_mm": round(box_x, 2),
                "ground_plane_y_mm": round(box_y, 2),
            },
            "calculated": {
                "z0_single_ohms": round(z0_single, 2),
                "z_even_ohms": round(z_even, 2),
                "z_odd_ohms": round(z_odd, 2),
                "z_diff_ohms": round(z_diff, 2),
                "z_common_ohms": round(z_common, 2),
                "coupling_coefficient": round(k_coupling, 4),
                "effective_er_even": round(er_even, 4),
                "effective_er_odd": round(er_odd, 4),
            },
            "geometry": {
                "unit": "mm",
                "primitives": [
                    {
                        "type": "box",
                        "name": "ground_plane",
                        "material": "PEC",
                        "start": [-box_x / 2, -box_y / 2, -ground_thickness],
                        "stop": [box_x / 2, box_y / 2, 0],
                    },
                    {
                        "type": "box",
                        "name": "substrate",
                        "material": "substrate",
                        "start": [-box_x / 2, -box_y / 2, 0],
                        "stop": [box_x / 2, box_y / 2, h],
                    },
                    {
                        "type": "box",
                        "name": "trace1",
                        "material": "PEC",
                        "start": [-trace_length_mm / 2, trace1_y_center - w / 2, h],
                        "stop": [trace_length_mm / 2, trace1_y_center + w / 2, h],
                    },
                    {
                        "type": "box",
                        "name": "trace2",
                        "material": "PEC",
                        "start": [-trace_length_mm / 2, trace2_y_center - w / 2, h],
                        "stop": [trace_length_mm / 2, trace2_y_center + w / 2, h],
                    },
                ],
                "ports": [
                    {
                        "type": "lumped",
                        "number": 1,
                        "position": [-trace_length_mm / 2, trace1_y_center, h / 2],
                        "direction": "z",
                        "impedance": round(z0_single, 1),
                        "description": "trace1 near end",
                    },
                    {
                        "type": "lumped",
                        "number": 2,
                        "position": [trace_length_mm / 2, trace1_y_center, h / 2],
                        "direction": "z",
                        "impedance": round(z0_single, 1),
                        "description": "trace1 far end",
                    },
                    {
                        "type": "lumped",
                        "number": 3,
                        "position": [-trace_length_mm / 2, trace2_y_center, h / 2],
                        "direction": "z",
                        "impedance": round(z0_single, 1),
                        "description": "trace2 near end",
                    },
                    {
                        "type": "lumped",
                        "number": 4,
                        "position": [trace_length_mm / 2, trace2_y_center, h / 2],
                        "direction": "z",
                        "impedance": round(z0_single, 1),
                        "description": "trace2 far end",
                    },
                ],
            },
        }

        self.designs[design_id] = design
        return {"success": True, "design_id": design_id, "design": design}

    def create_via_transition(
        self,
        frequency_ghz: float,
        drill_mm: float,
        pad_mm: float,
        dielectric_height_mm: float,
        substrate_er: float = 4.2,
        name: str = "Via Transition",
    ) -> dict:
        """Design a via transition between two layers.

        Models via barrel, pads, anti-pads, and feed traces on both layers.

        Args:
            frequency_ghz: Analysis frequency in GHz
            drill_mm: Via drill diameter in mm
            pad_mm: Via pad diameter in mm
            dielectric_height_mm: Dielectric thickness between layers in mm
            substrate_er: Substrate dielectric constant (default 4.2)
            name: Design name

        Returns:
            Design specification with estimated parasitics
        """
        design_id = str(uuid4())
        c = 299792458
        f = frequency_ghz * 1e9
        h = dielectric_height_mm
        er = substrate_er
        mu0 = 4 * math.pi * 1e-7
        eps0 = 8.854187817e-12

        # Via barrel inductance (simplified coaxial model)
        # L = (mu0 * h) / (2*pi) * ln(antipad/drill)
        # Assume antipad = pad + 0.25mm clearance
        antipad_mm = pad_mm + 0.25
        if antipad_mm > drill_mm:
            l_via = (mu0 * (h / 1000)) / (2 * math.pi) * math.log(antipad_mm / drill_mm)
        else:
            l_via = mu0 * (h / 1000) / (2 * math.pi) * 0.5  # fallback

        l_via_ph = l_via * 1e12  # Convert to pH

        # Via pad capacitance (parallel plate model for pad-to-plane)
        # C = eps0 * er * pi * (pad^2 - drill^2) / (4 * h)
        pad_area = math.pi * ((pad_mm / 2000) ** 2 - (drill_mm / 2000) ** 2)
        c_via = eps0 * er * pad_area / (h / 1000)
        c_via_ff = c_via * 1e15  # Convert to fF

        # Via impedance (coaxial approximation)
        z_via = (60 / math.sqrt(er)) * math.log(antipad_mm / drill_mm)

        # Resonance frequency of via
        f_res_ghz = 1 / (2 * math.pi * math.sqrt(l_via * c_via)) / 1e9

        # Feed trace width for ~50 ohm (rough approximation)
        feed_width_mm = max(drill_mm, 0.2)
        feed_length_mm = max(3 * pad_mm, 2)

        # Simulation box
        margin = max(10 * h, 5 * pad_mm, 5)
        box_xy = 2 * (feed_length_mm + pad_mm / 2 + margin)
        ground_thickness = 0.035

        design = {
            "id": design_id,
            "name": name,
            "type": "via_transition",
            "frequency_ghz": frequency_ghz,
            "parameters": {
                "substrate_er": er,
                "dielectric_height_mm": h,
                "drill_mm": drill_mm,
                "pad_mm": pad_mm,
            },
            "dimensions": {
                "drill_diameter_mm": round(drill_mm, 4),
                "pad_diameter_mm": round(pad_mm, 4),
                "antipad_diameter_mm": round(antipad_mm, 4),
                "dielectric_height_mm": round(h, 4),
                "feed_trace_width_mm": round(feed_width_mm, 3),
                "feed_trace_length_mm": round(feed_length_mm, 3),
            },
            "calculated": {
                "inductance_ph": round(l_via_ph, 1),
                "capacitance_ff": round(c_via_ff, 1),
                "z_via_ohms": round(z_via, 2),
                "resonance_ghz": round(f_res_ghz, 2),
            },
            "geometry": {
                "unit": "mm",
                "primitives": [
                    {
                        "type": "box",
                        "name": "ground_plane_bottom",
                        "material": "PEC",
                        "start": [-box_xy / 2, -box_xy / 2, -ground_thickness],
                        "stop": [box_xy / 2, box_xy / 2, 0],
                    },
                    {
                        "type": "box",
                        "name": "substrate",
                        "material": "substrate",
                        "start": [-box_xy / 2, -box_xy / 2, 0],
                        "stop": [box_xy / 2, box_xy / 2, h],
                    },
                    {
                        "type": "cylinder",
                        "name": "via_barrel",
                        "material": "PEC",
                        "start": [0, 0, 0],
                        "stop": [0, 0, h],
                        "radius": drill_mm / 2,
                    },
                    {
                        "type": "disc",
                        "name": "pad_bottom",
                        "material": "PEC",
                        "center": [0, 0, 0],
                        "radius": pad_mm / 2,
                    },
                    {
                        "type": "disc",
                        "name": "pad_top",
                        "material": "PEC",
                        "center": [0, 0, h],
                        "radius": pad_mm / 2,
                    },
                    {
                        "type": "box",
                        "name": "feed_trace_bottom",
                        "material": "PEC",
                        "start": [-pad_mm / 2 - feed_length_mm, -feed_width_mm / 2, 0],
                        "stop": [-pad_mm / 2, feed_width_mm / 2, 0],
                    },
                    {
                        "type": "box",
                        "name": "feed_trace_top",
                        "material": "PEC",
                        "start": [pad_mm / 2, -feed_width_mm / 2, h],
                        "stop": [pad_mm / 2 + feed_length_mm, feed_width_mm / 2, h],
                    },
                ],
                "ports": [
                    {
                        "type": "lumped",
                        "number": 1,
                        "position": [
                            -pad_mm / 2 - feed_length_mm,
                            0,
                            ground_thickness / 2,
                        ],
                        "direction": "z",
                        "impedance": 50,
                        "description": "bottom layer feed",
                    },
                    {
                        "type": "lumped",
                        "number": 2,
                        "position": [
                            pad_mm / 2 + feed_length_mm,
                            0,
                            h - ground_thickness / 2,
                        ],
                        "direction": "z",
                        "impedance": 50,
                        "description": "top layer feed",
                    },
                ],
            },
        }

        self.designs[design_id] = design
        return {"success": True, "design_id": design_id, "design": design}

    def generate_openems_script(self, design_id: str) -> dict:
        """Generate Python script for OpenEMS simulation of a PCB structure.

        Args:
            design_id: ID of the design

        Returns:
            OpenEMS Python script as string
        """
        if design_id not in self.designs:
            return {"success": False, "error": f"Design not found: {design_id}"}

        design = self.designs[design_id]
        design_type = design["type"]

        if design_type == "microstrip_trace":
            script = self._generate_microstrip_trace_script(design)
        elif design_type == "coupled_lines":
            script = self._generate_coupled_lines_script(design)
        elif design_type == "via_transition":
            script = self._generate_via_transition_script(design)
        else:
            return {"success": False, "error": f"Unknown PCB structure type: {design_type}"}

        return {
            "success": True,
            "design_id": design_id,
            "script_type": "openems_python",
            "script": script,
            "instructions": [
                "1. Save this script as simulate_pcb_structure.py",
                "2. Install OpenEMS: pip install CSXCAD openEMS",
                "3. Run: python simulate_pcb_structure.py",
                "4. View S-parameter results in generated plots",
            ],
        }

    def _generate_microstrip_trace_script(self, design: dict) -> str:
        """Generate OpenEMS script for microstrip trace S-parameter extraction."""
        dims = design["dimensions"]
        params = design["parameters"]
        calc = design["calculated"]

        return f'''#!/usr/bin/env python3
"""OpenEMS Microstrip Trace Simulation
Design: {design["name"]}
Frequency: {design["frequency_ghz"]} GHz
Calculated Z0: {calc["z0_ohms"]} ohms
Generated by mcp-openems
"""

import os
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

# Design parameters
f0 = {design["frequency_ghz"]}e9  # Center frequency
fc = f0 * 0.5  # 20 dB bandwidth
substrate_er = {params["substrate_er"]}
substrate_h = {params["dielectric_height_mm"]}  # mm
trace_w = {params["trace_width_mm"]}  # mm
trace_l = {params["trace_length_mm"]}  # mm
z0_calc = {calc["z0_ohms"]}  # Calculated Z0

# Simulation setup
unit = 1e-3  # mm
FDTD = openEMS(EndCriteria=1e-5)
FDTD.SetGaussExcite(f0, fc)
FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8'])

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)

# Materials
substrate = CSX.AddMaterial('substrate', epsilon=substrate_er)
metal = CSX.AddMetal('PEC')

# Geometry dimensions
ground_x = {dims["ground_plane_x_mm"]}
ground_y = {dims["ground_plane_y_mm"]}

# Ground plane
metal.AddBox(
    start=[-ground_x/2, -ground_y/2, -0.035],
    stop=[ground_x/2, ground_y/2, 0],
    priority=10
)

# Substrate
substrate.AddBox(
    start=[-ground_x/2, -ground_y/2, 0],
    stop=[ground_x/2, ground_y/2, substrate_h],
    priority=1
)

# Microstrip trace
metal.AddBox(
    start=[-trace_l/2, -trace_w/2, substrate_h],
    stop=[trace_l/2, trace_w/2, substrate_h],
    priority=10
)

# Port 1 (near end) - excitation port
port1 = FDTD.AddLumpedPort(
    1, z0_calc,
    start=[-trace_l/2, -trace_w/2, 0],
    stop=[-trace_l/2, trace_w/2, substrate_h],
    p_dir='z',
    excite=1
)

# Port 2 (far end) - measurement port
port2 = FDTD.AddLumpedPort(
    2, z0_calc,
    start=[trace_l/2, -trace_w/2, 0],
    stop=[trace_l/2, trace_w/2, substrate_h],
    p_dir='z',
    excite=0
)

# Mesh
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)

resolution = C0 / (f0 + fc) / unit / 20
trace_res = min(trace_w / 4, substrate_h / 4, resolution)

mesh.AddLine('x', np.concatenate([
    np.arange(-ground_x/2 - 10, -trace_l/2, resolution),
    np.linspace(-trace_l/2, trace_l/2, max(20, int(trace_l / trace_res))),
    np.arange(trace_l/2, ground_x/2 + 10, resolution)
]))
mesh.AddLine('y', np.concatenate([
    np.arange(-ground_y/2 - 10, -trace_w/2 - 2*substrate_h, resolution),
    np.linspace(-trace_w/2 - 2*substrate_h, trace_w/2 + 2*substrate_h,
                max(15, int((trace_w + 4*substrate_h) / trace_res))),
    np.arange(trace_w/2 + 2*substrate_h, ground_y/2 + 10, resolution)
]))
mesh.AddLine('z', np.concatenate([
    np.arange(-10, 0, resolution),
    np.linspace(0, substrate_h, max(5, int(substrate_h / (substrate_h/4)) + 1)),
    np.arange(substrate_h, 10*substrate_h + 10, resolution)
]))

# Run simulation
import tempfile
Sim_Path = os.path.join(tempfile.gettempdir(), 'openems_microstrip')
if os.path.exists(Sim_Path):
    import shutil
    shutil.rmtree(Sim_Path)
os.makedirs(Sim_Path)

FDTD.Run(Sim_Path, verbose=3, cleanup=True)

# Post-processing: S-parameters
f = np.linspace(max(1e6, f0 - fc), f0 + fc, 401)
port1.CalcPort(Sim_Path, f)
port2.CalcPort(Sim_Path, f)

s11 = port1.uf_ref / port1.uf_inc
s21 = port2.uf_ref / port1.uf_inc
s11_dB = 20 * np.log10(np.abs(s11))
s21_dB = 20 * np.log10(np.abs(s21))

# Find minimum S11 (best match)
idx = np.argmin(s11_dB)
print(f"Best match at: {{f[idx]/1e9:.3f}} GHz")
print(f"S11 = {{s11_dB[idx]:.1f}} dB")
print(f"S21 at center = {{s21_dB[len(f)//2]:.2f}} dB")
print(f"Calculated Z0 = {calc["z0_ohms"]} ohms")

# Plot results
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(f/1e9, s11_dB, 'b-', label='S11')
ax1.plot(f/1e9, s21_dB, 'r-', label='S21')
ax1.set_xlabel('Frequency (GHz)')
ax1.set_ylabel('Magnitude (dB)')
ax1.set_title('Microstrip Trace S-Parameters')
ax1.grid(True)
ax1.legend()
ax1.set_ylim([-40, 5])

ax2.plot(f/1e9, np.angle(s21, deg=True), 'r-')
ax2.set_xlabel('Frequency (GHz)')
ax2.set_ylabel('Phase S21 (deg)')
ax2.set_title('Transmission Phase')
ax2.grid(True)

plt.tight_layout()
plt.savefig('microstrip_s_params.png', dpi=150)
print("S-parameter plot saved to microstrip_s_params.png")
'''

    def _generate_coupled_lines_script(self, design: dict) -> str:
        """Generate OpenEMS script for coupled lines with 4-port S-parameter extraction."""
        dims = design["dimensions"]
        params = design["parameters"]
        calc = design["calculated"]

        trace1_y = -(params["trace_width_mm"] + params["spacing_mm"]) / 2
        trace2_y = (params["trace_width_mm"] + params["spacing_mm"]) / 2

        return f'''#!/usr/bin/env python3
"""OpenEMS Coupled Lines Simulation
Design: {design["name"]}
Frequency: {design["frequency_ghz"]} GHz
Zdiff: {calc["z_diff_ohms"]} ohms, Zcommon: {calc["z_common_ohms"]} ohms
Generated by mcp-openems
"""

import os
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

# Design parameters
f0 = {design["frequency_ghz"]}e9
fc = f0 * 0.5
substrate_er = {params["substrate_er"]}
substrate_h = {params["dielectric_height_mm"]}
trace_w = {params["trace_width_mm"]}
spacing = {params["spacing_mm"]}
trace_l = {params["trace_length_mm"]}
z0_single = {calc["z0_single_ohms"]}

# Trace Y-center positions
trace1_y = {trace1_y}
trace2_y = {trace2_y}

# Simulation setup
unit = 1e-3
FDTD = openEMS(EndCriteria=1e-5)
FDTD.SetGaussExcite(f0, fc)
FDTD.SetBoundaryCond(['PML_8'] * 6)

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)

substrate = CSX.AddMaterial('substrate', epsilon=substrate_er)
metal = CSX.AddMetal('PEC')

ground_x = {dims["ground_plane_x_mm"]}
ground_y = {dims["ground_plane_y_mm"]}

# Ground plane
metal.AddBox(
    start=[-ground_x/2, -ground_y/2, -0.035],
    stop=[ground_x/2, ground_y/2, 0],
    priority=10
)

# Substrate
substrate.AddBox(
    start=[-ground_x/2, -ground_y/2, 0],
    stop=[ground_x/2, ground_y/2, substrate_h],
    priority=1
)

# Trace 1
metal.AddBox(
    start=[-trace_l/2, trace1_y - trace_w/2, substrate_h],
    stop=[trace_l/2, trace1_y + trace_w/2, substrate_h],
    priority=10
)

# Trace 2
metal.AddBox(
    start=[-trace_l/2, trace2_y - trace_w/2, substrate_h],
    stop=[trace_l/2, trace2_y + trace_w/2, substrate_h],
    priority=10
)

# Port 1: Trace 1 near end
port1 = FDTD.AddLumpedPort(
    1, z0_single,
    start=[-trace_l/2, trace1_y - trace_w/2, 0],
    stop=[-trace_l/2, trace1_y + trace_w/2, substrate_h],
    p_dir='z', excite=1
)

# Port 2: Trace 1 far end
port2 = FDTD.AddLumpedPort(
    2, z0_single,
    start=[trace_l/2, trace1_y - trace_w/2, 0],
    stop=[trace_l/2, trace1_y + trace_w/2, substrate_h],
    p_dir='z', excite=0
)

# Port 3: Trace 2 near end
port3 = FDTD.AddLumpedPort(
    3, z0_single,
    start=[-trace_l/2, trace2_y - trace_w/2, 0],
    stop=[-trace_l/2, trace2_y + trace_w/2, substrate_h],
    p_dir='z', excite=0
)

# Port 4: Trace 2 far end
port4 = FDTD.AddLumpedPort(
    4, z0_single,
    start=[trace_l/2, trace2_y - trace_w/2, 0],
    stop=[trace_l/2, trace2_y + trace_w/2, substrate_h],
    p_dir='z', excite=0
)

# Mesh
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)

resolution = C0 / (f0 + fc) / unit / 20
fine_res = min(trace_w / 4, spacing / 3, substrate_h / 4, resolution)

# X mesh: along trace length
mesh.AddLine('x', np.concatenate([
    np.arange(-ground_x/2 - 10, -trace_l/2, resolution),
    np.linspace(-trace_l/2, trace_l/2, max(20, int(trace_l / fine_res))),
    np.arange(trace_l/2, ground_x/2 + 10, resolution)
]))

# Y mesh: across traces with fine resolution in coupling region
pair_half = (trace_w + spacing/2 + 2*substrate_h)
mesh.AddLine('y', np.concatenate([
    np.arange(-ground_y/2 - 10, -pair_half, resolution),
    np.linspace(-pair_half, pair_half,
                max(25, int(2*pair_half / fine_res))),
    np.arange(pair_half, ground_y/2 + 10, resolution)
]))

# Z mesh
mesh.AddLine('z', np.concatenate([
    np.arange(-10, 0, resolution),
    np.linspace(0, substrate_h, max(5, int(substrate_h / (substrate_h/4)) + 1)),
    np.arange(substrate_h, 10*substrate_h + 10, resolution)
]))

# Run simulation
import tempfile
Sim_Path = os.path.join(tempfile.gettempdir(), 'openems_coupled_lines')
if os.path.exists(Sim_Path):
    import shutil
    shutil.rmtree(Sim_Path)
os.makedirs(Sim_Path)

FDTD.Run(Sim_Path, verbose=3, cleanup=True)

# Post-processing: 4-port S-parameters
f = np.linspace(max(1e6, f0 - fc), f0 + fc, 401)
port1.CalcPort(Sim_Path, f)
port2.CalcPort(Sim_Path, f)
port3.CalcPort(Sim_Path, f)
port4.CalcPort(Sim_Path, f)

# Single-ended S-parameters (excitation on port 1)
s11 = port1.uf_ref / port1.uf_inc
s21 = port2.uf_ref / port1.uf_inc  # Through
s31 = port3.uf_ref / port1.uf_inc  # Near-end crosstalk (NEXT)
s41 = port4.uf_ref / port1.uf_inc  # Far-end crosstalk (FEXT)

s11_dB = 20 * np.log10(np.abs(s11))
s21_dB = 20 * np.log10(np.abs(s21))
s31_dB = 20 * np.log10(np.abs(s31))
s41_dB = 20 * np.log10(np.abs(s41))

# Mixed-mode S-parameters (approximate from single-ended)
# Sdd11 = (S11 - S13 - S31 + S33) / 2  (requires full 4-port excitation)
# For single excitation, report NEXT/FEXT coupling
print(f"=== Coupled Lines Results ===")
print(f"Calculated Zdiff = {calc["z_diff_ohms"]} ohms")
print(f"Calculated Zcommon = {calc["z_common_ohms"]} ohms")
print(f"Coupling coefficient k = {calc["coupling_coefficient"]}")
print(f"")
print(f"At {{f0/1e9:.2f}} GHz:")
mid = len(f) // 2
print(f"  S11 (return loss) = {{s11_dB[mid]:.1f}} dB")
print(f"  S21 (insertion loss) = {{s21_dB[mid]:.2f}} dB")
print(f"  S31 (NEXT) = {{s31_dB[mid]:.1f}} dB")
print(f"  S41 (FEXT) = {{s41_dB[mid]:.1f}} dB")

# Plot
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(f/1e9, s11_dB, 'b-')
axes[0, 0].set_title('S11 (Return Loss)')
axes[0, 0].set_ylabel('dB')
axes[0, 0].grid(True)
axes[0, 0].set_ylim([-40, 5])

axes[0, 1].plot(f/1e9, s21_dB, 'r-')
axes[0, 1].set_title('S21 (Insertion Loss)')
axes[0, 1].set_ylabel('dB')
axes[0, 1].grid(True)

axes[1, 0].plot(f/1e9, s31_dB, 'g-')
axes[1, 0].set_title('S31 (Near-End Crosstalk)')
axes[1, 0].set_xlabel('Frequency (GHz)')
axes[1, 0].set_ylabel('dB')
axes[1, 0].grid(True)

axes[1, 1].plot(f/1e9, s41_dB, 'm-')
axes[1, 1].set_title('S41 (Far-End Crosstalk)')
axes[1, 1].set_xlabel('Frequency (GHz)')
axes[1, 1].set_ylabel('dB')
axes[1, 1].grid(True)

plt.suptitle('Coupled Lines 4-Port S-Parameters')
plt.tight_layout()
plt.savefig('coupled_lines_s_params.png', dpi=150)
print("S-parameter plot saved to coupled_lines_s_params.png")
'''

    def _generate_via_transition_script(self, design: dict) -> str:
        """Generate OpenEMS script for via transition simulation."""
        dims = design["dimensions"]
        params = design["parameters"]
        calc = design["calculated"]
        geom = design["geometry"]

        # Extract feed positions from geometry ports
        port1_x = geom["ports"][0]["position"][0]
        port2_x = geom["ports"][1]["position"][0]

        return f'''#!/usr/bin/env python3
"""OpenEMS Via Transition Simulation
Design: {design["name"]}
Frequency: {design["frequency_ghz"]} GHz
Via: drill={params["drill_mm"]}mm, pad={params["pad_mm"]}mm
Estimated L={calc["inductance_ph"]} pH, C={calc["capacitance_ff"]} fF
Generated by mcp-openems
"""

import os
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

# Design parameters
f0 = {design["frequency_ghz"]}e9
fc = f0 * 0.5
substrate_er = {params["substrate_er"]}
substrate_h = {params["dielectric_height_mm"]}
drill_r = {params["drill_mm"]} / 2  # Via drill radius
pad_r = {params["pad_mm"]} / 2  # Pad radius
antipad_r = {dims["antipad_diameter_mm"]} / 2  # Antipad radius
feed_w = {dims["feed_trace_width_mm"]}
feed_l = {dims["feed_trace_length_mm"]}

# Simulation setup
unit = 1e-3
FDTD = openEMS(EndCriteria=1e-5)
FDTD.SetGaussExcite(f0, fc)
FDTD.SetBoundaryCond(['PML_8'] * 6)

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)

substrate = CSX.AddMaterial('substrate', epsilon=substrate_er)
metal = CSX.AddMetal('PEC')

# Ground/reference planes with antipad cutout
box_xy = {round(max(10 * params["dielectric_height_mm"], 5 * params["pad_mm"], 5) * 2 + 2 * (dims["feed_trace_length_mm"] + params["pad_mm"] / 2), 2)}

# Bottom reference plane (with clearance hole for via)
metal.AddBox(
    start=[-box_xy/2, -box_xy/2, -0.035],
    stop=[box_xy/2, box_xy/2, 0],
    priority=5
)

# Substrate
substrate.AddBox(
    start=[-box_xy/2, -box_xy/2, 0],
    stop=[box_xy/2, box_xy/2, substrate_h],
    priority=1
)

# Via barrel (cylinder through substrate)
n_cyl = 16  # Polygon approximation segments
theta = np.linspace(0, 2*np.pi, n_cyl+1)[:-1]

# Via barrel as cylinder
metal.AddCylinder(
    start=[0, 0, 0],
    stop=[0, 0, substrate_h],
    radius=drill_r,
    priority=15
)

# Bottom pad (annular ring on bottom layer)
metal.AddCylinder(
    start=[0, 0, -0.035],
    stop=[0, 0, 0],
    radius=pad_r,
    priority=15
)

# Top pad (annular ring on top layer)
metal.AddCylinder(
    start=[0, 0, substrate_h],
    stop=[0, 0, substrate_h + 0.035],
    radius=pad_r,
    priority=15
)

# Feed trace on bottom layer (extending in -x direction)
metal.AddBox(
    start=[-pad_r - feed_l, -feed_w/2, -0.035],
    stop=[-pad_r, feed_w/2, 0],
    priority=10
)

# Feed trace on top layer (extending in +x direction)
metal.AddBox(
    start=[pad_r, -feed_w/2, substrate_h],
    stop=[pad_r + feed_l, feed_w/2, substrate_h + 0.035],
    priority=10
)

# Port 1: Bottom layer feed
port1 = FDTD.AddLumpedPort(
    1, 50,
    start=[-pad_r - feed_l, -feed_w/2, -0.035],
    stop=[-pad_r - feed_l, feed_w/2, 0],
    p_dir='z',
    excite=1
)

# Port 2: Top layer feed
port2 = FDTD.AddLumpedPort(
    2, 50,
    start=[pad_r + feed_l, -feed_w/2, substrate_h],
    stop=[pad_r + feed_l, feed_w/2, substrate_h + 0.035],
    p_dir='z',
    excite=0
)

# Mesh
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)

resolution = C0 / (f0 + fc) / unit / 20
via_res = min(drill_r, substrate_h / 4, resolution / 2)

# Fine mesh around via
via_region = max(pad_r, antipad_r) * 2
mesh.AddLine('x', np.concatenate([
    np.arange(-box_xy/2 - 10, -via_region, resolution),
    np.linspace(-via_region, via_region, max(20, int(2*via_region / via_res))),
    np.arange(via_region, box_xy/2 + 10, resolution)
]))
mesh.AddLine('y', np.concatenate([
    np.arange(-box_xy/2 - 10, -via_region, resolution),
    np.linspace(-via_region, via_region, max(20, int(2*via_region / via_res))),
    np.arange(via_region, box_xy/2 + 10, resolution)
]))
mesh.AddLine('z', np.concatenate([
    np.arange(-10, -0.035, resolution),
    np.linspace(-0.035, substrate_h + 0.035,
                max(8, int((substrate_h + 0.07) / (substrate_h/6)) + 1)),
    np.arange(substrate_h + 0.035, 10*substrate_h + 10, resolution)
]))

# Run simulation
import tempfile
Sim_Path = os.path.join(tempfile.gettempdir(), 'openems_via')
if os.path.exists(Sim_Path):
    import shutil
    shutil.rmtree(Sim_Path)
os.makedirs(Sim_Path)

FDTD.Run(Sim_Path, verbose=3, cleanup=True)

# Post-processing
f = np.linspace(max(1e6, f0 - fc), f0 + fc, 401)
port1.CalcPort(Sim_Path, f)
port2.CalcPort(Sim_Path, f)

s11 = port1.uf_ref / port1.uf_inc
s21 = port2.uf_ref / port1.uf_inc
s11_dB = 20 * np.log10(np.abs(s11))
s21_dB = 20 * np.log10(np.abs(s21))

# Extract via equivalent circuit parameters from S-parameters
# Z_via approx from S11: Z = Z0 * (1+S11)/(1-S11)
mid = len(f) // 2
z_via_sim = 50 * (1 + s11[mid]) / (1 - s11[mid])

print(f"=== Via Transition Results ===")
print(f"Analytical estimates: L = {calc["inductance_ph"]} pH, C = {calc["capacitance_ff"]} fF")
print(f"Resonance estimate: {calc["resonance_ghz"]} GHz")
print(f"")
print(f"At {{f0/1e9:.2f}} GHz:")
print(f"  S11 = {{s11_dB[mid]:.1f}} dB")
print(f"  S21 = {{s21_dB[mid]:.2f}} dB")
print(f"  Via impedance (from S11) = {{np.abs(z_via_sim):.1f}} ohms")

# Plot
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(f/1e9, s11_dB, 'b-', label='S11 (Reflection)')
ax1.plot(f/1e9, s21_dB, 'r-', label='S21 (Transmission)')
ax1.set_xlabel('Frequency (GHz)')
ax1.set_ylabel('Magnitude (dB)')
ax1.set_title('Via Transition S-Parameters')
ax1.grid(True)
ax1.legend()
ax1.set_ylim([-40, 5])

# Smith chart data
ax2.plot(f/1e9, np.real(z_via_sim * np.ones_like(f)), 'b-', label='Re(Z)')
ax2.plot(f/1e9, np.imag(50 * (1 + s11) / (1 - s11)), 'r-', label='Im(Z)')
ax2.set_xlabel('Frequency (GHz)')
ax2.set_ylabel('Impedance (ohms)')
ax2.set_title('Via Impedance vs Frequency')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig('via_transition_s_params.png', dpi=150)
print("S-parameter plot saved to via_transition_s_params.png")
'''

    def list_designs(self) -> dict:
        """List all created PCB structure designs."""
        designs = []
        for design_id, design in self.designs.items():
            designs.append({
                "id": design_id,
                "name": design["name"],
                "type": design["type"],
                "frequency_ghz": design["frequency_ghz"],
            })
        return {"success": True, "designs": designs, "count": len(designs)}

    def get_design(self, design_id: str) -> dict:
        """Get full details of a PCB structure design."""
        if design_id not in self.designs:
            return {"success": False, "error": f"Design not found: {design_id}"}
        return {"success": True, "design": self.designs[design_id]}


# Global designer instances
designer = AntennaDesigner()
pcb_designer = PCBStructureDesigner()


# MCP Tool definitions
TOOLS = [
    Tool(
        name="openems_create_patch",
        description="Design a rectangular microstrip patch antenna. Calculates dimensions using transmission line model for specified frequency and substrate.",
        inputSchema={
            "type": "object",
            "properties": {
                "frequency_ghz": {
                    "type": "number",
                    "description": "Center frequency in GHz (e.g., 2.4 for WiFi)",
                },
                "substrate_er": {
                    "type": "number",
                    "description": "Substrate dielectric constant (default 4.4 for FR-4)",
                    "default": 4.4,
                },
                "substrate_height_mm": {
                    "type": "number",
                    "description": "Substrate thickness in mm (default 1.6)",
                    "default": 1.6,
                },
                "name": {
                    "type": "string",
                    "description": "Design name",
                },
            },
            "required": ["frequency_ghz"],
        },
    ),
    Tool(
        name="openems_create_dipole",
        description="Design a half-wave dipole antenna. Classic resonant antenna with well-known characteristics.",
        inputSchema={
            "type": "object",
            "properties": {
                "frequency_ghz": {
                    "type": "number",
                    "description": "Center frequency in GHz",
                },
                "wire_radius_mm": {
                    "type": "number",
                    "description": "Wire radius in mm (default 1.0)",
                    "default": 1.0,
                },
                "name": {
                    "type": "string",
                    "description": "Design name",
                },
            },
            "required": ["frequency_ghz"],
        },
    ),
    Tool(
        name="openems_create_monopole",
        description="Design a quarter-wave monopole antenna over a ground plane.",
        inputSchema={
            "type": "object",
            "properties": {
                "frequency_ghz": {
                    "type": "number",
                    "description": "Center frequency in GHz",
                },
                "ground_plane_mm": {
                    "type": "number",
                    "description": "Ground plane diameter in mm",
                    "default": 100,
                },
                "wire_radius_mm": {
                    "type": "number",
                    "description": "Wire radius in mm",
                    "default": 1.0,
                },
                "name": {
                    "type": "string",
                    "description": "Design name",
                },
            },
            "required": ["frequency_ghz"],
        },
    ),
    Tool(
        name="openems_create_horn",
        description="Design a pyramidal horn antenna for specified gain. Calculates aperture dimensions and length.",
        inputSchema={
            "type": "object",
            "properties": {
                "frequency_ghz": {
                    "type": "number",
                    "description": "Center frequency in GHz",
                },
                "gain_dbi": {
                    "type": "number",
                    "description": "Target gain in dBi (typically 10-25)",
                    "default": 15,
                },
                "name": {
                    "type": "string",
                    "description": "Design name",
                },
            },
            "required": ["frequency_ghz"],
        },
    ),
    Tool(
        name="openems_create_helix",
        description="Design an axial-mode helical antenna for circular polarization.",
        inputSchema={
            "type": "object",
            "properties": {
                "frequency_ghz": {
                    "type": "number",
                    "description": "Center frequency in GHz",
                },
                "turns": {
                    "type": "integer",
                    "description": "Number of turns (more turns = higher gain)",
                    "default": 10,
                },
                "name": {
                    "type": "string",
                    "description": "Design name",
                },
            },
            "required": ["frequency_ghz"],
        },
    ),
    Tool(
        name="openems_generate_script",
        description="Generate a complete OpenEMS Python simulation script for a design. The script can be run independently.",
        inputSchema={
            "type": "object",
            "properties": {
                "design_id": {
                    "type": "string",
                    "description": "ID of the design to generate script for",
                },
            },
            "required": ["design_id"],
        },
    ),
    Tool(
        name="openems_list_designs",
        description="List all antenna designs created in this session.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="openems_get_design",
        description="Get full details of a specific antenna design including dimensions and geometry.",
        inputSchema={
            "type": "object",
            "properties": {
                "design_id": {
                    "type": "string",
                    "description": "ID of the design",
                },
            },
            "required": ["design_id"],
        },
    ),
    Tool(
        name="openems_list_antenna_types",
        description="List available antenna types with descriptions, typical gain, and applications.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="openems_check_installation",
        description="Check if OpenEMS is installed and available for running simulations.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="openems_export_design",
        description="Export antenna design for visualization. Formats: json (full data), svg (vector graphic), ascii (terminal art).",
        inputSchema={
            "type": "object",
            "properties": {
                "design_id": {
                    "type": "string",
                    "description": "ID of the design to export",
                },
                "format": {
                    "type": "string",
                    "description": "Export format: json, svg, or ascii",
                    "enum": ["json", "svg", "ascii"],
                    "default": "ascii",
                },
            },
            "required": ["design_id"],
        },
    ),
    Tool(
        name="openems_optimize_hints",
        description="Get optimization suggestions for an antenna design including parameter sensitivities and improvement strategies.",
        inputSchema={
            "type": "object",
            "properties": {
                "design_id": {
                    "type": "string",
                    "description": "ID of the design to analyze",
                },
            },
            "required": ["design_id"],
        },
    ),
    Tool(
        name="openems_compare_designs",
        description="Compare multiple antenna designs side-by-side with metrics and recommendations.",
        inputSchema={
            "type": "object",
            "properties": {
                "design_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of design IDs to compare",
                },
            },
            "required": ["design_ids"],
        },
    ),
    # PCB Structure Design Tools
    Tool(
        name="openems_create_microstrip",
        description="Design a microstrip trace and calculate characteristic impedance using Hammerstad formula. Generates 2-port geometry for S-parameter extraction.",
        inputSchema={
            "type": "object",
            "properties": {
                "frequency_ghz": {
                    "type": "number",
                    "description": "Analysis frequency in GHz",
                },
                "trace_width_mm": {
                    "type": "number",
                    "description": "Trace width in mm",
                },
                "trace_length_mm": {
                    "type": "number",
                    "description": "Trace length in mm",
                },
                "dielectric_height_mm": {
                    "type": "number",
                    "description": "Dielectric (substrate) thickness in mm",
                },
                "substrate_er": {
                    "type": "number",
                    "description": "Substrate dielectric constant (default 4.2 for FR-4)",
                    "default": 4.2,
                },
                "name": {
                    "type": "string",
                    "description": "Design name",
                },
            },
            "required": ["frequency_ghz", "trace_width_mm", "trace_length_mm", "dielectric_height_mm"],
        },
    ),
    Tool(
        name="openems_create_coupled_lines",
        description="Design edge-coupled microstrip lines. Calculates even/odd mode impedances, differential/common-mode impedances, and coupling coefficient. 4-port geometry for NEXT/FEXT analysis.",
        inputSchema={
            "type": "object",
            "properties": {
                "frequency_ghz": {
                    "type": "number",
                    "description": "Analysis frequency in GHz",
                },
                "trace_width_mm": {
                    "type": "number",
                    "description": "Width of each trace in mm",
                },
                "spacing_mm": {
                    "type": "number",
                    "description": "Edge-to-edge spacing between traces in mm",
                },
                "trace_length_mm": {
                    "type": "number",
                    "description": "Coupled length in mm",
                },
                "dielectric_height_mm": {
                    "type": "number",
                    "description": "Dielectric thickness in mm",
                },
                "substrate_er": {
                    "type": "number",
                    "description": "Substrate dielectric constant (default 4.2 for FR-4)",
                    "default": 4.2,
                },
                "name": {
                    "type": "string",
                    "description": "Design name",
                },
            },
            "required": ["frequency_ghz", "trace_width_mm", "spacing_mm", "trace_length_mm", "dielectric_height_mm"],
        },
    ),
    Tool(
        name="openems_create_via",
        description="Design a via transition between two PCB layers. Models via barrel, pads, and feed traces. Estimates parasitic inductance and capacitance. 2-port geometry for reflection/transmission analysis.",
        inputSchema={
            "type": "object",
            "properties": {
                "frequency_ghz": {
                    "type": "number",
                    "description": "Analysis frequency in GHz",
                },
                "drill_mm": {
                    "type": "number",
                    "description": "Via drill diameter in mm",
                },
                "pad_mm": {
                    "type": "number",
                    "description": "Via pad diameter in mm",
                },
                "dielectric_height_mm": {
                    "type": "number",
                    "description": "Dielectric thickness between layers in mm",
                },
                "substrate_er": {
                    "type": "number",
                    "description": "Substrate dielectric constant (default 4.2)",
                    "default": 4.2,
                },
                "name": {
                    "type": "string",
                    "description": "Design name",
                },
            },
            "required": ["frequency_ghz", "drill_mm", "pad_mm", "dielectric_height_mm"],
        },
    ),
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return list of available tools."""
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute an OpenEMS design tool."""
    try:
        if name == "openems_create_patch":
            result = designer.create_patch_antenna(
                frequency_ghz=arguments["frequency_ghz"],
                substrate_er=arguments.get("substrate_er", 4.4),
                substrate_height_mm=arguments.get("substrate_height_mm", 1.6),
                name=arguments.get("name", "Patch Antenna"),
            )
        elif name == "openems_create_dipole":
            result = designer.create_dipole_antenna(
                frequency_ghz=arguments["frequency_ghz"],
                wire_radius_mm=arguments.get("wire_radius_mm", 1.0),
                name=arguments.get("name", "Dipole Antenna"),
            )
        elif name == "openems_create_monopole":
            result = designer.create_monopole_antenna(
                frequency_ghz=arguments["frequency_ghz"],
                ground_plane_mm=arguments.get("ground_plane_mm", 100),
                wire_radius_mm=arguments.get("wire_radius_mm", 1.0),
                name=arguments.get("name", "Monopole Antenna"),
            )
        elif name == "openems_create_horn":
            result = designer.create_horn_antenna(
                frequency_ghz=arguments["frequency_ghz"],
                gain_dbi=arguments.get("gain_dbi", 15),
                name=arguments.get("name", "Horn Antenna"),
            )
        elif name == "openems_create_helix":
            result = designer.create_helix_antenna(
                frequency_ghz=arguments["frequency_ghz"],
                turns=arguments.get("turns", 10),
                name=arguments.get("name", "Helix Antenna"),
            )
        elif name == "openems_generate_script":
            design_id = arguments["design_id"]
            if design_id in designer.designs:
                result = designer.generate_openems_script(design_id)
            elif design_id in pcb_designer.designs:
                result = pcb_designer.generate_openems_script(design_id)
            else:
                result = {"success": False, "error": f"Design not found: {design_id}"}
        elif name == "openems_list_designs":
            antenna_result = designer.list_designs()
            pcb_result = pcb_designer.list_designs()
            all_designs = antenna_result.get("designs", []) + pcb_result.get("designs", [])
            result = {"success": True, "designs": all_designs, "count": len(all_designs)}
        elif name == "openems_get_design":
            design_id = arguments["design_id"]
            if design_id in designer.designs:
                result = designer.get_design(design_id)
            elif design_id in pcb_designer.designs:
                result = pcb_designer.get_design(design_id)
            else:
                result = {"success": False, "error": f"Design not found: {design_id}"}
        elif name == "openems_list_antenna_types":
            result = designer.list_antenna_types()
        elif name == "openems_check_installation":
            result = {
                "success": True,
                "openems_available": OPENEMS_AVAILABLE,
                "message": (
                    "OpenEMS is installed and ready for simulations"
                    if OPENEMS_AVAILABLE
                    else "OpenEMS not installed. Design tools work, but simulation requires: pip install CSXCAD openEMS"
                ),
            }
        elif name == "openems_export_design":
            design_id = arguments["design_id"]
            if design_id in designer.designs:
                result = designer.export_design(
                    design_id=design_id,
                    format=arguments.get("format", "ascii"),
                )
            elif design_id in pcb_designer.designs:
                # PCB structures export as JSON (no SVG/ASCII art yet)
                result = {
                    "success": True,
                    "format": arguments.get("format", "json"),
                    "data": pcb_designer.designs[design_id],
                    "usage": "Import into CAD software or visualization tools",
                }
            else:
                result = {"success": False, "error": f"Design not found: {design_id}"}
        elif name == "openems_optimize_hints":
            design_id = arguments["design_id"]
            if design_id in designer.designs:
                result = designer.get_optimization_hints(design_id)
            elif design_id in pcb_designer.designs:
                # Basic optimization hints for PCB structures
                pcb_design = pcb_designer.designs[design_id]
                result = {
                    "success": True,
                    "design_id": design_id,
                    "design_type": pcb_design["type"],
                    "optimization_hints": [
                        {"category": "General", "suggestions": [
                            "Run full FDTD simulation for accurate S-parameters",
                            "Refine mesh around trace edges and via barrel",
                            "Verify impedance with frequency-dependent substrate properties",
                        ]}
                    ],
                    "parameter_sensitivity": {},
                    "next_steps": [
                        "Generate OpenEMS script and run simulation",
                        "Compare simulated Z0 with analytical calculation",
                        "Sweep frequency to find usable bandwidth",
                    ],
                }
            else:
                result = {"success": False, "error": f"Design not found: {design_id}"}
        elif name == "openems_compare_designs":
            result = designer.compare_designs(arguments["design_ids"])
        elif name == "openems_create_microstrip":
            result = pcb_designer.create_microstrip_trace(
                frequency_ghz=arguments["frequency_ghz"],
                trace_width_mm=arguments["trace_width_mm"],
                trace_length_mm=arguments["trace_length_mm"],
                dielectric_height_mm=arguments["dielectric_height_mm"],
                substrate_er=arguments.get("substrate_er", 4.2),
                name=arguments.get("name", "Microstrip Trace"),
            )
        elif name == "openems_create_coupled_lines":
            result = pcb_designer.create_coupled_lines(
                frequency_ghz=arguments["frequency_ghz"],
                trace_width_mm=arguments["trace_width_mm"],
                spacing_mm=arguments["spacing_mm"],
                trace_length_mm=arguments["trace_length_mm"],
                dielectric_height_mm=arguments["dielectric_height_mm"],
                substrate_er=arguments.get("substrate_er", 4.2),
                name=arguments.get("name", "Coupled Lines"),
            )
        elif name == "openems_create_via":
            result = pcb_designer.create_via_transition(
                frequency_ghz=arguments["frequency_ghz"],
                drill_mm=arguments["drill_mm"],
                pad_mm=arguments["pad_mm"],
                dielectric_height_mm=arguments["dielectric_height_mm"],
                substrate_er=arguments.get("substrate_er", 4.2),
                name=arguments.get("name", "Via Transition"),
            )
        else:
            result = {"success": False, "error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        error_result = {"success": False, "error": str(e)}
        return [TextContent(type="text", text=json.dumps(error_result))]


def main():
    """Run the MCP server."""

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(run())


if __name__ == "__main__":
    main()

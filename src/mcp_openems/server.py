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
from mcp.types import Tool, TextContent

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

        # Estimated directivity
        directivity_dbi = 10 * math.log10(4 * math.pi * w * l / ((c / f) ** 2) * 0.8)

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
Sim_Path = 'Sim_Patch'
if os.path.exists(Sim_Path):
    import shutil
    shutil.rmtree(Sim_Path)

FDTD.Run(Sim_Path, verbose=3, numThreads=4)

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

Sim_Path = 'Sim_Dipole'
if os.path.exists(Sim_Path):
    import shutil
    shutil.rmtree(Sim_Path)

FDTD.Run(Sim_Path, verbose=3, numThreads=4)

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

Sim_Path = 'Sim_Monopole'
if os.path.exists(Sim_Path):
    import shutil
    shutil.rmtree(Sim_Path)

FDTD.Run(Sim_Path, verbose=3, numThreads=4)

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


# Global designer instance
designer = AntennaDesigner()


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
            result = designer.generate_openems_script(arguments["design_id"])
        elif name == "openems_list_designs":
            result = designer.list_designs()
        elif name == "openems_get_design":
            result = designer.get_design(arguments["design_id"])
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
        else:
            result = {"success": False, "error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        error_result = {"success": False, "error": str(e)}
        return [TextContent(type="text", text=json.dumps(error_result))]


def main():
    """Run the MCP server."""
    import sys

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

<div align="center">

<img src="assets/logo-banner.svg" alt="mcp-openems — 3D FDTD electromagnetic simulation" width="100%"/>

<br/>

[![License](https://img.shields.io/badge/License-AGPL--3.0-1E40AF.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-server-A78BFA.svg)](https://modelcontextprotocol.io)
[![eng-mcp-suite](https://img.shields.io/badge/eng--mcp--suite-member-22D3EE.svg)](https://github.com/RFingAdam/eng-mcp-suite)

**AI-assisted antenna and RF-structure design via openEMS FDTD, driven over MCP.**
**Patch / dipole / monopole / horn / helix antennas, microstrip and coupled-line transmission lines, via transitions — geometry, analytical Z₀/εeff, and ready-to-run openEMS Python scripts.**

[Quick start](#installation) ·
[Tools](#tool-reference) ·
[Examples](examples/) ·
[Suite catalog](https://github.com/RFingAdam/eng-mcp-suite)

</div>

---

An MCP server for designing antennas and electromagnetic structures using openEMS FDTD simulation. Provides analytical design calculators that work immediately, plus openEMS script generation for full-wave simulation.

## Part of the engineering toolkit

This repo is part of [eng-mcp-suite](https://github.com/RFingAdam/eng-mcp-suite)
— an MCP-driven engineering toolkit for RF / EMC / PCB / signal-integrity /
lab-test workflows.

Related tools in the toolkit:

| Tool | When to reach for it |
|---|---|
| [**lineforge**](https://github.com/RFingAdam/lineforge) | 2D quasi-TEM closed-form for transmission lines (microstrip, stripline, CPWG, differential, three-conductor). Use this when you need impedance fast and the geometry is 2D. |
| [**mcp-nec2-antenna**](https://github.com/RFingAdam/mcp-nec2-antenna) | Wire-antenna method-of-moments (dipole / Yagi / vertical / loop / inverted-V). Use this when you have a wire-antenna geometry and don't need the 3D-field detail of FDTD. |
| [**mcp-pcb-emcopilot**](https://github.com/RFingAdam/mcp-pcb-emcopilot) | PCB layout review (decoupling, return paths, plane resonances, DDR/PCIe/USB SI). Often pairs with mcp-openems for full-wave validation of a flagged region. |

When to use mcp-openems specifically: full-wave 3D FDTD validation, broadband
S-parameters, near/far-field characterization, antenna geometries with 3D
features (horns, helices), or when closed-form is running out of accuracy.

## Features

### Antenna Design Tools
- **openems_create_patch** - Microstrip patch antenna (WiFi, GPS, satellite)
- **openems_create_dipole** - Half-wave dipole antenna
- **openems_create_monopole** - Quarter-wave monopole over ground plane
- **openems_create_horn** - Pyramidal horn antenna for specified gain
- **openems_create_helix** - Axial-mode helix for circular polarization

### Simulation & Export
- **openems_generate_script** - Generate complete OpenEMS Python simulation script
- **openems_check_installation** - Check if OpenEMS is available

### Design Management
- **openems_list_designs** - List all designs in session
- **openems_get_design** - Get full design details with geometry
- **openems_list_antenna_types** - Reference for antenna types and applications

## Installation

### 1. Clone and install

```bash
git clone https://github.com/RFingAdam/mcp-openems.git
cd mcp-openems
uv pip install -e .
```

### 2. (Optional) Install OpenEMS for simulation

```bash
# The MCP works without OpenEMS - design tools calculate dimensions analytically
# For full FDTD simulation, install OpenEMS:
pip install CSXCAD openEMS
```

### 3. Add to your MCP client

**Claude Code:**
```bash
claude mcp add openems -- uv run --directory /path/to/mcp-openems mcp-openems
```

**Config file format:**
```json
{
  "command": "uv",
  "args": ["run", "--directory", "/path/to/mcp-openems", "mcp-openems"]
}
```

## Usage Examples

### Design a WiFi patch antenna

```
Design a 2.4 GHz patch antenna on FR-4 substrate (er=4.4, 1.6mm thick)
```

The AI will:
1. Use `openems_create_patch` to calculate dimensions
2. Return patch length, width, feed position
3. Provide estimated directivity and impedance

### Design a satellite uplink antenna

```
I need a circularly polarized antenna for 5.8 GHz with about 12 dBi gain
```

The AI will use `openems_create_helix` for CP requirements.

### Generate simulation script

```
Generate an OpenEMS script for this antenna so I can run a full simulation
```

The AI will use `openems_generate_script` to create a complete Python script.

### Compare antenna types

```
What antenna types are available? I need something for a handheld radio at 440 MHz
```

The AI will use `openems_list_antenna_types` and recommend appropriate options.

## Tool Reference

### Design Output Format

Each design tool returns:
- **design_id**: UUID for referencing the design
- **dimensions**: Calculated physical dimensions in mm
- **calculated**: Derived parameters (impedance, gain estimates)
- **geometry**: OpenEMS-compatible geometry specification

### Example Output

```json
{
  "success": true,
  "design_id": "550e8400-e29b-41d4-a716-446655440000",
  "design": {
    "name": "2.4 GHz Patch",
    "type": "patch",
    "frequency_ghz": 2.4,
    "dimensions": {
      "patch_length_mm": 28.85,
      "patch_width_mm": 37.24,
      "feed_inset_mm": 8.92,
      "ground_plane_mm": 94.48
    },
    "calculated": {
      "effective_er": 3.33,
      "estimated_directivity_dbi": 7.2
    }
  }
}
```

### Generated Script

The `openems_generate_script` tool creates a complete Python script that:
1. Sets up the FDTD simulation
2. Creates geometry from the design
3. Adds mesh with appropriate resolution
4. Runs the simulation
5. Extracts S-parameters and plots results

## Antenna Design Formulas

| Antenna | Method | Key Formula |
|---------|--------|-------------|
| Patch | Transmission Line Model | L = c/(2f√εeff) - 2ΔL |
| Dipole | Classical | L = 0.95 × λ/2 |
| Monopole | Image Theory | H = 0.95 × λ/4 |
| Horn | Aperture Theory | G = 4πAe/λ² |
| Helix | Kraus Model | C ≈ λ, S = C tan(α) |

## Without OpenEMS

Even without OpenEMS installed, this MCP provides:
- Analytical dimension calculations
- Geometry specifications for manual modeling
- Reference impedance and gain estimates
- OpenEMS script generation for later use

## With OpenEMS

With OpenEMS installed, you can run the generated scripts to:
- Perform full-wave FDTD simulation
- Get accurate S-parameters and input impedance
- Calculate radiation patterns and gain
- Visualize fields in ParaView

## Supported Frequencies

The design tools work across the RF spectrum:
- **HF (3-30 MHz)**: Dipole, monopole
- **VHF (30-300 MHz)**: All types
- **UHF (300 MHz-3 GHz)**: All types
- **Microwave (3-30 GHz)**: Patch, horn, helix
- **mmWave (30-300 GHz)**: Patch, horn (with appropriate substrate)

## License

[AGPL-3.0-or-later](LICENSE). Relicensed from Apache-2.0 in v0.2.0 to
align with the eng-mcp-suite toolkit-wide AGPL move. The underlying
openEMS engine remains GPL-3.0; this wrapper is AGPL-3.0-or-later and
invokes the engine at runtime without redistribution.

## Author

Adam Engelbrecht - [@RFingAdam](https://github.com/RFingAdam)

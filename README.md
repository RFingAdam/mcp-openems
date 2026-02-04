<p align="center">
  <img src="assets/logo.svg" alt="MCP OpenEMS" width="400">
</p>

<p align="center">
  <strong>AI-assisted antenna design and electromagnetic simulation via MCP</strong>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#features">Features</a> •
  <a href="#usage-examples">Usage</a> •
  <a href="#tool-reference">Tool Reference</a>
</p>

---

An MCP server for designing antennas and electromagnetic structures using OpenEMS FDTD simulation. Provides analytical design calculators that work immediately, plus OpenEMS script generation for full-wave simulation.

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

Apache-2.0

## Author

Adam Engelbrecht - [@RFingAdam](https://github.com/RFingAdam)

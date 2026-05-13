# Changelog

All notable changes to **mcp-openems** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-05-13

### Changed
- **License: Apache-2.0 → AGPL-3.0-or-later.** Aligns with the
  eng-mcp-suite toolkit-wide AGPL move. The AGPL closes the
  "wrap as a paid SaaS without contributing back" gap by extending
  copyleft to network use. The underlying openEMS engine remains
  GPL-3.0; this wrapper invokes it at runtime without redistribution,
  so the wrapper's AGPL license is independent of the engine's GPL.
  See the
  [LICENSE_SUMMARY](https://github.com/RFingAdam/eng-mcp-suite/blob/main/LICENSE_SUMMARY.md)
  in eng-mcp-suite for the toolkit-wide rationale.

## [0.1.0] — 2026-05-13

### Added
- Initial MCP server wrapping openEMS FDTD for AI-assisted antenna and
  RF-structure design.
- Antenna design tools: `openems_create_patch`, `openems_create_dipole`,
  `openems_create_monopole`, `openems_create_horn`, `openems_create_helix`.
- Transmission-line and PCB-structure tools: `openems_create_microstrip`,
  `openems_create_coupled_lines`, `openems_create_via`.
- Design management: `openems_list_designs`, `openems_get_design`,
  `openems_compare_designs`, `openems_export_design`.
- Workflow helpers: `openems_check_installation`,
  `openems_generate_script`, `openems_optimize_hints`,
  `openems_list_antenna_types`.
- Runnable microstrip impedance reference example
  (`examples/microstrip_50ohm.py`) validated against `lineforge` closed-
  form to within ±2 %, doubling as an end-to-end smoke test of the
  openEMS install.
- Brand assets aligned with eng-mcp-suite design system (logo, banner).
- "Part of the engineering toolkit" cross-link in README.

### Notes
- Wrapper code is Apache-2.0; the underlying openEMS engine is GPL-3.0.
  Running openEMS from this wrapper at runtime does not affect this
  wrapper's licensing; redistributions of the openEMS engine remain
  subject to GPL terms.

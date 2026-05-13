# Examples

Runnable examples that exercise the mcp-openems wrapper end-to-end and
double as installation smoke tests.

## Available examples

| Example | What it does | Requires openEMS engine? |
|---|---|---|
| [`microstrip_50ohm.py`](microstrip_50ohm.py) | 50-Ω microstrip on 0.787 mm FR4. Sets up the geometry, reports the wrapper's analytical Z₀/εeff, and prints agreement vs the Hammerstad-Jensen reference. | Only for the FDTD step — runs offline with `--closed-form-only`. |

## Running

```bash
pip install -e .                                 # install mcp-openems
python examples/microstrip_50ohm.py              # full path
python examples/microstrip_50ohm.py --closed-form-only   # offline
```

The `--closed-form-only` mode is useful for CI / install-smoke testing
when openEMS isn't built locally. It prints the reference numbers
without invoking the FDTD engine.

## Related work in the toolkit

The same case is the anchor of lineforge's analytical-vs-FDTD validation
story. See [`lineforge/examples/09_l3_sig1_em_validation/`](https://github.com/RFingAdam/lineforge/tree/main/examples/09_l3_sig1_em_validation)
for the cross-validation framework — analytical solver in lineforge,
FDTD reference via openEMS, agreement to ±2 %.

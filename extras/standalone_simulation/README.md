# Optional standalone simulation

This utility runs a trained checkpoint through standalone MetaDrive episodes. It
is outside the primary `train -> identify + predict` workflow and smoke test.

From the repository root:

```bash
python -m extras.standalone_simulation.simulate \
  --checkpoint <path/to/latest_model.pt> \
  --episodes 1 \
  --max_steps 5 \
  --no_render \
  --device cpu
```

A successful run ends with:

```text
OPTIONAL_SIMULATION_OK
```

The command has no minimal-inference fallback. Any loading, environment, or
episode failure returns a non-zero status instead of printing the success marker.

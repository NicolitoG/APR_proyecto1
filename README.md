# MeteoGym

MeteoGym is a Python/Gymnasium environment for controlling which LEO satellites accept tasks from ground users while respecting battery availability.

- [`src/`](src/README.md): installable simulator, command-line interface and tests.
- [`presentacion/`](presentacion/): Spanish LaTeX presentation of the research problem.
- [`agents.md`](agents.md): repository context and development conventions.

From this directory, use the existing Conda environment:

```bash
conda activate meteogym
python -m pip install -e './src[dev]'
meteogym run --satellites 1 541 --users 100 --duration 5400 --step 10 --seed 10
```

The bundled constellation contains 1,080 synthetic satellites.
Only the selected satellites participate in an episode.
Simulations start at the bundled TLE epoch in October 2016 by default.
See the [simulator README](src/README.md) for workload, energy assumptions, contact-table export, and experiment configuration.

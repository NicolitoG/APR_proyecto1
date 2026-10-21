# Satellite task-processing presentation

This directory contains the Spanish Beamer presentation describing MeteoGym's research environment.
The Python implementation and installation instructions are in [`../src/`](../src/README.md).

Build from this directory:

```bash
latexmk -xelatex main.tex
```

The presentation defines binary activation decisions for LEO satellites, random ground-user tasks, nearest-active-satellite uploads, orbital charging/discharging, and reward equal to successful tasks minus failures.
The environment exposes satellite state and visible-user task information at each simulation step.

`main.tex` contains the slides, `estilo-diinf.tex` defines the visual style, and `images/` contains source and rendered figure assets.
Keep slide content in Spanish.
The voltage figure is schematic; implemented physical assumptions and sources are documented in the [energy model](../src/docs/energy-model.md).

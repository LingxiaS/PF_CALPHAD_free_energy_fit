# Gibbs Free Energy Curves: calculations, temperature-dependent fitting & interpolation

This project plots and fits the Gibbs Free Energy curves for binary system at various temperatures.
It uses CALPHAD principles to demonstrate phase stability, equilibrium masking, and common tangent determination.

## 📁 File Structure

| File | Purpose |
| :--- | :--- |
| `AlIn_CALPHAD_fit_clean.ipynb` | **Main Script:** Loads data, do calculations, processes results, do fitting and interpolation, and generates plots. |
| `thermo_data.py` | Defines and stores the CALPHAD thermodynamic database (TDB text) as a string constant. |
| `thermo_utilities.py` | Helper functions for composition masking, colormap truncation, and common tangent solving (via `fsolve`). |

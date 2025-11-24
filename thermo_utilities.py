import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import fsolve

# ========== define the colormap ==========
def truncate_colormap(cmap, min_val=0.2, max_val=0.8, n=256):
    """
    Truncates a given colormap to use a subset of its color range.
    
    Args:
        cmap (matplotlib.colors.Colormap): The original colormap object.
        min_val (float): The minimum fraction of the original colormap to use (0.0 to 1.0).
        max_val (float): The maximum fraction of the original colormap to use (0.0 to 1.0).
        n (int): Number of steps in the new colormap.

    Returns:
        matplotlib.colors.LinearSegmentedColormap: The new, truncated colormap.
    """
    # Use plt.cm.colors.LinearSegmentedColormap for correct import
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{min_val:.2f},{max_val:.2f})',
        cmap(np.linspace(min_val, max_val, n)))
    return new_cmap

# ========== do temperature interpolation ==========
# NOTE: This function relies on external variables 'fit_windows_lowT' and 
# 'fit_windows_highT' which must be defined in the main script where this function is called.
def interpolate_fit_windows(T, T_min, T_max, fit_windows_lowT, fit_windows_highT):
    """
    Performs linear interpolation of fitting window sizes based on temperature.

    Args:
        T (float): Current temperature for interpolation.
        T_min (float): Lower temperature bound.
        T_max (float): Upper temperature bound.
        fit_windows_lowT (dict): Dictionary of window sizes at T_min.
        fit_windows_highT (dict): Dictionary of window sizes at T_max.

    Returns:
        dict: Interpolated window sizes.
    """
    # Simple clamped linear interpolation from T_min to T_max
    alpha = (T - T_min) / (T_max - T_min)
    alpha = max(0.0, min(1.0, alpha))  # Clamp between 0 and 1
    print(f"T = {T:.2f} K, alpha = {alpha:.3f}")
    return {
        key: int(round(fit_windows_lowT[key] * (1 - alpha) + fit_windows_highT[key] * alpha))
        for key in fit_windows_lowT
    }

# ========== fit & plot ==========
# Function to fit and plot coefficients
def fit_and_plot_coeffs(temperatures, coeffs, phase, index):
    """
    Fits coefficient data vs. temperature to a linear function and plots the results.

    Args:
        temperatures (np.array): Array of temperatures (K).
        coeffs (np.array): Array of coefficient values corresponding to temperatures.
        phase (str): Name of the phase (for title/labeling).
        index (int): Index of the coefficient (for title/labeling).

    Returns:
        np.array: The fit parameters [slope, intercept].
    """
    fit_params = np.polyfit(temperatures, coeffs, 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, coeffs, 'o', label=f'{phase} Coeff[{index}] Data')
    plt.plot(temperatures, fit_params[0] * temperatures + fit_params[1], 
             '-', label=f'{phase} Coeff[{index}] Fit: {fit_params[0]:.3e}*T + {fit_params[1]:.3e}')
    
    plt.xlabel('Temperature (K)')
    plt.ylabel(f'{phase} Fit Coefficient [{index}]')
    plt.title(f'Temperature Dependency of {phase} Coefficient [{index}]')
    plt.legend()
    plt.grid(True)
    plt.show()
    return fit_params

# ========== Tangent Solver ==========
def calculate_common_tangent(A1, B1, C1, A2, B2, C2):
    """
    Calculates the common tangent compositions for two parabolic Gibbs energy curves.
    The curves are defined as G(x) = A(x - B)^2 + C.

    Args:
        A1, B1, C1 (float): Parameters for the first curve.
        A2, B2, C2 (float): Parameters for the second curve.

    Returns:
        np.array: The compositions [x1, x2] where the common tangent touches.
    """
    # Derivative of the first curve: dG1/dx = 2*A1*(x - B1)
    df1 = lambda x: 2 * A1 * (x - B1)
    # Derivative of the second curve: dG2/dx = 2*A2*(x - B2)
    df2 = lambda x: 2 * A2 * (x - B2)
    # First curve: G1(x) = A1*(x - B1)^2 + C1
    f1  = lambda x: A1 * (x - B1)**2 + C1
    # Second curve: G2(x) = A2*(x - B2)^2 + C2
    f2  = lambda x: A2 * (x - B2)**2 + C2
    
    # System of equations:
    # Eq 1 (Equal slope): dG1/dx1 = dG2/dx2
    # Eq 2 (Equal tangent line intercept): G1(x1) - x1*dG1/dx1 = G2(x2) - x2*dG2/dx2
    def system(x): 
        x1, x2 = x
        return [
            df1(x1) - df2(x2), # Eq 1: Slopes are equal
            f1(x1) - df1(x1) * x1 - (f2(x2) - df2(x2) * x2) # Eq 2: Tangent intercepts are equal
        ]
    
    # Solve the system starting from the minimums of the parabolas
    # The minimums are at x = B1 and x = B2
    return fsolve(system, [B1, B2])

# ========== Composition Masking ==========
def create_mask(x_sorted_data, min_bound, max_bound):
    """
    Generates a boolean mask for composition data based on min and max bounds.

    Args:
        x_sorted_data (np.array): The composition data (mole fraction) for the phase.
        min_bound (float): The minimum mole fraction to include (>=).
        max_bound (float): The maximum mole fraction to include (<=).

    Returns:
        np.array: A boolean mask of the same shape as x_sorted_data.
    """
    return (x_sorted_data >= min_bound) & (x_sorted_data <= max_bound)

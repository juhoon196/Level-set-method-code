#!/usr/bin/env python3
"""
Visualization script for G-equation Level-Set Solver results.

This script reads binary output files and creates:
- Contour plots of the level-set field
- Comparison between initial and final states
- Animation of the advection process (if multiple snapshots available)

Usage:
    python visualize.py                     # Default: visualize initial and final
    python visualize.py --file G_final.bin  # Visualize specific file
    python visualize.py --compare           # Compare initial vs final
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import argparse
import os
import struct


def read_binary_file(filename):
    """
    Read binary output file from the solver.

    File format:
    - Header: nx, ny, nghost (3 integers)
    - Data: G values in row-major order

    Returns:
        G: 2D numpy array (interior points only)
        nx, ny: Grid dimensions
        nghost: Number of ghost cells
    """
    with open(filename, 'rb') as f:
        # Read header
        header = struct.unpack('3i', f.read(12))
        nx, ny, nghost = header

        # Calculate total size
        nx_total = nx + 2 * nghost
        ny_total = ny + 2 * nghost
        total_size = nx_total * ny_total

        # Read data
        data = np.frombuffer(f.read(total_size * 8), dtype=np.float64)
        G_full = data.reshape((ny_total, nx_total))

        # Extract interior points (remove ghost cells)
        G = G_full[nghost:nghost+ny, nghost:nghost+nx]

    return G, nx, ny, nghost


def read_interior_binary_file(filename):
    """
    Read binary file containing only interior points.

    File format:
    - Header: nx, ny (2 integers)
    - Data: G values in row-major order
    """
    with open(filename, 'rb') as f:
        # Read header
        header = struct.unpack('2i', f.read(8))
        nx, ny = header

        # Read data
        data = np.frombuffer(f.read(nx * ny * 8), dtype=np.float64)
        G = data.reshape((ny, nx))

    return G, nx, ny


def plot_contour(G, title='Level-Set Field', filename=None, show=True):
    """
    Create a contour plot of the level-set field.
    """
    ny, nx = G.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Color map with zero at white
    vmax = max(abs(G.min()), abs(G.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Filled contour
    cf = ax.contourf(X, Y, G, levels=50, cmap='RdBu_r', norm=norm)
    plt.colorbar(cf, ax=ax, label='G')

    # Zero contour (interface)
    cs = ax.contour(X, Y, G, levels=[0], colors='k', linewidths=2)
    ax.clabel(cs, inline=True, fontsize=10, fmt='G=0')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(G_initial, G_final, filename=None, show=True):
    """
    Create side-by-side comparison of initial and final states.
    """
    ny, nx = G_initial.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Initial state
    vmax = max(abs(G_initial.min()), abs(G_initial.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    cf1 = axes[0].contourf(X, Y, G_initial, levels=50, cmap='RdBu_r', norm=norm)
    axes[0].contour(X, Y, G_initial, levels=[0], colors='k', linewidths=2)
    plt.colorbar(cf1, ax=axes[0], label='G')
    axes[0].set_title('Initial State (t=0)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')

    # Final state
    vmax = max(abs(G_final.min()), abs(G_final.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    cf2 = axes[1].contourf(X, Y, G_final, levels=50, cmap='RdBu_r', norm=norm)
    axes[1].contour(X, Y, G_final, levels=[0], colors='k', linewidths=2)
    plt.colorbar(cf2, ax=axes[1], label='G')
    axes[1].set_title('Final State (t=T)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')

    # Difference
    diff = G_final - G_initial
    vmax_diff = max(abs(diff.min()), abs(diff.max()))
    if vmax_diff < 1e-10:
        vmax_diff = 1e-10
    norm_diff = TwoSlopeNorm(vmin=-vmax_diff, vcenter=0, vmax=vmax_diff)

    cf3 = axes[2].contourf(X, Y, diff, levels=50, cmap='RdBu_r', norm=norm_diff)
    axes[2].contour(X, Y, G_initial, levels=[0], colors='b', linewidths=1.5,
                    linestyles='dashed', label='Initial')
    axes[2].contour(X, Y, G_final, levels=[0], colors='r', linewidths=1.5,
                    label='Final')
    plt.colorbar(cf3, ax=axes[2], label='ΔG')
    axes[2].set_title('Difference (Final - Initial)')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_aspect('equal')

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_interface_only(G_initial, G_final, filename=None, show=True):
    """
    Plot only the zero contours (interfaces) for comparison.
    """
    ny, nx = G_initial.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Initial interface (blue dashed)
    cs1 = ax.contour(X, Y, G_initial, levels=[0], colors='blue',
                     linewidths=2, linestyles='dashed')

    # Final interface (red solid)
    cs2 = ax.contour(X, Y, G_final, levels=[0], colors='red',
                     linewidths=2, linestyles='solid')

    ax.plot([], [], 'b--', linewidth=2, label='Initial (t=0)')
    ax.plot([], [], 'r-', linewidth=2, label='Final (t=T)')
    ax.legend(loc='upper right')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Interface Comparison')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")

    if show:
        plt.show()
    else:
        plt.close()


def calculate_error_metrics(G_initial, G_final):
    """
    Calculate error metrics between initial and final states.
    """
    # L2 error
    diff = G_final - G_initial
    l2_error = np.sqrt(np.mean(diff**2))

    # L_inf error
    linf_error = np.max(np.abs(diff))

    # Area conservation (count cells where G < 0)
    initial_area = np.sum(G_initial < 0)
    final_area = np.sum(G_final < 0)
    area_error = abs(final_area - initial_area) / initial_area if initial_area > 0 else 0

    print("\n=== Error Metrics ===")
    print(f"L2 Error:              {l2_error:.6e}")
    print(f"L_inf Error:           {linf_error:.6e}")
    print(f"Initial Area (cells):  {initial_area}")
    print(f"Final Area (cells):    {final_area}")
    print(f"Area Conservation:     {(1 - area_error)*100:.2f}%")

    return l2_error, linf_error, area_error


def main():
    parser = argparse.ArgumentParser(description='Visualize G-equation solver results')
    parser.add_argument('--dir', type=str, default='output',
                        help='Output directory (default: output)')
    parser.add_argument('--file', type=str, default=None,
                        help='Specific file to visualize')
    parser.add_argument('--compare', action='store_true',
                        help='Compare initial and final states')
    parser.add_argument('--interface', action='store_true',
                        help='Plot interface only')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots (only save)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save plot to file')

    args = parser.parse_args()

    output_dir = args.dir
    show = not args.no_show

    if args.file:
        # Visualize specific file
        filepath = os.path.join(output_dir, args.file)
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            return

        G, nx, ny, nghost = read_binary_file(filepath)
        print(f"Loaded: {filepath}")
        print(f"Grid size: {nx} x {ny}, ghost cells: {nghost}")

        save_file = args.save if args.save else None
        plot_contour(G, title=args.file, filename=save_file, show=show)

    elif args.compare or args.interface:
        # Compare initial and final states
        initial_file = os.path.join(output_dir, 'G_initial.bin')
        final_file = os.path.join(output_dir, 'G_final.bin')

        if not os.path.exists(initial_file):
            print(f"Error: Initial file not found: {initial_file}")
            return
        if not os.path.exists(final_file):
            print(f"Error: Final file not found: {final_file}")
            return

        G_initial, nx, ny, nghost = read_binary_file(initial_file)
        G_final, _, _, _ = read_binary_file(final_file)

        print(f"Grid size: {nx} x {ny}")

        # Calculate error metrics
        calculate_error_metrics(G_initial, G_final)

        if args.interface:
            save_file = args.save if args.save else 'interface_comparison.png'
            plot_interface_only(G_initial, G_final, filename=save_file, show=show)
        else:
            save_file = args.save if args.save else 'comparison.png'
            plot_comparison(G_initial, G_final, filename=save_file, show=show)

    else:
        # Default: visualize final state
        final_file = os.path.join(output_dir, 'G_final.bin')

        if os.path.exists(final_file):
            G, nx, ny, nghost = read_binary_file(final_file)
            print(f"Loaded: {final_file}")
            print(f"Grid size: {nx} x {ny}")

            save_file = args.save if args.save else None
            plot_contour(G, title='Final State', filename=save_file, show=show)
        else:
            print(f"No output files found in {output_dir}")
            print("Run the solver first: ./g_equation_solver")


if __name__ == '__main__':
    main()

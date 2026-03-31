"""
G-equation Interface Animation 3D (GPU version)

Reads binary output files and visualizes the G=0 isosurface animation.
Binary format: int32[4] (nx, ny, nz, nghost) + float64[nx_total * ny_total * nz_total]
Memory layout: x-fastest (i), then y (j), then z (k)
"""

import os
import re
import struct
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

# ─── Configuration ───────────────────────────────────────────────────────────
output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')

# ─── Gather and sort step files ──────────────────────────────────────────────
pattern = re.compile(r'G_step_(\d+)\.bin')
files = []
for fname in os.listdir(output_dir):
    m = pattern.match(fname)
    if m:
        files.append((int(m.group(1)), fname))
files.sort()

if not files:
    print(f"파일이 없습니다: {output_dir}")
    exit()

print(f"총 {len(files)}개 프레임 발견")

# ─── Visualization setup ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8), facecolor='white')
ax = fig.add_subplot(111, projection='3d')

for step_num, fname in files:
    filepath = os.path.join(output_dir, fname)

    with open(filepath, 'rb') as f:
        # Read header: 4 int32 values (nx, ny, nz, nghost)
        header = struct.unpack('4i', f.read(16))
        nx, ny, nz, nghost = header

        nx_total = nx + 2 * nghost
        ny_total = ny + 2 * nghost
        nz_total = nz + 2 * nghost
        total_size = nx_total * ny_total * nz_total

        # Read field data (float64)
        data = np.frombuffer(f.read(total_size * 8), dtype=np.float64)

    if data.size != total_size:
        print(f"Data size mismatch in {fname}, skipping")
        continue

    # Reshape: C++ layout is k*ny_total*nx_total + j*nx_total + i
    # → numpy shape (nz_total, ny_total, nx_total) where last dim = x (fastest)
    G_full = data.reshape((nz_total, ny_total, nx_total))

    # Remove ghost cells → interior (nz, ny, nx)
    G = G_full[nghost:-nghost, nghost:-nghost, nghost:-nghost]

    # Coordinates (matching physical domain [0, 1]^3)
    x_coords = np.linspace(0, 1, nx)
    y_coords = np.linspace(0, 1, ny)
    z_coords = np.linspace(0, 1, nz)

    # G shape is (nz, ny, nx); marching_cubes expects (z, y, x) ordering
    # spacing corresponds to (dz, dy, dx)
    dx = x_coords[1] - x_coords[0] if nx > 1 else 1.0
    dy = y_coords[1] - y_coords[0] if ny > 1 else 1.0
    dz = z_coords[1] - z_coords[0] if nz > 1 else 1.0

    try:
        verts, faces, _, _ = measure.marching_cubes(G, level=0.0, spacing=(dz, dy, dx))
    except ValueError:
        print(f"Step {step_num}: no isosurface found, skipping")
        continue

    # Offset vertices to match domain origin
    # verts columns are (z, y, x) since input array is (nz, ny, nx)
    verts_plot = verts.copy()
    verts_plot[:, 0] += z_coords[0]  # z offset
    verts_plot[:, 1] += y_coords[0]  # y offset
    verts_plot[:, 2] += x_coords[0]  # x offset

    # Plot
    ax.clear()
    mesh = Poly3DCollection(verts_plot[faces], alpha=0.7, edgecolor='none')
    mesh.set_facecolor('steelblue')
    ax.add_collection3d(mesh)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('Z')
    ax.set_ylabel('Y')
    ax.set_zlabel('X')
    ax.set_title(f'Step: {step_num} (Grid: {nx}x{ny}x{nz})')
    ax.view_init(elev=25, azim=45)

    plt.draw()
    plt.pause(0.1)

print(f"애니메이션 종료 (총 {len(files)} 프레임)")
plt.show()

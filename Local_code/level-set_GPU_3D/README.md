# G-Equation Level-Set Solver (CUDA)

A GPU-accelerated solver for the G-equation (flame propagation equation) using the level-set method.

## Features

- **Spatial Discretization**: 5th-order WENO (WENO-5) scheme for convective terms
- **Time Integration**: 3rd-order TVD Runge-Kutta (RK3) scheme
- **Reinitialization**: Hartmann HCR-2 (High-order Constrained Reinitialization) method
- **Boundary Conditions**: Periodic (x-direction), Zero-gradient (y-direction)
- **Test Cases**: Pyramid (diamond) advection, Circle advection

## Governing Equations

The solver solves the G-equation:

$$\frac{\partial G}{\partial t} + \mathbf{u}_{eff} \cdot \nabla G = 0$$

where the effective velocity includes the flame speed:

$$\mathbf{u}_{eff} = \mathbf{u} - S_L \frac{\nabla G}{|\nabla G|}$$

## Building

### Requirements

- CUDA Toolkit (tested with CUDA 11.0+)
- GCC compiler
- GNU Make

### Compilation

```bash
# Build with default settings
make

# Build for specific GPU architecture
make CUDA_ARCH=sm_80  # For Ampere GPUs

# Build with debug flags
make debug

# Clean build files
make clean
```

## Usage

### Basic Usage

```bash
# Run with default parameters (Pyramid test, T=1.0, with reinitialization)
./g_equation_solver

# Show all options
./g_equation_solver -h
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-t <test>` | Test case: `pyramid` or `circle` | pyramid |
| `-n <N>` | Grid size (NxN) | 201 |
| `-T <time>` | Final simulation time | 1.0 |
| `-cfl <val>` | CFL number | 0.5 |
| `-sl <val>` | Laminar flame speed (S_L) | 0.0 |
| `-u <val>` | X-velocity | 1.0 |
| `-v <val>` | Y-velocity | 0.0 |
| `-reinit` | Enable reinitialization | - |
| `-no-reinit` | Disable reinitialization | - |
| `-ri <N>` | Reinit interval (steps) | 10 |
| `-riter <N>` | Reinit iterations | 5 |
| `-o <dir>` | Output directory | ./output |
| `-no-output` | Disable file output | - |
| `-q` | Quiet mode | - |

### Examples

```bash
# Pyramid advection test with reinitialization
./g_equation_solver -t pyramid -T 1.0 -reinit

# Circle advection without reinitialization
./g_equation_solver -t circle -T 1.0 -no-reinit

# Flame propagation with S_L = 0.1
./g_equation_solver -t pyramid -sl 0.1 -reinit

# Custom velocity field
./g_equation_solver -u 0.5 -v 0.5 -T 2.0
```

## Output Files

The solver generates the following files in the output directory:

- `G_initial.bin` - Initial level-set field (binary)
- `G_initial.vtk` - Initial field in VTK format (for visualization)
- `G_final.bin` - Final level-set field (binary)
- `G_final.vtk` - Final field in VTK format

### Binary File Format

```
Header: [nx, ny, nghost] (3 x int32)
Data: G[nx_total * ny_total] (float64, row-major order)
```

## Visualization

### Using Python

```bash
# Install dependencies
pip install numpy matplotlib

# Visualize final state
python scripts/visualize.py

# Compare initial and final states
python scripts/visualize.py --compare

# Plot interface only
python scripts/visualize.py --interface
```

### Using ParaView/VisIt

The VTK output files can be directly opened in ParaView or VisIt for advanced visualization.

## Numerical Methods

### WENO-5 Scheme

The 5th-order WENO reconstruction uses three candidate stencils with optimal weights:
- d₀ = 0.1, d₁ = 0.6, d₂ = 0.3

Nonlinear weights are computed using smoothness indicators to avoid oscillations near discontinuities.

### TVD RK3 (Shu-Osher Form)

The three-stage scheme:
1. G⁽¹⁾ = Gⁿ + Δt L(Gⁿ)
2. G⁽²⁾ = ¾Gⁿ + ¼G⁽¹⁾ + ¼Δt L(G⁽¹⁾)
3. Gⁿ⁺¹ = ⅓Gⁿ + ⅔G⁽²⁾ + ⅔Δt L(G⁽²⁾)

### HCR-2 Reinitialization

Solves the reinitialization equation with interface constraint:

$$\frac{\partial \phi}{\partial \tau} + S(\phi_0)(|\nabla\phi| - 1) = \beta F$$

where F is the forcing term to maintain interface position:

$$F_{i,j} = \frac{\psi_{i,j} - \phi_{i,j}}{\Delta x}$$

## Project Structure

```
level-set/
├── include/
│   ├── config.cuh           # Configuration and parameters
│   ├── weno5.cuh            # WENO-5 spatial discretization
│   ├── rk3.cuh              # TVD RK3 time integration
│   ├── reinitialization.cuh # HCR-2 reinitialization
│   ├── boundary.cuh         # Boundary conditions
│   ├── initial_conditions.cuh # Initial condition generators
│   └── io.cuh               # I/O and error calculation
├── src/
│   ├── solver.cu            # Main solver class
│   └── main.cu              # Entry point
├── scripts/
│   └── visualize.py         # Python visualization
├── output/                  # Output files
├── Makefile
└── README.md
```

## Configuration

To modify default parameters, edit `include/config.cuh`:

```cpp
// Grid parameters
constexpr int NX = 201;
constexpr int NY = 201;
constexpr int NGHOST = 3;

// Physical parameters
constexpr double S_L = 0.0;      // Flame speed
constexpr double U_CONST = 1.0;  // X-velocity
constexpr double V_CONST = 0.0;  // Y-velocity

// Time parameters
constexpr double CFL = 0.5;
constexpr double T_FINAL = 1.0;

// Reinitialization
constexpr bool ENABLE_REINIT = true;
constexpr int REINIT_INTERVAL = 10;
```

## References

1. Jiang, G.-S., & Shu, C.-W. (1996). Efficient implementation of weighted ENO schemes. *Journal of Computational Physics*, 126(1), 202-228.

2. Shu, C.-W., & Osher, S. (1988). Efficient implementation of essentially non-oscillatory shock-capturing schemes. *Journal of Computational Physics*, 77(2), 439-471.

3. Hartmann, D., Meinke, M., & Schröder, W. (2010). Differential equation based constrained reinitialization for level set methods. *Journal of Computational Physics*, 229(5), 1585-1611.

## License

KAIST Combustion Modeling Lab.

# G-Equation Level-Set Solver (MPI Version)

A parallel implementation of the G-equation level-set solver using MPI for domain decomposition.

## Features

- **WENO-5** spatial discretization for high-order accuracy
- **TVD RK3** time integration (Shu-Osher form)
- **HCR-2 reinitialization** (Hartmann et al. 2010)
- **MPI parallelization** with 1D domain decomposition in Y direction

## Building

```bash
# Standard build
make

# Debug build
make debug

# Clean
make clean
```

## Running

```bash
# Run with default 4 processes
make run

# Run with specific number of processes
make NP=8 run

# Run pyramid advection test
make test

# Run with custom options
mpirun -np 4 ./g_equation_solver -t pyramid -T 1.0 -reinit
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-t <test>` | Test case: pyramid, circle, zalesak | pyramid |
| `-T <time>` | Final simulation time | 6.283185 |
| `-cfl <val>` | CFL number | 0.2 |
| `-sl <val>` | Laminar flame speed | 0.0 |
| `-u <val>` | X-velocity | 1.0 |
| `-v <val>` | Y-velocity | 0.0 |
| `-reinit` | Enable reinitialization | disabled |
| `-no-reinit` | Disable reinitialization | - |
| `-ri <N>` | Reinit interval (steps) | 10 |
| `-riter <N>` | Reinit iterations | 2 |
| `-o <dir>` | Output directory | ./output |
| `-no-output` | Disable file output | - |
| `-q` | Quiet mode | - |
| `-h` | Show help | - |

## Test Cases

### Pyramid Advection
```bash
mpirun -np 4 ./g_equation_solver -t pyramid -T 1.0 -reinit
```

### Circle Advection
```bash
mpirun -np 4 ./g_equation_solver -t circle -T 1.0 -reinit
```

### Zalesak's Slotted Disk
```bash
mpirun -np 4 ./g_equation_solver -t zalesak -T 6.283185 -reinit
```

## Domain Decomposition

The domain is decomposed in the Y direction:

```
┌─────────────────────────────────┐
│       Process 0 (rank 0)        │  ← Bottom portion
│         local_ny rows           │
├─────────────────────────────────┤
│       Process 1 (rank 1)        │
│         local_ny rows           │
├─────────────────────────────────┤
│            ...                  │
├─────────────────────────────────┤
│     Process N-1 (rank N-1)      │  ← Top portion
│         local_ny rows           │
└─────────────────────────────────┘
```

Each process maintains 3 ghost cells (for WENO-5 stencil) on each side that are exchanged via MPI.

## Output Files

- `output/G_initial.bin` - Initial field
- `output/G_final.bin` - Final field
- `output/G_step_XXXXXX.bin` - Snapshots every 10 steps

## Comparison with GPU Version

This MPI version produces numerically identical results to the CUDA GPU version. The key differences are:

| Aspect | GPU Version | MPI Version |
|--------|-------------|-------------|
| Parallelization | CUDA threads | MPI processes |
| Memory | Device memory | Distributed memory |
| Communication | Implicit (shared memory) | Explicit (MPI messages) |
| Scalability | Single GPU | Multiple nodes |

## References

- Jiang & Shu (1996) - WENO-5 scheme
- Shu & Osher (1988) - TVD RK3 scheme
- Hartmann et al. (2010) - HCR-2 reinitialization

## License

KAIST Combustion Modeling Lab.

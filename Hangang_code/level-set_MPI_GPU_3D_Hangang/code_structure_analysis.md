# G-Equation Level-Set Solver 3D (MPI + GPU, Hangang HPC) -- Code Structure Analysis

## 1. Project Overview

| Property | Value |
|---|---|
| **Purpose** | Solve the G-equation (level-set interface tracking) in 3D |
| **Parallelism** | MPI + CUDA hybrid (multi-GPU), 1D Y-decomposition |
| **Target HPC** | Hangang supercomputer (NVIDIA GH200 120GB, Hopper arch) |
| **Spatial scheme** | WENO-5 (5th-order Weighted ENO) upwind |
| **Time integration** | TVD RK3 (3rd-order Shu-Osher) |
| **Reinitialization** | Not included |
| **Grid** | 201x201x201 structured, uniform spacing |
| **Domain** | [0, 1]^3, periodic in all directions |
| **Language** | CUDA C++14 with MPI |
| **Build** | Makefile with nvcc (sm_90 Hopper) + MPI |

---

## 2. Directory Structure

```
level-set_MPI_GPU_3D_Hangang/
├── Makefile                        # Build (nvcc sm_90, MPI auto-detect)
├── include/
│   ├── config.cuh                  # Grid 201^3, NGHOST=3, MPI decomp
│   ├── weno5.cuh                   # WENO-5 3D device functions
│   ├── rk3.cuh                     # RK3 3D kernels + RHS
│   ├── boundary.cuh                # MPI halo exchange + periodic X,Z
│   └── initial_conditions.cuh      # Sphere SDF + deformation velocity
├── src/
│   └── main.cu                     # MPI+CUDA driver + I/O
├── scripts/
│   ├── run.sh                      # SLURM job script (8 ranks, 2 nodes)
│   ├── animation.py                # Python 3D visualization
│   └── animation.m                 # MATLAB 3D visualization
└── log/
    ├── simulation.log              # Final results
    ├── profile_0..7.nsys-rep       # NVIDIA Nsys profiling (8 ranks)
    └── slurm-4031.out              # SLURM output
```

---

## 3. Key Differences from Standard MPI_GPU_3D

| Aspect | MPI_GPU_3D (Local) | MPI_GPU_3D_Hangang |
|---|---|---|
| **Grid** | 64x64x64 | **201x201x201** (27x more cells) |
| **Time step** | DT=0.001 (fixed) | **DT=0.0 (auto CFL)** |
| **CUDA arch** | sm_86 (RTX 3090) | **sm_90 (GH200 Hopper)** |
| **Job scheduler** | Manual mpirun | **SLURM + modules** |
| **Profiling** | None | **NVIDIA Nsys (all ranks)** |
| **Network** | Local | **CXI fabric (libfabric)** |

---

## 4. Hangang HPC Environment (run.sh)

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=2        # 2 GH200 GPUs per node
#SBATCH --partition=cas_gh200

module load CUDA/12.8 OpenMPI/5.0.10 GCC-native/14 libfabric/2.3.1

export NCCL_CROSS_NIC=1
export NCCL_NET="AWS Libfabric"
export FI_PROVIDER=cxi

nsys profile --trace=cuda,mpi,nvtx --output=log/profile_%q{OMPI_COMM_WORLD_RANK} \
    mpirun -np 8 ./g_equation_solver_mpi_gpu
```

---

## 5. Architecture & Halo Exchange

```mermaid
flowchart LR
    A["packYHalo kernel"] --> B["cudaMemcpy D2H"]
    B --> C["MPI_Isend/Irecv via CXI"]
    C --> D["MPI_Waitall"]
    D --> E["cudaMemcpy H2D"]
    E --> F["unpackYHalo kernel"]
```

- 8 ranks across 2 nodes, GPU via rank % num_devices
- Y-periodic: rank 0 <-> rank 7 wrap-around
- Buffer size: NGHOST x nx_total x nz_total doubles

---

## 6. Module Descriptions

### 6.1 `config.cuh`
- Grid: NX=NY=NZ=201, NGHOST=3, DX=0.005
- DT=0.0 (auto-computed by CFL=0.2), T_FINAL=1.5
- CUDA blocks: 8x8x8, SimParams.ny = local_ny

### 6.2 `weno5.cuh`
- Identical to MPI_GPU_3D: WENO-5 3D upwind derivatives

### 6.3 `rk3.cuh`
- computeRHS + rk3Stage1/2/3 kernels, same as standard version

### 6.4 `boundary.cuh`
- HaloBuffers with pinned host staging
- GPU pack/unpack + MPI non-blocking exchange

### 6.5 `initial_conditions.cuh`
- Sphere (0.35, 0.35, 0.35), r=0.15 + deformation velocity

### 6.6 `main.cu`
- Same structure as MPI_GPU_3D with larger grid
- I/O: MPI_Gatherv + binary save

---

## 7. Performance Results

| Metric | Value |
|---|---|
| **MPI Ranks** | 8 (2 nodes x 4 ranks) |
| **GPUs** | NVIDIA GH200 120GB (sm_90) |
| **Grid** | 201 x 201 x 201 |
| **Steps** | 3001 |
| **L2 Error** | 1.669e-4 |
| **Volume Change** | 0.049% |
| **Wall Time** | 27.0 s |
| **Time/Step** | 9.0 ms |

---

## 8. Usage on Hangang

```bash
module load CUDA/12.8 OpenMPI/5.0.10
make CUDA_ARCH=sm_90
sbatch scripts/run.sh
cat log/simulation.log
python scripts/animation.py
```

#!/bin/bash
#SBATCH -J g_equation
#SBATCH -N 2
#SBATCH -t 1:00:00
#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --hint=nomultithread

# ===== 모듈 로드 =====
source $MODULESHOME/init/bash
module purge
module load cuda/12.8
module load openmpi/5.0.10
module load gcc-native/14
module load libfabric/2.3.1

# ===== 네트워크 환경 변수 =====
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_COLLNET_ENABLE=1
export NCCL_NET="AWS Libfabric"
export FI_MR_CACHE_MONITOR=memhooks
export FI_CXI_RX_MATCH_MODE=software
export FI_PROVIDER=cxi
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_RDZV_GET_MIN=4096

# ===== 실행 =====
SOLVER=/scratch/paop41a05/juhoon/level-set_MPI_GPU_3D/g_equation_solver_mpi_gpu

srun --mpi=pmix -N 2 --ntasks-per-node=4 --cpu-bind=none \
    nsys profile \
    --output=/scratch/paop41a05/juhoon/level-set_MPI_GPU_3D/log/profile_%q{SLURM_PROCID} \
    --trace=cuda,mpi,nvtx \
    $SOLVER

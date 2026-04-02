#!/bin/bash
#SBATCH -J g_equation_ncu
#SBATCH -N 1
#SBATCH -t 0:30:00
#SBATCH -p combust
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH --output=log/slurm-%j.out
#SBATCH --error=log/slurm-%j.err

source $MODULESHOME/init/bash
module purge
module load PrgEnv-nvidia
module load craype-arm-grace
module load cuda/12.8                    # ← 추가

export LD_LIBRARY_PATH=/opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3/lib:$LD_LIBRARY_PATH
export MPICH_GPU_SUPPORT_ENABLED=1       # ← 추가
export FI_MR_CACHE_MONITOR=memhooks     # ← 추가

LOG_DIR=$(pwd)/log
mkdir -p $LOG_DIR
SOLVER=$(pwd)/g_equation_solver_mpi_gpu  # ← Makefile TARGET과 동일하게

srun --mpi=cray_shasta --cpu-bind=none \
    ncu \
    -o ${LOG_DIR}/computeRHS_profile_maxreegcount_64v4_launch10_block_888 \
    --kernel-name regex:.*computeRHS.* \
    --set full \
    --import-source yes \
    --launch-count 10 \
    --target-processes all \
    --force-overwrite \
    $SOLVER

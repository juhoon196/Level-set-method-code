#!/bin/bash
#SBATCH -J g_equation
#SBATCH -N 1
#SBATCH -t 0:30:00
#SBATCH -p combust
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --hint=nomultithread
#SBATCH --output=log/slurm-%j.out
#SBATCH --error=log/slurm-%j.err

source $MODULESHOME/init/bash
module purge
module load PrgEnv-nvidia
module load craype-arm-grace
module load cuda

export LD_LIBRARY_PATH=/opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3/lib:$LD_LIBRARY_PATH

LOG_DIR=$(pwd)/log
mkdir -p $LOG_DIR
SOLVER=$(pwd)/g_equation_solver_mpi_gpu


export MPICH_GPU_SUPPORT_ENABLED=1
export FI_MR_CACHE_MONITOR=memhooks


SOLVER=$(pwd)/g_equation_solver_mpi_gpu_mpi
srun --mpi=cray_shasta --cpu-bind=none \
    nsys profile \
    --output=${LOG_DIR}/profile_mpi_v1_%q{SLURM_PROCID} \
    --trace=cuda,mpi,nvtx \
    --force-overwrite true \
    $SOLVER

SOLVER=$(pwd)/g_equation_solver_mpi_gpu_cudampi
srun --mpi=cray_shasta --cpu-bind=none \
    nsys profile \
    --output=${LOG_DIR}/profile_mpi_v2_%q{SLURM_PROCID} \
    --trace=cuda,mpi,nvtx \
    --force-overwrite true \
    $SOLVER

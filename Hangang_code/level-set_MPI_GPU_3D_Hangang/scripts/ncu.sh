#!/bin/bash
#SBATCH -J g_equation_ncu
#SBATCH -N 1
#SBATCH -t 0:30:00
#SBATCH -p gpu
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

export LD_LIBRARY_PATH=/opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3/lib:$LD_LIBRARY_PATH

mkdir -p $LOG_DIR
SOLVER=./g_equation_solver_mpi_gpu

srun --mpi=cray_shasta --cpu-bind=none \
    ncu \
    -o computeRHS_profile.64reg.v2 \
    --set full \
    --kernel-name regex:computeRHS \
    --force-overwrite \
    --launch-count 1 \
    --import-source yes \
    $SOLVER

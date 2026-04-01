#!/bin/bash
#SBATCH -J g_equation
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH -p combust 
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --hint=nomultithread
#SBATCH --output=log/slurm-%j.out
#SBATCH --error=log/slurm-%j.err

source $MODULESHOME/init/bash
module purge
module load PrgEnv-nvidia
module load craype-arm-grace

export LD_LIBRARY_PATH=/opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3/lib:$LD_LIBRARY_PATH

LOG_DIR=$(pwd)/log
mkdir -p $LOG_DIR
SOLVER=$(pwd)/g_equation_solver_mpi_gpu

srun --mpi=cray_shasta --cpu-bind=none $SOLVER

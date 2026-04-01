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

export LD_LIBRARY_PATH=/opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3/lib:$LD_LIBRARY_PATH

LOG_DIR=$(pwd)/log
mkdir -p $LOG_DIR
SOLVER=$(pwd)/g_equation_solver_mpi_gpu_v2

srun --mpi=cray_shasta --cpu-bind=none \
    ncu \
    -o ${LOG_DIR}/computeRHS_profile_maxreegcount_64_v2 \
    --kernel-name regex:.*computeRHS.* \
    --set full \
    --import-source yes \
    --launch-skip 3 --launch-count 1 \
    --target-processes all \
    --force-overwrite \
    $SOLVER

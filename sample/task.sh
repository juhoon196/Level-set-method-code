#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16           # 코드의 16스레드에 맞춤
#SBATCH --partition=normal          # 확인된 파티션 이름으로 변경
#SBATCH --mem=32G
#SBATCH --exclude=worker[06-07]
#SBATCH --output=logs/out_%j.log
#SBATCH --error=logs/err_%j.log

# 환경 설정 (module avail gcc로 확인한 버전 입력)
module purge
module load gcc/9.4.0              

EXE=$1
FREQ=$2

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Running $EXE with $FREQ Hz on $(hostname)"
$EXE $FREQ

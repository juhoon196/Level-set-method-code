source $MODULESHOME/init/bash
module purge
module load PrgEnv-nvidia
module load craype-arm-grace

export LD_LIBRARY_PATH=/opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3/lib:$LD_LIBRARY_PATH

make clean
make


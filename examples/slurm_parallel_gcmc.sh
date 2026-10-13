#!/bin/bash
# Parallel GCMC of a nanoparticle at fixed mu_O on 192-core CPU nodes.
#
#   sbatch examples/slurm_parallel_gcmc.sh
#
# Every MPI rank runs one Python process with its own copy of the potential.
# ParallelGCMC splits the ranks into N_WALKERS independent chains (set in
# examples/05_parallel_gcmc_nanoparticle.py) and the ranks of each chain
# evaluate trial moves in parallel. The rank count must be a multiple of
# N_WALKERS. Layouts for one node (192 cores):
#
#   ntasks-per-node  cpus-per-task  N_WALKERS  ranks/walker  threads/rank
#   192              1              8          24            1   (default)
#   192              1              16         12            1
#    96              2              8          12            2
#    48              4              8           6            4
#
# More ranks per walker only help while ranks/walker <~ 1/acceptance ratio:
# check "trials/round" in parallel_gcmc_summary.txt after a short test run.
# If it is well below ranks/walker, move ranks to more walkers instead.
# Multi-node: raise --nodes (and N_WALKERS); nothing else changes.
#
# Memory: each rank holds its own MACE model (a few hundred MB for
# mace-mp medium in float64), i.e. tens of GB per node at 192 ranks. Use
# fewer ranks with more threads each (rows 3-4) if the node runs short.

#SBATCH --job-name=np_gcmc_O
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=24:00:00
#SBATCH --output=np_gcmc_O.%j.log
##SBATCH --account=<your_account>
##SBATCH --partition=<cpu_partition>

set -euo pipefail

# Site specific: load the MPI that mpi4py was built against and activate
# the Python environment holding mcpy, mpi4py and mace-torch, e.g.
# module load gcc openmpi python
# source ~/venvs/mcpy/bin/activate

# One thread pool per rank, sized to its cores; without this every rank's
# torch/BLAS spawns 192 threads and the node is massively oversubscribed.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export OMP_PROC_BIND=close
export OMP_PLACES=cores

cd "$SLURM_SUBMIT_DIR"
srun --cpu-bind=cores python examples/05_parallel_gcmc_nanoparticle.py

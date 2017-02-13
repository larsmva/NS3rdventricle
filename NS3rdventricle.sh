#!/bin/bash
# Job name:
#SBATCH --job-name=kent
#
# Project:
#SBATCH --account=nn9279k
# Wall clock limit:
#SBATCH --time='16:00:00'
#
# Max memory usage per task:
#SBATCH --mem-per-cpu=3800M
#
# Number of tasks (cores):
##SBATCH --nodes=1 --ntasks=15
#SBATCH --ntasks=4
##SBATCH --hint=compute_bound
#SBATCH --cpus-per-task=1

#SBATCH --partition=long
##SBATCH --output=output.$SCRATCH 

## Set up job environment
source /cluster/bin/jobsetup

#module load gcc/4.9.2
#module load openmpi.gnu/1.8.4
#source ~oyvinev/intro/hashstack/fenics-1.5.0.abel.gnu.conf
#source ~oyvinev/fenics1.6/fenics1.6
source ~johannr/fenics-2016.2.0.abel.gnu.conf
# Expand pythonpath with locally installed packages
export PYTHONPATH=$PYTHONPATH:$HOME/.local/lib/python2.7/site-packages/

# Define what to do when job is finished (or crashes)
cleanup "cp -r $SCRATCH/results* $HOME/" 

echo "SCRATCH is $SCRATCH"
# Copy necessary files to $SCRATCH
cp NS3rdventricle.py 3rdventricle.xml.gz $SCRATCH

mpirun --bind-to none python NS3rdventricle.py 


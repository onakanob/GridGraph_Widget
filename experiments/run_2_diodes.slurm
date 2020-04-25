#!/bin/bash

## Job Name

#SBATCH --job-name=grid_runs_1_2_5_10

## Allocation Definition

## On mox and ikt, the account and partition options should be the same.
#SBATCH --account=stf
#SBATCH --partition=stf

## Resources

## Nodes
## For mox, eg: --nodes=4 \ --ntasks-per-node=28

#SBATCH --nodes=1   

## Walltime (3 hours). Do not specify a walltime substantially more than your job needs.

#SBATCH --time=24:00:00

## Memory per node. It is important to specify the memory since the default memory is very small.

## For mox, --mem may be more than 100G depending on the memory of your nodes.

## For ikt, --mem may be 58G or more depending on the memory of your nodes.

## See above section on "Specifying memory" for choices for --mem.

#SBATCH --mem=100G

## Specify the working directory for this job

#SBATCH --workdir=/gscratch/stf/onb/GridGraph/experiments

##turn on e-mail notification

#SBATCH --mail-type=ALL

#SBATCH --mail-user=onb@uw.edu

## export all your environment variables to the batch job session

#SBATCH --export=all

log_dir="2diode_1_2_5_10"
max_iters=2

python .\run_greedy_2diode.py \
       --log_dir log_dir \
       --recipe_file "../recipes/1 cm test.csv" \
       --resolutions "27" \
       --max_iters max_iters \
       --save_video

python .\run_greedy_2diode.py \
       --log_dir log_dir \
       --recipe_file "../recipes/2 cm test.csv" \
       --resolutions "46" \
       --max_iters max_iters \
       --save_video

python .\run_greedy_2diode.py \
       --log_dir log_dir \
       --recipe_file "../recipes/5 cm test.csv" \
       --resolutions "117" \
       --max_iters max_iters \
       --save_video

python .\run_greedy_2diode.py \
       --log_dir log_dir \
       --recipe_file "../recipes/10 cm test.csv" \
       --resolutions "136" \
       --max_iters max_iters \
       --save_video

#!/bin/bash
#SBATCH -J Train_RIFE                           #use -N only if you use both GPUs on the nodes, otherwise leave this line out
#SBATCH --partition zen2_0256_a40x2
#SBATCH --qos zen2_0256_a40x2
#SBATCH --gres=gpu:1                   #or --gres=gpu:1 if you only want to use half a node
#SBATCH --output=output.txt
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=justin.lelouedec@gmail.com


module purge

nvidia-smi
module load miniconda3

eval "$(conda shell.bash hook)"

conda activate gan
conda list 


python RIFE.py configs/config_rife.json

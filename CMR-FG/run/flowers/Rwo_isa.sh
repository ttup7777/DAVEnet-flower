#!/bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu
##SBATCH --nodelist=ewi1
#SBATCH --exclude=insy6,insy12
#SBATCH --workdir=/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/code/MSc/Tiant/Retrieval_v4.3/

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.5.1.10
srun -u --output=run/flowers/wo_isa.outputs sh run/flowers/wo_isa.sh
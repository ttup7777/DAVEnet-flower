#!/bin/sh
#SBATCH --partition=general --qos=long
#SBATCH --time=09:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu
##SBATCH --nodelist=ewi1
#SBATCH --exclude=insy6,insy12
#SBATCH --workdir=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/TianTian/xinsheng/Retrieval_v4.3/

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.5.1.10
srun -u --output=run/birds/full_5.outputs sh run/birds/full_5.sh
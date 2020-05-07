#!/bin/sh
#SBATCH --partition=general --qos=long
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu
#SBATCH --workdir=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/TianTian/DAVEnet-pytorch

module use /opt/insy/modulefiles
module load cuda/10.0

srun -u --output=result.outputs python3 run.py --exp-dir exp/Data-filenames.pickle/AudioModel-Davenet_ImageModel-VGG16_Optim-sgd_LR-0.001_Epochs-100 --resume 
#srun -u --output=result.outputs python3 run.py --data-train ../data/Oxford102/train/filenames.pickle --data-val ../data/Oxford102/test/filenames.pickle
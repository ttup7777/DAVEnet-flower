# DAVEnet Pytorch

Implementation in Pytorch of the DAVEnet (Deep Audio-Visual Embedding network) model, as described in

David Harwath, Adrià Recasens, Dídac Surís, Galen Chuang, Antonio Torralba, and James Glass, "Jointly Discovering Visual Objects and Spoken Words from Raw Sensory Input," ECCV 2018

## Requirements

- pytorch
- torchvision
- librosa

## Data

You will need the 102 category Flower dataset and Caltech-UCSD Birds 200


## Model Training

python3 run.py --exp-dir exp/Data-filenames.pickle/AudioModel-Davenet_ImageModel-VGG16_Optim-sgd_LR-0.001_Epochs-100 --resume 

to continue training and testing

See the run.py script for more training options.

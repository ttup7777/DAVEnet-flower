import argparse
import os
import pickle
import sys
import time
import torch
import random
import datetime
import pprint
import dateutil.tz
import numpy as np
from PIL import Image
import sys
sys.path.append("..")

from dataloaders.datasets import SpeechDataset, pad_collate 
from models import AudioModels, ImageModels, classification

from steps.traintest import train, validate, feat_extract_co, feat_extract_sne,feat_extract_sne_imgaud, feat_extract_gan
import torchvision.transforms as transforms 
from utils.config import cfg, cfg_from_file
# from retrieval_visualization import retrieval_visualization

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #'/media/shawn/data/Data/birds'
    #'/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/birds'
    # /run/user/1000/gvfs/sftp:host=sftp.tudelft.nl
   # parser.add_argument('--data_path', type = str, default='/tudelft.net/staff-bulk/ewi/insy/SpeechLab/TianTian/data/flowers/Oxford102') #
    parser.add_argument('--data_path', type = str, default='H:/staff-bulk/ewi/insy/SpeechLab/TianTian/data/birds') #
    parser.add_argument('--exp_dir', type = str, default= '')
    parser.add_argument('--save_root', type=str, default='outputs/01_Baseline/flower/full')
    parser.add_argument('--result_file',type=str,default=None)
    parser.add_argument("--resume", action="store_true", default=True,
            help="load from exp_dir if True")
    parser.add_argument("--optim", type=str, default="adam",
            help="training optimizer", choices=["sgd", "adam"])
    parser.add_argument('--batch_size', default=32, type=int,
        metavar='N', help='mini-batch size (default: 100)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_decay', default=100, type=int, metavar='LRDECAY',
        help='Divide the learning rate by 10 every lr_decay epochs')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
        metavar='W', help='weight decay (default: 1e-4)')     #5e-7
    # TRAIN.SMOOTH.GAMMA3
    parser.add_argument('--smooth_gamm3',type=float,default=10.0,help='temperature of softmax in batch loss')
    parser.add_argument('--smooth_imgatt1',type=float,default=1.0,help='smooth factor in image self attention')
    parser.add_argument('--smooth_imgatt2',type=float,default=1.0,help='smooth factor in image self attention')

    parser.add_argument("--start_epoch",type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=120,
            help="number of maximum training epochs")
    parser.add_argument("--RNN_dropout",type=float,default=0.0)
    parser.add_argument("--n_heads",type=int,default=5,help='number of attention')
    parser.add_argument('--topk',type=int,default=3,help='top k score for testing')
    parser.add_argument("--n_print_steps", type=int, default=2,
            help="number of steps to print statistics")
    parser.add_argument("--audio-model", type=str, default="Davenet",
            help="audio model architecture", choices=["Davenet"])
    parser.add_argument("--image-model", type=str, default="VGG16",
            help="image model architecture", choices=["VGG16"])
    parser.add_argument("--pretrained-image-model", action="store_true",
        dest="pretrained_image_model", help="Use an image network pretrained on ImageNet")
    parser.add_argument("--margin", type=float, default=1.0, help="Margin paramater for triplet loss")
    parser.add_argument("--simtype", type=str, default="MISA",
            help="matchmap similarity function", choices=["SISA", "MISA", "SIMA"])
    parser.add_argument('--tasks',type = str, default='extraction', help="training | extraction")

    parser.add_argument('--rnn-type',type = str, default='LSTM', help='LSTM | GRU')
    parser.add_argument('--cfg_file',type = str, default='Confg/birds_train_batch.yml',help='optional config file')
    parser.add_argument('--img_size',type = int, default=256, help = 'the size of image')

    parser.add_argument('--gpu_id',type = int, default= 0)
    parser.add_argument('--manualSeed',type=int,default= 200, help='manual seed')

    parser.add_argument('--WORKERS',type=int, default=0, help='number of workers for loading data')

    # parameters for loss function
    parser.add_argument('--loss_clss',default=None)
    parser.add_argument('--gamma_clss',type=float,default=1.0)
    # parser.add_argument('--loss_disc',default=None)
    # parser.add_argument('--gamma_disc',type=float,default=1.0)

    parser.add_argument('--image_attention',default=None)
    parser.add_argument('--audio_attention',default=None)
    
    args = parser.parse_args()

    resume = args.resume

    # print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_path != '':
        cfg.DATA_DIR = args.data_path

    if args.WORKERS != None:
        cfg.WORKERS = args.WORKERS

    if args.batch_size != None:
        cfg.TRAIN.BATCH_SIZE = args.batch_size

    if args.n_heads != None:
        cfg.CNNRNN_ATT.n_heads = args.n_heads

    if args.n_epochs !=None:
        cfg.TRAIN.MAX_EPOCH = args.n_epochs

    if args.topk != None:
        cfg.TEST.topk = args.topk

    if args.result_file != None:
        cfg.result_file = args.result_file

    if args.RNN_dropout != None:
        cfg.RNN.dropout = args.RNN_dropout

    if args.start_epoch != None:
        cfg.start_epoch = args.start_epoch
    
    if args.smooth_gamm3 != None:
        cfg.TRAIN.SMOOTH.GAMMA3 = args.smooth_gamm3
    
    if args.smooth_imgatt1 != None:
        cfg.TRAIN.SMOOTH.IMGATT = args.smooth_imgatt1
    
    if args.smooth_imgatt2 != None:
        cfg.TRAIN.SMOOTH.IMGATT2 = args.smooth_imgatt2

    if args.gamma_clss != None:
        cfg.Loss.gamma_clss = args.gamma_clss
    
    # if args.gamma_disc != None:
    #     cfg.Loss.gamma_disc = args.gamma_disc
    
    if args.loss_clss != None:
        cfg.Loss.clss = args.loss_clss
    
    # if args.loss_disc != None:
    #     cfg.Loss.disc = args.loss_disc

    if args.image_attention != None:
        cfg.image_attention = args.image_attention

    if args.audio_attention != None:
        cfg.audio_attention = args.audio_attention
    
    # print('Using config:')
    # pprint.pprint(cfg)

    cfg.exp_dir = os.path.join(args.save_root,'pre-train')
    print(cfg.TRAIN.BATCH_SIZE,cfg.TRAIN.SMOOTH.GAMMA3,cfg.image_attention,cfg.audio_attention)
    if not cfg.TRAIN.FLAG:
        args.manualSeed = 200
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.set_num_threads(4)
    if cfg.CUDA:    
        torch.cuda.manual_seed(args.manualSeed)  
        torch.cuda.manual_seed_all(args.manualSeed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def worker_init_fn(worker_id):   # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
        np.random.seed(args.manualSeed + worker_id)


    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        # bshuffle = False
        split_dir = 'test'

    # Get data loader
    imsize = args.img_size
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    image_transform_test = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize)])

    if cfg.TRAIN.MODAL == 'co-train':
        dataset = SpeechDataset(cfg.DATA_DIR, 'train',
                                img_size = imsize,
                                transform=image_transform)
        dataset_test = SpeechDataset(cfg.DATA_DIR, 'test',
                                img_size = imsize,
                                transform=image_transform_test)
        dataset_val = SpeechDataset(cfg.DATA_DIR, 'val',
                                img_size = imsize,
                                transform=image_transform_test)


        assert dataset


        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
            drop_last=True, shuffle=bshuffle,num_workers=cfg.WORKERS,worker_init_fn=worker_init_fn)
        val_loader = torch.utils.data.DataLoader(
            dataset_val, batch_size=cfg.TRAIN.BATCH_SIZE,
            drop_last=False, shuffle=False,num_workers=cfg.WORKERS,worker_init_fn=worker_init_fn) 
        test_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=cfg.TRAIN.BATCH_SIZE,
            drop_last=False, shuffle=False,num_workers=cfg.WORKERS,worker_init_fn=worker_init_fn)        

    # Dataloader for classificaiton of single modal
    elif cfg.TRAIN.MODAL == 'extraction':
        
        dataset = SpeechDataset(cfg.DATA_DIR, 'train',
                                img_size = imsize,
                                transform=image_transform)
        dataset_test = SpeechDataset(cfg.DATA_DIR, 'test',
                                img_size = imsize,
                                transform=image_transform_test)
        dataset_val = SpeechDataset(cfg.DATA_DIR, 'val',
                                img_size = imsize,
                                transform=image_transform_test)
        
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
            drop_last=False, shuffle=False,num_workers=cfg.WORKERS,collate_fn=pad_collate,worker_init_fn=worker_init_fn)
        test_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=cfg.TRAIN.BATCH_SIZE,
            drop_last=False, shuffle=False,num_workers=cfg.WORKERS,collate_fn=pad_collate,worker_init_fn=worker_init_fn)

    elif cfg.TRAIN.MODAL == 'sne':
        
        dataset_val = SpeechDataset(cfg.DATA_DIR, 'sne',
                                img_size = imsize,
                                transform=image_transform_test)
        
        val_loader = torch.utils.data.DataLoader(
            dataset_val, batch_size=cfg.TRAIN.BATCH_SIZE,
            drop_last=False, shuffle=False,num_workers=cfg.WORKERS,worker_init_fn=worker_init_fn)
    
    elif cfg.TRAIN.MODAL == 'gan':
        
        dataset_val = SpeechDataset(cfg.DATA_DIR, 'gan',
                                img_size = imsize,
                                transform=image_transform_test)
        
        val_loader = torch.utils.data.DataLoader(
            dataset_val, batch_size=cfg.TRAIN.BATCH_SIZE,
            drop_last=False, shuffle=False,num_workers=cfg.WORKERS,worker_init_fn=worker_init_fn)
       
    # class_model = classification.CLASSIFIER()
    if cfg.SPEECH.model == 'RNN':
        audio_model = AudioModels.RNN_ENCODER(cfg.SPEECH.input_dim, cfg.SPEECH.hidden_size,cfg.SPEECH.num_layers)
    elif cfg.SPEECH.model == 'CRNN':
        # audio_model = AudioModels.CNN_RNN_ENCODER()
        audio_model = AudioModels.CNN_PRNN_ENCODER(args)
    elif cfg.SPEECH.model == 'CNN':
        audio_model = AudioModels.CNN_ENCODER(cfg.SPEECH.embedding_dim)

    image_model = ImageModels.Resnet101()


    # train(audio_model, image_model,train_loader, val_loader, args)

    if cfg.TRAIN.MODAL == 'co-train':
        MODELS = [audio_model, image_model]
        # retrieval_visualization(MODELS,test_loader, args)
        train(MODELS,train_loader,val_loader,test_loader, args)
        # validate(audio_model, image_model,image_cnn,val_loader,args)
        
    if cfg.TRAIN.MODAL == 'extraction':
        feat_extract_co(audio_model,cfg.DATA_DIR,args)

    if cfg.TRAIN.MODAL == 'sne':
        feat_extract_sne_imgaud(audio_model,image_model,val_loader,args)
    if cfg.TRAIN.MODAL == 'gan':
        feat_extract_gan(audio_model, val_loader,args)
    
    


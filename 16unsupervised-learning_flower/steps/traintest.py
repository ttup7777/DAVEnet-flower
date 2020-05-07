import time
import shutil
import torch
import torch.nn as nn
import numpy as np
import pickle


from .util import *

def train(audio_model, image_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    progress = []
    best_epoch, best_acc = 0, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir
    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_acc, 
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    # create/load exp
    if args.resume:
        
        progress_pkl = "%s/progress.pkl" % exp_dir
        progress, epoch, global_step, best_epoch, best_acc = load_progress(progress_pkl)
        print("\nResume training from:")
        print("  epoch = %s" % epoch)
        print("  global_step = %s" % global_step)
        print("  best_epoch = %s" % best_epoch)
        print("  best_acc = %.4f" % best_acc)

    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model) #making the model run parallelly

    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)

    if epoch != 0:
        audio_model.load_state_dict(torch.load("%s/models/audio_model.%d.pth" % (exp_dir, epoch))) # load params and persistent buffers
        image_model.load_state_dict(torch.load("%s/models/image_model.%d.pth" % (exp_dir, epoch)))
        print("loaded parameters from epoch %d" % epoch)

    audio_model = audio_model.to(device) #cuda or cpu   #returns a new copy of audio_model
    image_model = image_model.to(device)
    # Set up the optimizer
    audio_trainables = [p for p in audio_model.parameters() if p.requires_grad]
    image_trainables = [p for p in image_model.parameters() if p.requires_grad]
    trainables = audio_trainables + image_trainables
    if args.optim == 'sgd':
       optimizer = torch.optim.SGD(trainables, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(trainables, args.lr,
                                weight_decay=args.weight_decay,
                                betas=(0.95, 0.999))
    else:
        raise ValueError('Optimizer %s is not supported' % args.optim)

    if epoch != 0:
        optimizer.load_state_dict(torch.load("%s/models/optim_state.%d.pth" % (exp_dir, epoch)))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("loaded state dict from epoch %d" % epoch)

    epoch += 1
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")

    audio_model.train() #set to train mode
    image_model.train()
    while epoch<=100:
        adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch)
        end_time = time.time()
        audio_model.train()
        image_model.train()
        for i, (image_input, audio_input, Imlabel, Audiolabel) in enumerate(train_loader): #call train data
            # measure data loading time
            data_time.update(time.time() - end_time)
            B = audio_input.size(0)

            audio_input = audio_input.to(device)
            image_input = image_input.to(device)

            optimizer.zero_grad()

            audio_output = audio_model(audio_input)
            image_output = image_model(image_input)
            
            
            loss = sampled_margin_rank_loss(image_output, audio_output,
                Imlabel, Audiolabel, margin=args.margin)

            loss.backward()
            optimizer.step()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)

            if global_step % args.n_print_steps == 0 and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss total {loss_meter.val:.4f} ({loss_meter.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        results = validate(audio_model, image_model, test_loader, args)
        
        avg_acc = ( results['A_r1'] + results['I_r1']) / 2

        torch.save(audio_model.state_dict(),
                "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
        torch.save(image_model.state_dict(),
                "%s/models/image_model.%d.pth" % (exp_dir, epoch))
        torch.save(optimizer.state_dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))
        
        if avg_acc > best_acc:
            best_epoch = epoch
            best_acc = avg_acc
            shutil.copyfile("%s/models/audio_model.%d.pth" % (exp_dir, epoch), 
                "%s/models/best_audio_model.pth" % (exp_dir))
            shutil.copyfile("%s/models/image_model.%d.pth" % (exp_dir, epoch), 
                "%s/models/best_image_model.pth" % (exp_dir))
        _save_progress()
        epoch += 1

def validate(audio_model, image_model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)
    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    # switch to evaluate mode
    image_model.eval()
    audio_model.eval()

    end = time.time()
    N_examples = val_loader.dataset.__len__()
    I_embeddings = [] 
    A_embeddings = [] 
    Il_embeddings = []
    Al_embeddings = []
    with torch.no_grad():
        for i, (image_input, audio_input, Imlabel, Audiolabel) in enumerate(val_loader):
            
            image_input,inverse = torch.unique(image_input,sorted = False,return_inverse = True, dim=0)

            perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
            inverse, perm = inverse.flip([0]), perm.flip([0])
            perm = inverse.new_empty(image_input.size(0)).scatter_(0, inverse, perm)

            image_input = image_input.to(device)
            audio_input = audio_input.to(device)

            # compute output
            image_output = image_model(image_input)
            audio_output = audio_model(audio_input)

            image_output = image_output.to('cpu').detach()
            audio_output = audio_output.to('cpu').detach()

            I_embeddings.append(image_output)
            A_embeddings.append(audio_output)
            
            Il_embeddings.append(Imlabel[perm])
            Al_embeddings.append(Audiolabel)
            
            # pooling_ratio = round(audio_input.size(-1) / audio_output.size(-1))
            # nframes.div_(pooling_ratio)

            # frame_counts.append(nframes.cpu())

            batch_time.update(time.time() - end)
            end = time.time()


        image_output = torch.cat(I_embeddings) 
        
        audio_output = torch.cat(A_embeddings) 
        Imlabel = torch.cat(Il_embeddings)
        Audiolabel = torch.cat(Al_embeddings)

        results = calc_rank_k(image_output, audio_output,  Imlabel, Audiolabel,k=1,m=50)
        A_r1 = results['A_r1']
        I_r1 = results['I_r1']
        A_ap_k = results['A_ap_k']
        I_ap_k = results['I_ap_k'] 

    

    print(' * Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} over {N:d} validation pairs'
          .format(A_r1=A_r1, I_r1=I_r1, N=N_examples), flush=True)
    print(' * Audio AP@50 {A_ap_k:.3f} Image AP@50 {I_ap_k:.3f} over {N:d} validation pairs'
          .format(A_ap_k=A_ap_k, I_ap_k=I_ap_k, N=N_examples), flush=True)

    return results

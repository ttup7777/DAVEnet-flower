import os 
import time
import shutil
import torch
import torch.nn as nn
import numpy as np
import pickle
from steps.util import *
from tqdm import tqdm
# from torch.optim.lr_scheduler import StepLR

def train(Models,train_loader,val_loader, test_loader, args):
    
    audio_model, image_model = Models[0],Models[1]
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    progress = []
    best_epoch, best_acc, best_I2A, best_A2I = 0, -np.inf,-np.inf,-np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    # exp_dir = os.path.join(args.save_root,'pre_train_without_clssLoss') 
    exp_dir = args.save_root
    save_model_dir = os.path.join(exp_dir,'models')
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
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
        audio_model = nn.DataParallel(audio_model)

    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)

   
    epoch = 55
    
    if epoch != 0:
        # audio_model.load_state_dict(torch.load("%s/models/audio_model_%d.pth" % (exp_dir,epoch)))
        # image_model.load_state_dict(torch.load("%s/models/image_model_%d.pth" % (exp_dir,epoch)))
        audio_model.load_state_dict(torch.load("%s/models/best_audio_model_%d.pth" % (exp_dir,epoch)))
        image_model.load_state_dict(torch.load("%s/models/best_image_model_%d.pth" % (exp_dir,epoch)))            
        print("loaded parameters from epoch %d" % epoch)
    

    audio_model = audio_model.to(device)    
    image_model = image_model.to(device)   
    # Set up the optimizer
    # audio_trainables = [p for p in audio_model.parameters() if p.requires_grad]
    audio_trainables = [p for p in audio_model.parameters() if p.requires_grad]
    image_trainables = [p for p in image_model.parameters() if p.requires_grad] # if p.requires_grad
   
    
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
        optimizer.load_state_dict(torch.load("%s/models/optim_state_%d.pth" % (exp_dir, epoch)))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("loaded state dict from epoch %d" % epoch)
    

    
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    # criterion = nn.MSELoss()   
    
    audio_model.train()
    image_model.train()
  
    while epoch<=cfg.TRAIN.MAX_EPOCH:              
        epoch += 1
        adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch)      
        end_time = time.time()
        audio_model.train()
        image_model.train()       
        for i, (image_input, audio_input, cls_id, key,label) in enumerate(train_loader):
                        
            # measure data loading time
            data_time.update(time.time() - end_time)
            B = audio_input.size(0)

            audio_input = audio_input.float().to(device)            
            label = label.long().to(device)            
            
            image_input = image_input.float().to(device)
            image_input = image_input.squeeze(1)            
            
            optimizer.zero_grad()  
            output = image_model(image_input)
            # image_class_output = class_model(image_output)                      
            audio_output = audio_model(audio_input)               

            loss = 0
            
            losso1,losso2 = batch_loss(output,audio_output,cls_id)
            loss = losso1+losso2
            

            loss.backward()
            optimizer.step()    
            
            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)              
            
            if i % 5 == 0:                
                # mAP = validate(audio_model, image_model, test_loader)
                print('iteration = %d | loss = %f '%(i,loss))                
            
            end_time = time.time()
            global_step += 1
            # Rank1_I2A, mAP_I2A, Rank1_A2I, mAP_A2I= validate_all(audio_model, image_model,test_loader,args) 
            # mAP = validate(audio_model, image_model, test_loader)
        # print('mpa=',map)
        # mAP = validate(audio_model, image_model, test_loader)
        # Rank1_I2A, mAP_I2A, Rank1_A2I, mAP_A2I= validate_all(audio_model, image_model,test_loader,args)   
        if epoch % 5 ==0:
            Rank1_I2A, mAP_I2A, Rank1_A2I, mAP_A2I= validate(audio_model, image_model,val_loader,args)   
            # Rank1_I2A, mAP_I2A, Rank1_A2I, mAP_A2I= validate(audio_model, image_model,test_loader,args)            
            info = ' Epoch: [{0}] Loss: {loss_meter.val:.4f}  R1_I2A: {R1_:.4f} R1_A2I: {R2_:.4f} \n \
                '.format(epoch,loss_meter=loss_meter,R1_=Rank1_I2A,R2_=Rank1_A2I)
            print (info)
            
            save_path = os.path.join(exp_dir, cfg.result_file)
            with open(save_path, "a") as file:
                file.write(info)

            avg_acc = Rank1_I2A + Rank1_A2I            
            
            # haven't use the test processing yet
            if avg_acc > best_acc and epoch>30:    #or Rank1_I2A > best_I2A or Rank1_A2I > best_A2I

                Rank1_I2A, mAP_I2A, Rank1_A2I, mAP_A2I= validate_all(audio_model, image_model,test_loader,args)            
                info = ' Epoch: [{0}] Loss: {loss_meter.val:.4f}  R1_I2A: {R1_:.4f} mAP_I2A: {mAP1_:.4f}  R1_A2I: {R2_:.4f} mAP_A2I: {mAP2_:.4f} \n \
                    '.format(epoch,loss_meter=loss_meter,R1_=Rank1_I2A,mAP1_=mAP_I2A,R2_=Rank1_A2I,mAP2_=mAP_A2I)
                print (info)
                
                save_path = os.path.join(exp_dir, cfg.result_file)
                with open(save_path, "a") as file:
                    file.write(info)
 
                torch.save(audio_model.state_dict(),
                    "%s/models/best_audio_model_%d.pth" % (exp_dir,epoch))
                torch.save(image_model.state_dict(),
                    "%s/models/best_image_model_%d.pth" % (exp_dir,epoch))               
                torch.save(optimizer.state_dict(), "%s/models/optim_state_%d.pth" % (exp_dir,epoch))
                _save_progress()
            if avg_acc > best_acc:                
                best_acc = avg_acc
            
        # if epoch%20 == 0:
        #         torch.save(audio_model.state_dict(),
        #             "%s/models/audio_model_%d.pth" % (exp_dir,epoch))
        #         torch.save(image_model.state_dict(),
        #             "%s/models/image_model_%d.pth" % (exp_dir,epoch))                
        #         torch.save(optimizer.state_dict(), "%s/models/optim_state_%d.pth" % (exp_dir,epoch))
           


# mAP: evaltuion for the embedding with a cross modal retreival task
def validate(audio_model, image_model,val_loader,args):


    exp_dir = args.save_root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()    
    
    if cfg.TRAIN.MODAL=='co-train':
        if not isinstance(image_model, torch.nn.DataParallel):
            image_model = nn.DataParallel(image_model)
        
        image_model = image_model.to(device)
        # switch to evaluate mode
        image_model.eval()
    

    # audio_model.load_state_dict(torch.load("%s/models/best_audio_model.pth" % (exp_dir)))
    # image_model.load_state_dict(torch.load("%s/models/best_image_model.pth" % (exp_dir)))    
    

    end = time.time()
    N_examples = val_loader.dataset.__len__()
    I_embeddings = [] 
    A_embeddings = [] 
    AL_embeddings = []
    frame_counts = []
    class_ids = []
    with torch.no_grad():
        
        for i, (image_input, audio_input, cls_id, key, label) in enumerate(val_loader):
            image_input = image_input.to(device)
            audio_input = audio_input.to(device)
            image_input = image_input.squeeze(1)

            audio_input = audio_input.float().to(device)
            image_input = image_input.float().to(device)            

            # compute output
            if cfg.TRAIN.MODAL == 'co-train':
                image_output = image_model(image_input)   
            else:
                image_output = image_input
            
            if cfg.SPEECH.model == 'CNN':
                audio_output = audio_model(audio_input)
            else:
                audio_output = audio_model(audio_input)
            
            
            image_output = image_output.to('cpu').detach()
            audio_output = audio_output.to('cpu').detach()           

            I_embeddings.append(image_output)
            A_embeddings.append(audio_output)            
            class_ids.append(cls_id)     
            
            batch_time.update(time.time() - end)
            end = time.time()

        image_output = torch.cat(I_embeddings)
        audio_output = torch.cat(A_embeddings)       
        cls_id = torch.cat(class_ids)     
        
        Rank1_I2A, mAP_I2A, Rank1_A2I, mAP_A2I  = retrieval_evaluation(image_output,audio_output,cls_id)        

    return Rank1_I2A, mAP_I2A, Rank1_A2I, mAP_A2I

def validate_all(audio_model, image_model,val_loader,args):
    

    exp_dir = args.save_root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()    
    
    if cfg.TRAIN.MODAL=='co-train':
        if not isinstance(image_model, torch.nn.DataParallel):
            image_model = nn.DataParallel(image_model)
        
        image_model = image_model.to(device)
        # switch to evaluate mode
        image_model.eval()
    

    # audio_model.load_state_dict(torch.load("%s/models/best_audio_model.pth" % (exp_dir)))
    # image_model.load_state_dict(torch.load("%s/models/best_image_model.pth" % (exp_dir)))    
    

    end = time.time()
    N_examples = val_loader.dataset.__len__()
    I_embeddings = [] 
    A_embeddings = [] 
    AL_embeddings = []
    frame_counts = []
    I_class_ids = []
    A_class_ids = []
    with torch.no_grad():
        
        for i, (image_input, audio_input, cls_id, key, label) in enumerate(val_loader):
            
            image_input,inverse = torch.unique(image_input,sorted = False,return_inverse = True, dim=0)
            perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
            inverse, perm = inverse.flip([0]), perm.flip([0])
            perm = inverse.new_empty(image_input.size(0)).scatter_(0, inverse, perm)
            
            image_input = image_input.to(device)
            audio_input = audio_input.to(device)
            image_input = image_input.squeeze(1)

            audio_input = audio_input.float().to(device)
            image_input = image_input.float().to(device)            

            # compute output
            if cfg.TRAIN.MODAL == 'co-train':
                image_output= image_model(image_input)   
            else:
                image_output = image_input
            
            if cfg.SPEECH.model == 'CNN':
                audio_output = audio_model(audio_input)
            else:
                audio_output = audio_model(audio_input)
            
            
            image_output = image_output.to('cpu').detach()
            audio_output = audio_output.to('cpu').detach()           

            I_embeddings.append(image_output)
            A_embeddings.append(audio_output)  
            I_class_ids.append(cls_id[perm])          
            A_class_ids.append(cls_id)     
            
            batch_time.update(time.time() - end)
            end = time.time()

        image_output = torch.cat(I_embeddings)
        audio_output = torch.cat(A_embeddings)  
        I_ids = torch.cat(I_class_ids) 
        A_ids = torch.cat(A_class_ids)     
        
        Rank1_I2A, mAP_I2A, Rank1_A2I, mAP_A2I  = retrieval_evaluation_all(image_output,audio_output,I_ids,A_ids)        

    return Rank1_I2A, mAP_I2A, Rank1_A2I, mAP_A2I




def feat_extract_co(audio_model, path,args):
    audio_model = nn.DataParallel(audio_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    exp_dir = args.save_root
    
    audio_model.load_state_dict(torch.load("%s/models/best_audio_model_85.pth" % (exp_dir)))  #best_audio_model
    

    audio_model = audio_model.to(device)  
    
    audio_model.eval()     
    
    
    # extract speech embeding of train set
    info = 'starting extract speech embedding feature of trainset \n'
    print (info)            
    save_path = os.path.join(exp_dir, 'embedding_extract.txt')
    with open(save_path, "a") as file:
        file.write(info)

    filepath = '%s/%s/filenames.pickle' % (path, 'train')
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)     #filenames中保存的形式'002.Laysan_Albatross/Laysan_Albatross_0044_784',
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))       

    if path.find('birds') != -1:
        data_dir = '%s/CUB_200_2011' % path
    else:
        data_dir = '%s/Oxford102' % path
    
    audio_feat = []
    j = 0
    for key in filenames:  
        if path.find('birds') != -1:
            audio_file = '%s/audio_mel_800/%s.npy' % (data_dir, key) 
        else:
            audio_file = '%s/audio_mel_640/%s.npy' % (data_dir, key) 
        audios = np.load(audio_file,allow_pickle=True)
        num_cap = audios.shape[0]
        if num_cap!=cfg.SPEECH.CAPTIONS_PER_IMAGE:
            print('erro with the number of captions')
            print(audio_file)
        for i in range(num_cap):
            cap = audios[i]
            cap = torch.tensor(cap)
            input_length = cap.shape[0]
            input_length  = torch.tensor(input_length)
            audio_input = cap.float().to(device)  
            audio_input = audio_input.unsqueeze(0)  
            input_length = input_length.float().to(device)        
            input_length = input_length.unsqueeze(0)
            audio_output = audio_model(audio_input)
            audio_output = audio_output.cpu().detach().numpy()
            if i == 0:
                outputs = audio_output
            else:
                outputs = np.vstack((outputs,audio_output))

        audio_feat.append(outputs)
        
        if j % 50 ==0:
            print('extracted the %ith audio feature'%j)   
        j += 1

    with open("%s/speech_embeddings_train.pickle" % (exp_dir), "wb") as f:
            pickle.dump(audio_feat, f)
    

    info = 'extracting speech embedding feature of trainset is finished \n'
    print (info)            
    save_path = os.path.join(exp_dir, 'embedding_extract.txt')
    with open(save_path, "a") as file:
        file.write(info)
    

    #extract speech embedding of test set
    save_path = os.path.join(exp_dir, 'embedding_extract.txt')
    info = 'starting extract speech embedding feature of testset \n'
    print (info)            

    with open(save_path, "a") as file:
        file.write(info)
    
    filepath = '%s/%s/filenames.pickle' % (path, 'test')
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)     #filenames中保存的形式'002.Laysan_Albatross/Laysan_Albatross_0044_784',
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))       


    # data_dir = '%s/CUB_200_2011' % path
    if path.find('birds') != -1:
        data_dir = '%s/CUB_200_2011' % path
    else:
        data_dir = '%s/Oxford102' % path
    audio_feat = []
    j = 0
    for key in filenames:
        if path.find('birds') != -1:
            audio_file = '%s/audio_mel_800/%s.npy' % (data_dir, key) 
        else:
            audio_file = '%s/audio_mel_640/%s.npy' % (data_dir, key)  
        audios = np.load(audio_file,allow_pickle=True)
        num_cap = audios.shape[0]
        if num_cap!=cfg.SPEECH.CAPTIONS_PER_IMAGE:
            print('erro with the number of captions')
            print(audio_file)
        for i in range(num_cap):
            cap = audios[i]
            cap = torch.tensor(cap)
            input_length = cap.shape[0]
            input_length  = torch.tensor(input_length)
            audio_input = cap.float().to(device)  
            audio_input = audio_input.unsqueeze(0)  
            input_length = input_length.float().to(device)        
            input_length = input_length.unsqueeze(0)
            audio_output = audio_model(audio_input)
            audio_output = audio_output.cpu().detach().numpy()
            if i == 0:
                outputs = audio_output
            else:
                outputs = np.vstack((outputs,audio_output))

        audio_feat.append(outputs)
        
        if j % 50 ==0:
            print('extracted the %ith audio feature'%j)   
        j += 1

    info = 'extracting speech embedding feature of testset is finished \n'
    print (info)            
    with open(save_path, "a") as file:
        file.write(info)
    
    with open("%s/speech_embeddings_test.pickle" % (exp_dir), "wb") as f:
            pickle.dump(audio_feat, f)
    
    info = 'speech embedding is saved \n'
    print (info)            
    with open(save_path, "a") as file:
        file.write(info)




def feat_extract_sne(audio_model, image_model,path,args):
    audio_model = nn.DataParallel(audio_model)
    image_model = nn.DataParallel(image_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    exp_dir = args.save_root
    
    audio_model.load_state_dict(torch.load("%s/models/best_audio_model.pth" % (exp_dir)))  #best_audio_model
    image_model.load_state_dict(torch.load("%s/models/best_image_model.pth" % (exp_dir)))
    

    audio_model = audio_model.to(device)  
    image_model = image_model.to(device)
    
    audio_model.eval()   
    image_model.eval()  
    
    
    # extract speech embeding of train set
    info = 'starting extract speech embedding feature of testset \n'
    print (info)            
    save_path = os.path.join(exp_dir, 'embedding_extract.txt')
    with open(save_path, "a") as file:
        file.write(info)

    filepath = '%s/%s/filenames.pickle' % (path, 't_sne')
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)     #filenames中保存的形式'002.Laysan_Albatross/Laysan_Albatross_0044_784',
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))       

    if path.find('birds') != -1:
        data_dir = '%s/CUB_200_2011' % path
    else:
        data_dir = '%s/Oxford102' % path
    
    audio_feat = []
    j = 0
    for key in filenames:    
        audio_file = '%s/audio_mel/%s.npy' % (data_dir, key) 
        audios = np.load(audio_file,allow_pickle=True)
        num_cap = audios.shape[0]
        if num_cap!=cfg.SPEECH.CAPTIONS_PER_IMAGE:
            print('erro with the number of captions')
            print(audio_file)        
        
        cap = audios[0]
        cap = torch.tensor(cap)        
        audio_input = cap.float().to(device)  
        audio_input = audio_input.unsqueeze(0)          
        audio_output = audio_model(audio_input)
        audio_output = audio_output.cpu().detach().numpy()
            

        audio_feat.append(audio_output)
        
        if j % 50 ==0:
            print('extracted the %ith audio feature'%j)   
        j += 1

    with open("%s/speech_embeddings_sne.pickle" % (exp_dir), "wb") as f:
            pickle.dump(audio_feat, f)
    

def feat_extract_sne_imgaud(audio_model, image_model,val_loader,args):
    exp_dir = args.save_root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)          
    
    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)
        
    audio_model = audio_model.to(device)
    audio_model.eval() 
    image_model = image_model.to(device)    
    image_model.eval()
    

    audio_model.load_state_dict(torch.load("%s/models/best_audio_model.pth" % (exp_dir)))
    image_model.load_state_dict(torch.load("%s/models/best_image_model.pth" % (exp_dir)))    
    

    end = time.time()
    N_examples = val_loader.dataset.__len__()
    I_embeddings = [] 
    A_embeddings = []
    class_ids = []
    
    with torch.no_grad():
        
        for i, (image_input, audio_input, cls_id, key, label) in enumerate(val_loader):            
                        
            image_input = image_input.to(device)
            audio_input = audio_input.to(device)
            image_input = image_input.squeeze(1)

            audio_input = audio_input.float().to(device)
            image_input = image_input.float().to(device)   

            image_output = image_model(image_input)   
            audio_output = audio_model(audio_input)            
            
            image_output = image_output.to('cpu').detach()
            audio_output = audio_output.to('cpu').detach()           

            I_embeddings.append(image_output)
            A_embeddings.append(audio_output)  
            class_ids.append(cls_id)  
           
            batch_time.update(time.time() - end)
            end = time.time()
            if i ==50:
                print(i)

        image_output = torch.cat(I_embeddings)
        audio_output = torch.cat(A_embeddings)  
        ids = torch.cat(class_ids)    

        with open("%s/speech_embeddings_sne.pickle" % (exp_dir), "wb") as f:
            pickle.dump(audio_output, f)       
            
        with open("%s/image_embeddings_sne.pickle" % (exp_dir), "wb") as f:
            pickle.dump(image_output, f) 
        
        with open("%s/class_ids_sne.pickle" % (exp_dir), "wb") as f:
            pickle.dump(ids, f) 
        
        

def feat_extract_gan(audio_model, val_loader,args):
    exp_dir = args.save_root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)    
    
    audio_model = audio_model.to(device)
    audio_model.eval() 
    
    audio_model.load_state_dict(torch.load("%s/models/best_audio_model.pth" % (exp_dir)))  

    end = time.time()
    N_examples = val_loader.dataset.__len__()
    I_embeddings = []    
    
    with torch.no_grad():
        
        for i, (image_input, audio_input, cls_id, key, label) in enumerate(val_loader):         
                 
            audio_input = audio_input.to(device)
            audio_input = audio_input.float().to(device)
              
            audio_output = audio_model(audio_input)
            audio_output = audio_output.to('cpu').detach() 
            A_embeddings.append(audio_output)  
            class_ids.append(cls_id)  
           
            batch_time.update(time.time() - end)
            end = time.time()
            if i ==50:
                print(i)
        
        audio_output = torch.cat(A_embeddings)  
        ids = torch.cat(class_ids)    

        with open("%s/selected_speech_embeddings_for_gan.pickle" % (exp_dir), "wb") as f:
            pickle.dump(audio_output, f)     
            
        

        
        
                

    
import math
import pickle
import numpy as np
import torch

def calc_rank_k(image_outputs, audio_outputs, nframes, Imlabel, Audiolabel, simtype='MISA',k = 1,m = 50):
    """
    Computes recall at 1
    Recall@K (or R@K) indicates the percentage of the queries
    where at least one ground-truth is retrieved among the top-K results
    """
    S = compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype=simtype)
    n_images, n_audios = S.shape
    
    A2I_match = 0
    I2A_match = 0
    A2I_scores, R_A2I_ind = S.topk(k, 0) #rows
    I2A_scores, R_I2A_ind = S.topk(k, 1) #cols
    
    for i in range(n_audios):
        A2I_match += int(sum(Audiolabel[i] == Imlabel[R_A2I_ind[:k,i]]) > 0)
    for i in range(n_images):
        I2A_match += int(sum(Imlabel[i] == Audiolabel[R_I2A_ind[i,:k]]) > 0)

    A2I_s = A2I_match * 1.0 / n_audios
    I2A_s = I2A_match * 1.0 / n_images

    _, A2I_ind = S.topk(m, 0) #rows
    _, I2A_ind = S.topk(m, 1) #cols
    ##### A2I
    A2I_ap = 1.0 * np.zeros_like(Audiolabel)
    for i in range(n_audios):
        relevant_size = sum(Imlabel == Audiolabel[i]) # a2i compute how much relevants
        hit_index = np.where(Imlabel[A2I_ind[:m,i]] == Audiolabel[i]) # a2i top k, relevant index
        precision = 1.0 * np.zeros_like(hit_index[0]) # a2i compute precision
        for j in range(hit_index[0].shape[0]):
            hitid = hit_index[0][j]
            precision[j] = sum(Imlabel[A2I_ind[:hitid+1,i]] == Audiolabel[i]) * 1.0 / (hit_index[0][j] + 1)
            
        A2I_ap[i] = np.sum(precision) * 1.0 / int(relevant_size)
        

    A2I_aps = np.mean(A2I_ap)
    #### I2A
    I2A_average_precision = 1.0 * np.zeros_like(Imlabel)

    for i in range(n_images):
        relevant_size = sum(Audiolabel == Imlabel[i]) # i2a compute how much relevants
        hit_index = np.where(Audiolabel[I2A_ind[i,:m]] == Imlabel[i]) # i2a top k, relevant index
        precision = 1.0 * np.zeros_like(hit_index[0]) # i2a compute precision
        for j in range(hit_index[0].shape[0]):
            hitid = hit_index[0][j]
            precision[j] = sum(Audiolabel[I2A_ind[i, :hitid+1]] == Imlabel[i]) * 1.0 / (hit_index[0][j] + 1)
        I2A_average_precision[i] = np.sum(precision) * 1.0 / int(relevant_size)
    I2A_aps = np.mean(I2A_average_precision)

    Results = {'A_ap_k':A2I_aps, 'I_ap_k':I2A_aps, 'A_r1':A2I_s, 'I_r1':I2A_s}
    return Results
    


def computeMatchmap(I, A):
    assert(I.dim() == 3) # 14*14*1024
    assert(A.dim() == 2) # 128*1024
    D = I.size(0)
    H = I.size(1)
    W = I.size(2)
    T = A.size(1)                                                                                                                     
    Ir = I.view(D, -1).t() #concatenate H and W, then transpose
    matchmap = torch.mm(Ir, A)
    matchmap = matchmap.view(H, W, T)  
    return matchmap

def matchmapSim(M, simtype):
    assert(M.dim() == 3)
    if simtype == 'SISA':
        return M.mean()
    elif simtype == 'MISA':
        M_maxH, _ = M.max(0)
        M_maxHW, _ = M_maxH.max(0)
        return M_maxHW.mean()
    elif simtype == 'SIMA':
        M_maxT, _ = M.max(2)
        return M_maxT.mean()
    else:
        raise ValueError

def sampled_margin_rank_loss(image_outputs, audio_outputs, nframes, Imlabel, Audiolabel, margin=1., simtype='MISA'): #loss function
    """
    Computes the triplet margin ranking loss for each anchor image/caption pair
    The impostor image/caption is randomly sampled from the minibatch
    """
    
    assert(image_outputs.dim() == 4) 
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    for i in range(n):
        I_imp_ind = i
        A_imp_ind = i
        while I_imp_ind == i and Imlabel[I_imp_ind]==Audiolabel[i]:#Imposter不同类
            I_imp_ind = np.random.randint(0, n)
        while A_imp_ind == i and Imlabel[i]==Audiolabel[A_imp_ind]:
            A_imp_ind = np.random.randint(0, n)
        nF = nframes[i]
        nFimp = nframes[A_imp_ind]
        anchorsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[i][:, 0:nF]), simtype) # anchor
        Iimpsim = matchmapSim(computeMatchmap(image_outputs[I_imp_ind], audio_outputs[i][:, 0:nF]), simtype) 
        Aimpsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[A_imp_ind][:, 0:nFimp]), simtype)
        A2I_simdif = margin + Iimpsim - anchorsim
        if (A2I_simdif.data > 0).all():
            loss = loss + A2I_simdif
        I2A_simdif = margin + Aimpsim - anchorsim
        if (I2A_simdif.data > 0).all():
            loss = loss + I2A_simdif
    loss = loss / n
    return loss

def compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype='MISA'):
    """
    Assumes image_outputs is a (batchsize, embedding_dim, rows, height) tensor
    Assumes audio_outputs is a (batchsize, embedding_dim, 1, time) tensor
    Returns similarity matrix S where images are rows and audios are along the columns
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n_images = image_outputs.size(0)
    n_audios = audio_outputs.size(0)
    S = torch.zeros(n_images, n_audios, device=image_outputs.device)
    for image_idx in range(n_images): 
            for audio_idx in range(n_audios):
                nF = max(1, nframes[audio_idx])
                S[image_idx, audio_idx] = matchmapSim(computeMatchmap(image_outputs[image_idx], audio_outputs[audio_idx][:, 0:nF]), simtype)
    return S

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay epochs"""
    lr = base_lr * (0.1 ** (epoch // lr_decay))
    for param_group in optimizer.param_groups: 
        param_group['lr'] = lr

def load_progress(prog_pkl, quiet=False):
    """
    load progress pkl file
    Args:
        prog_pkl(str): path to progress pkl file
    Return:
        progress(list):
        epoch(int):
        global_step(int):
        best_epoch(int):
        best_avg_r10(float):
    """
    def _print(msg):
        if not quiet:
            print(msg)

    with open(prog_pkl, "rb") as f:
        prog = pickle.load(f)
        epoch, global_step, best_epoch, best_avg_r10, _ = prog[-1]

    _print("\nPrevious Progress:")
    msg =  "[%5s %7s %5s %7s %6s]" % ("epoch", "step", "best_epoch", "best_avg_r10", "time")
    _print(msg)
    return prog, epoch, global_step, best_epoch, best_avg_r10

import os 
import time
import shutil
import torch
import torch.nn as nn
import numpy as np
import pickle
import torchvision.transforms as transforms 
from dataloaders.datasets import SpeechDataset, pad_collate 
from models import AudioModels, ImageModels



def test(audio_model, image_model,val_loader):
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # N_examples = val_loader.dataset.__len__()
    # I_embeddings = [] 
    # A_embeddings = [] 
    # AL_embeddings = []
    # frame_counts = []
    # I_class_ids = []
    # A_class_ids = []
    # keys = []
    # audios = []
    # audio_idx = []
    # with torch.no_grad():
        
    #     for i, (image_input, audio_input, cls_id, key, audio_file, audio_ix) in enumerate(val_loader):
    #         image_input = image_input.to(device)
    #         audio_input = audio_input.to(device)
    #         image_input = image_input.squeeze(1)

    #         audio_input = audio_input.float().to(device)
    #         image_input = image_input.float().to(device)            

    #         # compute output
    #         image_output = image_model(image_input)   
    #         audio_output = audio_model(audio_input)
            
            
    #         image_output = image_output.to('cpu').detach()
    #         audio_output = audio_output.to('cpu').detach()           

    #         I_embeddings.append(image_output)
    #         A_embeddings.append(audio_output)            
    #         I_class_ids.append(cls_id)          
    #         A_class_ids.append(cls_id)     
    #         keys.append(key)
    #         audios.append(audio_file)
    #         audio_idx.append(audio_ix)
          

    #     image_output = torch.cat(I_embeddings)
    #     audio_output = torch.cat(A_embeddings)       
    #     I_ids = torch.cat(I_class_ids) 
    #     A_ids = torch.cat(A_class_ids)    
    #     image_names = np.concatenate(keys)
    #     audio_names = np.concatenate(audios)      
    #     audio_idex = np.concatenate(audio_idx)
    dir = 'H:/staff-bulk/ewi/insy/SpeechLab/TianTian/data/intermediate_result/birds'
    image_output = torch.load(dir+'image_output.t7')
    audio_output = torch.load(dir+'audio_output.t7')       
    I_ids = torch.load(dir+'I_ids.t7') 
    A_ids = torch.load(dir+'A_ids.t7')    
    image_names = np.load(dir+'image_names.npy')
    audio_names = np.load(dir+'audio_names.npy')     
    audio_idex = np.load(dir+'audio_index.npy')
    
    Rank1_I2A, mAP_I2A, Rank1_A2I, mAP_A2I  = retrieval_visualize_all(image_output,audio_output,I_ids,A_ids,image_names,audio_names,audio_idex)       

    return Rank1_I2A, mAP_I2A, Rank1_A2I, mAP_A2I

def normalizeFeature(x):	
    
    x = x + 1e-10 # for avoid RuntimeWarning: invalid value encountered in divide\
    feature_norm = torch.sum(x**2, axis=1)**0.5 # l2-norm
    feat = x / feature_norm.unsqueeze(-1)
    return feat

def retrieval_visualize_all(image_output,audio_output,I_id,A_id,image_names,audio_names,audio_idex):      
    # using consine similarity    
    img_f = normalizeFeature(image_output)
    aud_f = normalizeFeature(audio_output) 
    S = img_f.mm(aud_f.t())
    # #for image to audio retrieval
    _, indx_I2A = torch.sort(S,dim=1,descending=True)
    
    # class_sorted_I2A = A_id[indx_I2A]
    # Correct_num_I2A = sum(class_sorted_I2A[:,0]==I_id)
    # Rank1_I2A = Correct_num_I2A*1.0/img_f.shape[0]
    visual_I2A(indx_I2A,image_names,audio_names,audio_idex,k=7)
    # mAP_I2A = AP_K(indx_I2A,I_id,A_id,image_names,audio_names,audio_idex)
    # # mAP_I2A = 0.0
    # dir ='H:/staff-bulk/ewi/insy/SpeechLab/TianTian/data/intermediate_result/birds/'
    # S =torch.load(dir+'S.t7')
     
    # for audio to image retrieval
    S_T = S.T
    _, indx_A2I = torch.sort(S_T,dim=1,descending=True)
    visual_A2I(indx_A2I,image_names,audio_names,audio_idex,k=70)
    # class_sorted_A2I = I_id[indx_A2I]
    # Correct_num_A2I = sum(class_sorted_A2I[:,0]==A_id)
    # Rank1_A2I = Correct_num_A2I*1.0/aud_f.shape[0]
    # mAP_A2I = AP_K(indx_A2I,A_id,I_id)
    # # mAP_A2I = 0.0
    # return Rank1_I2A, mAP_I2A, Rank1_A2I, mAP_A2I

def AP_K(rank, label_view1, label_view2, image_names,audio_names,audio_idex,k=50):
    
    n_probe, n_gallery = rank.shape
    match = 0
    average_precision = 1.0 * np.zeros_like(label_view1)
    for i in range(n_probe):
        relevant_size = sum(label_view2[rank[i,:k]]  == label_view1[i])
        relevant_size = max(1,relevant_size)
        hit_index = np.where(label_view2[rank[i,:k]] == label_view1[i])
        precision = 1.0 * np.zeros_like(hit_index[0])
        for j in range(hit_index[0].shape[0]):
            hitid = max(1, hit_index[0][j])
            precision[j] = sum(label_view2[rank[i, :hitid]] == label_view1[i]) * 1.0 / (hit_index[0][j]*1.0 + 1.0)
        average_precision[i] = np.sum(precision) * 1.0 / (relevant_size*1.0)

    score = np.mean(average_precision)

    return score

def visual_I2A(rank,image_names,audio_names,audio_idex,k=7):
    # img_dir = '/tudelft.net/staff-bulk/ewi/insy/SpeechLab/TianTian/data/birds/CUB_200_2011/images/'
    # txt_dir = '/tudelft.net/staff-bulk/ewi/insy/SpeechLab/TianTian/data/birds/CUB_200_2011/text/'
    img_dir = '/tudelft.net/staff-bulk/ewi/insy/SpeechLab/TianTian/data/flowers/Oxford102/images/'
    txt_dir = '/tudelft.net/staff-bulk/ewi/insy/SpeechLab/TianTian/data/flowers/Oxford102/text_c10/'
    for i in range(0,len(image_names),10):
        # img = img_dir+image_names[i]+'.jpg'
        # Reading an image in default mode 
        # image = cv.imread(path) 
  
        # cv.imshow(str(image_names[i]),img)
        print('img:'+image_names[i])
        audio = audio_names[rank[i,:k]]
        indx = audio_idex[rank[i,:k]]
        for j in range(len(audio)):
            clss = audio[j].split('/')[10]
            aud_name = audio[j].split('/')[11].split('.')[0]
            aud = txt_dir+clss+'/'+aud_name+'.txt'
            print('txt:'+clss+'/'+aud_name)
            with open(aud,'r') as f:
                txts = f.readlines()
            print(txts[indx[j]])

def visual_A2I(rank,image_names,audio_names,audio_idex,k=70):
    img_dir = '/tudelft.net/staff-bulk/ewi/insy/SpeechLab/TianTian/data/flowers/Oxford102/images/'
    txt_dir = '/tudelft.net/staff-bulk/ewi/insy/SpeechLab/TianTian/data/flowers/Oxford102/text_c10/'
    for i in range(0,len(audio_names),10):
        # img = img_dir+image_names[i]+'.jpg'
        # Reading an image in default mode 
        # image = cv.imread(path) 
  
        # cv.imshow(str(image_names[i]),img)
        clss = audio_names[i].split('/')[10]
        aud_name = audio_names[i].split('/')[11].split('.')[0]
        aud = txt_dir+clss+'/'+aud_name+'.txt'
        print('TXT:'+clss+'/'+aud_name)
        with open(aud,'r') as f:
            txts = f.readlines()
        print(txts[audio_idex[i]])
        image = image_names[rank[i,:k]]
        for j in range(0,len(image),10):
            print('image:'+image[j])
            


def retrieval_visualization(MODELS,test_loader, args):
    audio_model, image_model = MODELS[0],MODELS[1]
    audio_model = nn.DataParallel(audio_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_model = nn.DataParallel(image_model)
    exp_dir = "H:/staff-bulk/ewi/insy/SpeechLab/TianTian/xinsheng/Retrieval_v4.3/outputs/01_Baseline/birds/full_13"
    epoch = 65

    audio_model.load_state_dict(torch.load("%s/models/best_audio_model_%d.pth" % (exp_dir,epoch)))
    image_model.load_state_dict(torch.load("%s/models/best_image_model_%d.pth" % (exp_dir,epoch))) 

    audio_model = audio_model.to(device)    
    image_model = image_model.to(device)  
    audio_model.eval()  
    image_model.eval()
    test(audio_model,image_model,test_loader)

if __name__ == '__main__':

    # dir ='/tudelft.net/staff-bulk/ewi/insy/SpeechLab/TianTian/data/intermediate_result/flowers/'
    # image_output = torch.load(dir+'image_output.t7')
    # audio_output = torch.load(dir+'audio_output.t7')       
    # I_ids = torch.load(dir+'I_ids.t7') 
    # A_ids = torch.load(dir+'A_ids.t7')    
    # image_names = np.load(dir+'image_names.npy')
    # audio_names = np.load(dir+'audio_names.npy')     
    # audio_idex = np.load(dir+'audio_index.npy')
    # print("successful load!")
    # retrieval_visualize_all(image_output,audio_output,I_ids,A_ids,image_names,audio_names,audio_idex)       


    # import matplotlib.pyplot as plt
    # x = [10,11,12,13,14,15,16,17,18,19,20]#点的横坐标
    # # # #birds 
    # # k1 = [33.73,33.81,34.83,35.7,33.33,33.42,33.16,33.34,32.82,32.61,32.26]#线1的纵坐标
    # # k2 = [49.15,50.09,49.88,51.88,50.64,50.21,49.97,48.89,49.39,48.58,49.79]#线2的纵坐标
    # #flowers
    # k1 = [48.86,48.69,48.74,49.26,47.7,47.82,47.82,47.76,49.02,48.38,47.31]#线1的纵坐标
    # k2 = [59.82,60.66,61.89,63.74,61.05,62.43,63.13,62.36,62.12,61.97,61.73]#线2的纵坐标
    # def list_add(a,b):
    #     c = []
    #     for i in range(len(a)):
    #         c.append(a[i]+b[i])
    #     return c
    # k3 = list_add(k1,k2)
    
    # plt.plot(x,k1,'s-',color = 'r',label="Speech-to-Image")#s-:方形
    # plt.plot(x,k2,'o-',color = 'g',label="Image-to-Speech")#o-:圆形
    # plt.plot(x,k3,'*-',color = 'b',label="SUM")#o-:圆形
    # plt.xlabel("GAMMA")#横坐标名字
    # plt.ylabel("R@1")#纵坐标名字
    # # for a, b in zip(x, k1):
    # #     plt.text(a, b, b, ha='center', va='bottom', fontsize=8)
    # plt.legend(loc = "upper right")#图例
    # plt.grid(True)
    # plt.title("R@1 score tested on Flower dataset")
    # plt.show()

    # import matplotlib.pyplot as plt
    # x = [1,5,10,20,50,100]#点的横坐标
    # # # #birds 
    # k1 = [21.71,31.44,33.73,32.26,30.77,28.4]#线1的纵坐标
    # k2 = [29.01,45.61,49.15,49.79,42.85,42.7]#线2的纵坐标
    # def list_add(a,b):
    #     c = []
    #     for i in range(len(a)):
    #         c.append(a[i]+b[i])
    #     return c
    # k3 = list_add(k1,k2)
    
    # plt.plot(x,k1,'s-',color = 'r',label="Speech-to-Image")#s-:方形
    # plt.plot(x,k2,'o-',color = 'g',label="Image-to-Speech")#o-:圆形
    # plt.plot(x,k3,'*-',color = 'b',label="SUM")#o-:圆形
    # plt.vlines(10, 0, 100, colors = "r", linestyles = "dashed")
    # plt.vlines(20, 0, 100, colors = "r", linestyles = "dashed")
    # plt.xlabel("GAMMA")#横坐标名字
    # plt.ylabel("R@1")#纵坐标名字
    #     # for a, b in zip(x, k1):
    # #     plt.text(a, b, b, ha='center', va='bottom', fontsize=8)
    # plt.legend(loc = "lower right")#图例
    # plt.grid(True)
    # plt.title("R@1 score tested on Bird dataset")
    # plt.show()


    # from matplotlib import pyplot as plt 
    # import numpy as np  

    # # y = [7,5,2,1,3,3,2,5,1,2,3,1,3,2,2,1,2,1,3,1,3,4,2,1,1,2,3,4,1,2,2,1,2,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    # # x =np.array( [40,40,40,40,40,40,40,41,41,41,41,41,42,42,43,45,45,45,46,46,46,48,48,49,49,49,49,49,50,52,52,54,54,54,55,56,56,56,
    # # 58,58,59,59,60,61,61,62,63,63,63,65,66,66,66,67,67,67,67,71,71,75,76,78,78,82,82,82,85,85,85,85,86,87,87,91,91,
    # # 92,93,93,96,102,105,107,108,109,109,112,114,120,127,128,130,131,137,154,162,166,171,184,194,196,251,258])
    
    # plt.hist(x) 
    # plt.xlabel("Number of images/Class")#横坐标名字
    # plt.ylabel("Number of classes")#纵坐标名字
    # plt.title("Class Image Count") 

    # plt.show()

    # coding:utf-8
# coding:utf-8
    import multiprocessing
    from gensim.models import word2vec,Word2Vec

    sentences = word2vec.LineSentence("actorALL.v")#数据集原文件
    model = Word2Vec(sentences, sg=1, hs=0, iter=5, negative=5, size=2, window=2, min_count=1,workers=multiprocessing.cpu_count())
    model.wv.save_word2vec_format('actorALL.vector')#向量化之后
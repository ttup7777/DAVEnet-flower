import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.config import cfg
import math
#pyramidal RNN

def l2norm(x):
  """L2-normalize columns of x"""
  norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
  return torch.div(x, norm)


class pBLSTMLayer(nn.Module):
    def __init__(self,input_feature_dim,hidden_dim,rnn_unit='LSTM',dropout_rate=0.0):
        super(pBLSTMLayer, self).__init__()
        self.rnn_unit = getattr(nn,rnn_unit.upper())
        # feature dimension will be doubled since time resolution reduction
        self.BLSTM = self.rnn_unit(input_feature_dim*2,hidden_dim,1, bidirectional=True, 
                                   dropout=dropout_rate,batch_first=True)
    
    def forward(self,input_x):
        batch_size = input_x.size(0)
        timestep = input_x.size(1)
        feature_dim = input_x.size(2)
        # Reduce time resolution
        input_x = input_x.contiguous().view(batch_size,int(timestep/2),feature_dim*2)
        # Bidirectional RNN
        output,hidden = self.BLSTM(input_x)
        return output,hidden


# multi_head_attention  from  "Polysemous Visual-Semantic Embedding for cross-Modal Retrieval"
class MultiHeadSelfAttention(nn.Module):
  """Self-attention module by Lin, Zhouhan, et al. ICLR 2017"""

  def __init__(self, n_head, d_in, d_hidden):
    super(MultiHeadSelfAttention, self).__init__()

    self.n_head = n_head
    self.w_1 = nn.Linear(d_in, d_hidden, bias=False)
    self.w_2 = nn.Linear(d_hidden, n_head, bias=False)
    self.tanh = nn.Tanh()
    self.softmax = nn.Softmax(dim=1)
    self.init_weights()

  def init_weights(self):
    nn.init.xavier_uniform_(self.w_1.weight)
    nn.init.xavier_uniform_(self.w_2.weight)

  def forward(self, x, mask=None):
    # This expects input x to be of size (b x seqlen x d_feat)
    attn = self.w_2(self.tanh(self.w_1(x)))
    if mask is not None:
      mask = mask.repeat(self.n_head, 1, 1).permute(1,2,0)
      attn.masked_fill_(mask, -np.inf)
    attn = self.softmax(attn)

    output = torch.bmm(attn.transpose(1,2), x)
    # if output.shape[1] == 1:
    #   output = output.squeeze(1)
    return output, attn




# class for making multi headed attenders.
class multi_attention(nn.Module):
    def __init__(self, in_size, hidden_size, n_heads):
        super(multi_attention, self).__init__()
        self.att_heads = nn.ModuleList()
        for x in range(n_heads):
            self.att_heads.append(attention(in_size, hidden_size))
    def forward(self, input):
        out, self.alpha = [], []
        for head in self.att_heads:
            o = head(input)
            out.append(o) 
            # save the attention matrices to be able to use them in a loss function
            self.alpha.append(head.alpha)
        # return the resulting embedding 
        return torch.cat(out, 1)

class attention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(attention, self).__init__()
        self.hidden = nn.Linear(in_size, hidden_size)
        nn.init.orthogonal(self.hidden.weight.data)
        self.out = nn.Linear(hidden_size, in_size)
        nn.init.orthogonal(self.hidden.weight.data)
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, input):
        # calculate the attention weights
        self.alpha = self.softmax(self.out(F.tanh(self.hidden(input))))
        # apply the weights to the input and sum over all timesteps
        x = torch.sum(self.alpha * input, 1)
        # return the resulting embedding
        return x 


class CNN_ENCODER(nn.Module):
    def __init__(self, embedding_dim=2048):
        super(CNN_ENCODER, self).__init__()
        self.embedding_dim = embedding_dim
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(40,1), stride=(1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(1,11), stride=(1,1), padding=(0,5))
        self.conv3 = nn.Conv2d(256, 256, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv5 = nn.Conv2d(512, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.pool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2),padding=(0,1))
        self.gpool = nn.AvgPool2d(kernel_size = (1,64),stride=(1,1),padding =(0,0))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.batchnorm1(x)
        x = F.relu(self.conv1(x))     #128*1*2048
        x = self.pool(x)              #1024
        x = F.relu(self.conv2(x))     #256*1*1024
        x = self.pool(x)              #256*1*512
        x = F.relu(self.conv3(x))     #512*1*512
        x = self.pool(x)              #512*1*256
        x = F.relu(self.conv4(x))     #1024*1*256
        x = self.pool(x)              #1024*1*128
        x = F.relu(self.conv5(x))     #1024*1*128
        x = self.pool(x)              #1024*1*128
        x = F.relu(self.conv6(x))     #2048*1*64
        x = self.gpool(x)
        # x = x.squeeze(2)
        x = x.view(x.size(0), -1)
        return nn.functional.normalize(x,p=2,dim=1)



class RNN_ENCODER(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers ,dropout=0.0, bidirectional=True):
        super(RNN_ENCODER, self).__init__()

        self.rnn_type = cfg.rnn_type

        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(input_size, hidden_size , num_layers, batch_first=True, dropout=dropout,
                          bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout,
                          bidirectional=bidirectional)
        else:
            raise NotImplementedError
        
    def forward(self, input_x, enc_len):
        total_length = input_x.size(1)  # get the max sequence length
        # print('total_length: ' + str(total_length))
        # print('input_x.size(): ' + str(input_x.size()))
        packed_input = pack_padded_sequence(input_x, enc_len, batch_first=True)
        # print('enc_len: ' + str(enc_len))
        packed_output, hidden = self.rnn(packed_input)
        
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=total_length)
        local_emb = output.transpose(1, 2)
        global_emb = F.avg_pool1d(local_emb,kernel_size = total_length)
        global_emb = global_emb.squeeze(-1)
        # if self.rnn_type == 'LSTM':
        #     global_emb = hidden[0].transpose(0, 1).contiguous()
        # else:
        #     global_emb = hidden.transpose(0, 1).contiguous()       
        
        return  global_emb



class CNN_RNN_ENCODER(nn.Module):
    def __init__(self):
        super(CNN_RNN_ENCODER,self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=cfg.CNNRNN.in_channels,out_channels=cfg.CNNRNN.hid_channels,
                              kernel_size=cfg.CNNRNN.kernel_size,stride=cfg.CNNRNN.stride,
                              padding=cfg.CNNRNN.padding)
        self.Conv2 = nn.Conv1d(in_channels=cfg.CNNRNN.hid_channels,out_channels=cfg.CNNRNN.out_channels,
                              kernel_size=cfg.CNNRNN.kernel_size,stride=cfg.CNNRNN.stride,
                              padding=cfg.CNNRNN.padding)
        
        # self.Conv3 = nn.Conv1d(in_channels=cfg.CNNRNN.hid2_channels,out_channels=cfg.CNNRNN.out_channels,
        #                       kernel_size=cfg.CNNRNN.kernel_size,stride=cfg.CNNRNN.stride,
        #                       padding=cfg.CNNRNN.padding)
        self.bnorm1 = nn.BatchNorm1d(cfg.CNNRNN.hid_channels)
        self.bnorm2 = nn.BatchNorm1d(cfg.CNNRNN.out_channels)
        # self.bnorm3 = nn.BatchNorm1d(cfg.CNNRNN.out_channels)
        if cfg.CNNRNN.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(cfg.CNNRNN_RNN.input_size, cfg.CNNRNN_RNN.hidden_size , cfg.CNNRNN_RNN.num_layers, batch_first=True, dropout=cfg.CNNRNN_RNN.dropout,
                          bidirectional=cfg.CNNRNN_RNN.bidirectional)
        elif cfg.CNNRNN.rnn_type == 'GRU':
            self.rnn = nn.GRU(cfg.CNNRNN_RNN.input_size, cfg.CNNRNN_RNN.hidden_size , cfg.CNNRNN_RNN.num_layers, batch_first=True, dropout=cfg.CNNRNN_RNN.dropout,
                          bidirectional=cfg.CNNRNN_RNN.bidirectional)
        else:
            raise NotImplementedError

        self.att = multi_attention(in_size = cfg.CNNRNN_ATT.in_size, hidden_size = cfg.CNNRNN_ATT.hidden_size, n_heads = 1)
    def forward(self, input, l):
            input = input.transpose(2,1)
            x = self.Conv1(input)
            x = self.bnorm1(x)
            x = self.Conv2(x)
            x = self.bnorm2(x)
            # x = self.Conv3(x)
            # x = self.bnorm3(x)

            # update the lengths to compensate for the convolution subsampling
            l = [int((y-(self.Conv1.kernel_size[0]-self.Conv1.stride[0]))/self.Conv1.stride[0]) for y in l]
            l = [int((y-(self.Conv2.kernel_size[0]-self.Conv2.stride[0]))/self.Conv2.stride[0]) for y in l]
            # l = [int((y-(self.Conv3.kernel_size[0]-self.Conv3.stride[0]))/self.Conv3.stride[0]) for y in l]
            # create a packed_sequence object. The padding will be excluded from the update step
            # thereby training on the original sequence length only
            x = torch.nn.utils.rnn.pack_padded_sequence(x.transpose(2,1), l, batch_first=True)
            x, hx = self.rnn(x)
            # unpack again as at the moment only rnn layers except packed_sequence objects
            x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)

            if cfg.SPEECH.self_att:
                x = self.att(x)
            else:
                x = x.mean(dim=1)
            x = nn.functional.normalize(x, p=2, dim=1)    
            return x


class AttentionHead(nn.Module):
    """A single attention head"""
    def __init__(self, d_model, d_feature, dropout=0.1):
        super().__init__()
        # We will assume the queries, keys, and values all have the same feature size
        self.attn = ScaledDotProductAttention(dropout)
        self.query_tfm = nn.Linear(d_model, d_feature)
        self.key_tfm = nn.Linear(d_model, d_feature)
        self.value_tfm = nn.Linear(d_model, d_feature)
 
    def forward(self, queries, keys, values, mask=None):
        Q = self.query_tfm(queries) # (Batch, Seq, Feature)
        K = self.key_tfm(keys) # (Batch, Seq, Feature)
        V = self.value_tfm(values) # (Batch, Seq, Feature)
        # compute multiple attention weighted sums
        x = self.attn(Q, K, V)
        return x
 
class MultiHeadAttention(nn.Module):
    """The full multihead attention block"""
    def __init__(self, d_model, d_feature, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_feature = d_feature
        self.n_heads = n_heads
        # in practice, d_model == d_feature * n_heads
        assert d_model == d_feature * n_heads
 
        # Note that this is very inefficient:
        # I am merely implementing the heads separately because it is 
        # easier to understand this way
        self.attn_heads = nn.ModuleList([
            AttentionHead(d_model, d_feature, dropout) for _ in range(n_heads)
        ])
        self.projection = nn.Linear(d_feature * n_heads, d_model) 
     
    def forward(self, queries, keys, values, mask=None):
        x = [attn(queries, keys, values, mask=mask) # (Batch, Seq, Feature)
             for i, attn in enumerate(self.attn_heads)]
         
        # reconcatenate
        x = torch.cat(x, dim=2) # (Batch, Seq, D_Feature * n_heads)
        x = self.projection(x) # (Batch, Seq, D_Model)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, query, key, value, mask=None, dropout=None):
        d_k = key.size(-1) # get the size of the key
        scores = torch.matmul(query, key.transpose(-2,-1))/math.sqrt(d_k)
        # fill attention weights with 0s where padded
        if mask is not None: scores = scores.masked_fill(mask, 0)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
          p_attn = dropout(p_attn)
        output = torch.matmul(p_attn, value)
        return output

class CNN_PRNN_ENCODER(nn.Module):
    def __init__(self,args):
        super(CNN_PRNN_ENCODER,self).__init__()   
        self.args = args
        self.pLSTM_layer0 = pBLSTMLayer(cfg.RNN.input_size,cfg.RNN.hidden_size, rnn_unit=cfg.RNN_TYPE, dropout_rate=cfg.RNN.dropout)

        for i in range(1,cfg.RNN.num_layers):
            setattr(self, 'pLSTM_layer'+str(i), pBLSTMLayer(cfg.RNN.hidden_size*2,cfg.RNN.hidden_size, rnn_unit=cfg.RNN_TYPE, dropout_rate=cfg.RNN.dropout))

        # self.attention = MultiHeadSelfAttention(cfg.RNN_ATT.n_heads, cfg.RNN.hidden_size*2, cfg.RNN.hidden_size)   #cfg.CNNRNN_ATT.n_heads
        self.att = multi_attention(in_size = cfg.RNN.hidden_size*2, hidden_size = cfg.RNN.hidden_size, n_heads = 1)
        self.LayerNorm1 = nn.LayerNorm(40)
        self.LayerNorm2 = nn.LayerNorm(1024)
        self.fc = nn.Linear(1024,2048)
        self.bnorm = nn.BatchNorm1d(1024)
    def forward(self, input):
        # input = self.LayerNorm1(input)
        output,_  = self.pLSTM_layer0(input)
        for i in range(1,cfg.RNN.num_layers):
            output, _ = getattr(self,'pLSTM_layer'+str(i))(output)
        if cfg.audio_attention:
            # print(self.args.audio_attention)
            # print('with the audio attention')
            x = self.att(output)
        
            # att = self.attn_head(global_feature, output, output)
        
        else:
            # print('without the audio attention')
            x = output.mean(1)
        # x =  self.bnorm(x)
        # x = self.fc(x)  
        x = nn.functional.normalize(x, p=2, dim=1)  
        return x #,features,loc_feature

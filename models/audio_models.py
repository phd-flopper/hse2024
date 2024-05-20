from models.base import EncoderMaskerDecoder
import torch
import torch.nn.functional as F
from models.base import TasNetEncoder
from models.base import TasNetDecoder

from models.norm import ChannelwiseLayerNorm
from models.base import Conv1D, ConvTrans1D, ResBlock
from models.base import DualPathRNN

class SourceSeparationModel(EncoderMaskerDecoder):
    def __init__(self, encoder, e_params, separator, s_params, decoder, d_params, num_speakers=2):
        super(SourceSeparationModel, self).__init__(encoder, e_params, separator, s_params, decoder, d_params)




class SpEx_PlusDPRNN(torch.nn.Module):
    def __init__(self,num_spks,L=20,N=256,B=8,O=256,P=512,Q=3,spk_embed_dim=256,causal=False):
        super(SpEx_PlusDPRNN, self).__init__()
        self.L = L
        self.encoder = TasNetEncoder(N, L, L//2)
        self.ln = ChannelwiseLayerNorm(3 * N)
        self.proj = Conv1D(3 * N, O, 1)
        self.mask = Conv1D(O, N, 1)
        self.decoder = TasNetDecoder(N, L, L//2, bias=True)
        self.num_spks = num_spks

        self.spk_encoder = torch.nn.Sequential(
            ChannelwiseLayerNorm(3*N),
            Conv1D(3*N, O, 1),
            ResBlock(O, O),
            ResBlock(O, P),
            ResBlock(P, P),
            Conv1D(P, spk_embed_dim, 1),
        )

        self.pred_linear = torch.nn.Linear(spk_embed_dim, self.num_spks)
        self.conv1x1 = Conv1D(O+spk_embed_dim, O, 1)
        self.dprnn = DualPathRNN(256, 128, 256, num_speakers=1)
       
    

    def forward(self, x, aux, aux_len):

        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)

        w1 = F.relu(self.encoder(x))
        T = w1.shape[-1]
        xlen1 = x.shape[-1]
        xlen2 = (T - 1) * (self.L // 2) + self.L
        xlen3 = (T - 1) * (self.L // 2) + self.L
        w2 = F.relu(self.encoder(F.pad(x, (0, xlen2 - xlen1), "constant", 0)))
        w3 = F.relu(self.encoder(F.pad(x, (0, xlen3 - xlen1), "constant", 0)))

        y = self.ln(torch.cat([w1, w2, w3], 1))
        y = self.proj(y)
        
        # speaker encoder (share params from speech encoder)
        aux_w1 = F.relu(self.encoder(aux))
        aux_T_shape = aux_w1.shape[-1]
        aux_len1 = aux.shape[-1]
        aux_len2 = (aux_T_shape - 1) * (self.L // 2) + self.L
        aux_len3 = (aux_T_shape - 1) * (self.L // 2) + self.L
        aux_w2 = F.relu(self.encoder(F.pad(aux, (0, aux_len2 - aux_len1), "constant", 0)))
        aux_w3 = F.relu(self.encoder(F.pad(aux, (0, aux_len3 - aux_len1), "constant", 0)))

        aux = self.spk_encoder(torch.cat([aux_w1, aux_w2, aux_w3], 1))
        aux_T = (aux_len - self.L) // (self.L // 2) + 1
        aux_T = ((aux_T // 3) // 3) // 3
        aux = (torch.sum(aux, -1)/aux_T).view(-1,1).transpose(0, 1).float()
        
        #print(y.size())
        T = y.shape[-1]
        _aux = torch.unsqueeze(aux, -1)
        _aux = _aux.repeat(1,1,T)
        y = torch.cat([y, _aux], 1)

        y = self.conv1x1(y)
        y = self.dprnn(y)
        y = y.squeeze(dim=1)


        m1 = F.relu(self.mask(y))
        m2 = F.relu(self.mask(y))
        m3 = F.relu(self.mask(y))
        S1 = w1 * m1
        S2 = w2 * m2
        S3 = w3 * m3

        return self.decoder(S1), self.decoder(S2)[:, :xlen1], self.decoder(S3)[:, :xlen1], self.pred_linear(aux)
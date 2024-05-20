import torch

class EncoderMaskerDecoder(torch.nn.Module):
    def __init__(self, encoder, e_params, separator, s_params, decoder, d_params, num_speakers=2):
        '''
        A base Encoder-Separator-Decoder model for BSS.
        Input
        callable encoder --- Encoder to use
        dict e_params --- params of the passed Encoder
        callable separator --- Masking model to use
        dict s_params --- params of the passed Masking model
        callable decoder --- Decoder to use
        dict d_params --- params of the passed Decoder
        int num_speakers --- number of distinct speakers in the mix (default is 2)
        '''
        super(EncoderMaskerDecoder, self).__init__()
        self.encoder = encoder(**e_params)
        self.separator = separator(**s_params)
        self.decoder = decoder(**d_params)
        self.num_speakers = num_speakers

    def forward(self, x):
        x = self.encoder(x)
        x = self.separator(x)
        output = [self.decoder(x[i]) for i in range(self.num_speakers)]
        return output


class TasNetEncoder(torch.nn.Module):
    def __init__(self, enc_dim, win=2, stride=1, bias=False):
        '''
        A TasNet encoder. Wrapped in a class just in case.
        Input
        int enc_dim --- number of channels produced by the convolution
        int win --- size of convolving kernel (default is 2)
        int stride --- stride of the convolution (default is win // 2 == 1)
        bool bias --- if True, adds a learnable bias to the output (default False)
        '''
        super(TasNetEncoder, self).__init__()
        self.encoder = torch.nn.Conv1d(1, enc_dim, win, bias=bias, stride=stride)
    
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        return self.encoder(x)


class TasNetDecoder(torch.nn.Module):
    def __init__(self, enc_dim, win=2, stride=1, bias=False):
        '''
        A TasNet decoder. Wrapped in a class just in case.
        Input
        int enc_dim --- number of channels in the input, i.e. in the output of Masking net
        int win --- size of convolving kernel (default is 2)
        int stride --- stride of the convolution (default is win // 2 == 1)
        bool bias --- if True, adds a learnable bias to the output (default False)
        '''
        super(TasNetDecoder, self).__init__()
        self.decoder = torch.nn.ConvTranspose1d(enc_dim, 1, win, bias=bias, stride=stride)
    
    def forward(self, x):
        x = self.decoder(x)
        if torch.squeeze(x).dim() == 1:
            return torch.squeeze(x, dim=1)
        return torch.squeeze(x)


class BaseSingleRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True):
        '''
        Base class for different RNN types as in https://github.com/yluo42/TAC/blob/master/utility/models.py
        Input
        int input_size --- number of expected features in the input
        int hidden_size --- number of features in the hidden state
        int num_layers --- number of recurrent layers
        bool bidirectional --- if True, becomes bidirectional (default True)
        '''
            #currently unused
        super(BaseSingleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        pass

    def forward(self, x):
        pass


class DualPathRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
                 RNN=torch.nn.LSTM, dropout=0, bidirectional=True, 
                 num_layers=6, num_rnn_layers=1, segment_size=250, num_speakers=2, as_separator=True):
        '''
        The DPRNN model proposed in https://arxiv.org/pdf/1910.06379.
        Input
        int input_size --- number of input channels
        int hidden_size --- number of features in the hidden state
        int output_size --- number of output channels
        callable RNN --- RNN achitecture to use (default is pytorch implementation of LSTM)
                                            (can be a custom RNN inherited from BaseSingleRNN or torch.nn.Module)
        float dropout [0; 1] --- dropout ratio (if non-zero, introduces a Dropout layer; default is 0)
        bool bidirectional --- whether the rnn layers are bidirectional (default is True)
        int num_layers --- number of DPRNN blocks in the stack (default is 2)
        int num_rnn_layers --- number of layers used in RNN (default is 1)
                                            (note: unrelated to the number of DPRNN blocks)
        int segment_size --- length K of chunk for SEGMENTATION stage (default is 250, 
                                                                       optimal is ~\sqrt(2L), where L is input length)
        int num_speakers --- number of distinct speakers in the mix (default is 2)
        bool as_separator --- whether the model is used as a separator
        '''
        super(DualPathRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_rnn_layers = num_rnn_layers
        self.segment_size = segment_size
        self.num_speakers = num_speakers

        self.bottleneck = torch.nn.Conv1d(input_size, output_size, 1, bias=False)
        self.norm = torch.nn.GroupNorm(1, input_size, eps=1e-8)

        net = []
        for i in range(self.num_layers):
            net.append(DPRNNBlock(self.output_size, self.hidden_size, dropout=dropout, bidirectional=bidirectional))
        self.dprnn_blocks = torch.nn.Sequential(*net)

        self.conv2d = torch.nn.Conv2d(output_size, output_size*num_speakers, kernel_size=1)

        self.mask = torch.nn.Conv1d(output_size, input_size, 1, bias=False)
        self.prelu = torch.nn.PReLU()
        self.activation = torch.nn.ReLU()

        self.conv1d = torch.nn.Conv1d(output_size, output_size, 1)
        self.output = torch.nn.Sequential(self.conv1d, torch.nn.Tanh())
        self.output_gate = torch.nn.Sequential(self.conv1d, torch.nn.Sigmoid())

    def forward(self, x):
        '''
        B --- batch size
        N --- num features
        L --- input frames
        K --- chunk size
        S --- number of chunks
        input shape [B, N, L] --> output shape [B, N, K, S]
        '''
        x = self.norm(x)
        x = self.bottleneck(x)

        x, gap = self.Segmentation(x, self.segment_size)

        x = self.dprnn_blocks(x)

        x = self.prelu(x)
        x = self.conv2d(x)

        B, _, K, S = x.shape
        x = x.view(B*self.num_speakers,-1, K, S)

        x = self.OverlapAdd(x, gap)
        x = self.prelu(x)

        x = self.output(x) * self.output_gate(x)
        x = self.mask(x)
        x = self.activation(x)

        _, N, L = x.shape

        x = x.view(self.num_speakers, B, N, L)

        return x

    def padding(self, x, K):
        B, N, L = x.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(x.type())
            x = torch.cat([x, pad], dim=2)
        _pad = torch.Tensor(torch.zeros(B, N, P)).type(x.type())
        x = torch.cat([_pad, x, _pad], dim=2)

        return x, gap
        
    def Segmentation(self, x, K):
        '''
        The segmentation stage of the DPRNN.
        B --- batch size
        N --- num features
        L --- input frames
        K --- chunk size
        S --- number of chunks
        input shape [B, N, L] --> output shape [B, N, K, S]
        '''
        B, N, L = x.shape
        P = K // 2
        x, gap = self.padding(x, K)
        input1 = x[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = x[:, :, P:].contiguous().view(B, N, -1, K)
        x = torch.cat([input1, input2], dim=3).view(B, N, -1, K).transpose(2, 3)

        return x.contiguous(), gap

    def OverlapAdd(self, x, gap):
        '''
        The overlap-add stage of the DPRNN.
        B --- batch size
        N --- num features
        L --- input frames
        K --- chunk size
        S --- number of chunks
        input shape [B, N, K, S] --> output shape [B, N, L]
        '''
        B, N, K, S = x.shape
        P = K // 2
        x = x.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = x[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = x[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        x = input1 + input2
        if gap > 0:
            x = x[:, :, :-gap]

        return x


class DPRNNBlock(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0, 
                 num_rnn_layers=1, bidirectional=True, RNN=torch.nn.LSTM):
        '''
        A single DPRNN block.
        Input
        int input_size --- number of input channels
        int hidden_size --- number of features in the hidden state
        float dropout [0; 1] --- dropout ratio (if non-zero, introduces a Dropout layer; default is 0)
        int num_rnn_layers --- number of layers used in RNN (default is 1)
                                            (note: unrelated to the number of DPRNN blocks)
        bool bidirectional --- whether the rnn layers are bidirectional (default is True)
        callable RNN --- RNN achitecture to use (default is pytorch implementation of LSTM)
                                            (can be a custom RNN inherited from BaseSingleRNN or torch.nn.Module)
        '''
        super(DPRNNBlock, self).__init__()

        self.intra_rnn = RNN(
            input_size, 
            hidden_size, 
            num_rnn_layers, 
            batch_first=True, 
            dropout=dropout, 
            bidirectional=True #intra-chunk rnn MUST be bidirectional
        )

        self.intra_linear = torch.nn.Linear(
            hidden_size * 2, #2 is number of directions
            input_size
        )

        self.intra_norm = torch.nn.GroupNorm(
            1, 
            input_size, 
            eps=1e-8
        )

        self.inter_rnn = RNN(
            input_size,
            hidden_size,
            num_rnn_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional #inter-chunk rnn can be unidirectional
        )

        self.inter_linear = torch.nn.Linear(
            hidden_size * (1 + int(bidirectional)), 
            input_size
        )

        self.inter_norm = torch.nn.GroupNorm(
            1, 
            input_size, 
            eps=1e-8
        )

    def forward(self, x):
        '''
        B --- batch size
        N --- num features
        K --- chunk size
        S --- number of chunks
        input shape [B, N, K, S] --> output shape [B, N, K, S]
        '''
        B, N, K, S = x.size()

        intra_output = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        intra_output, _ = self.intra_rnn(intra_output)
        intra_output = self.intra_linear(intra_output)
        intra_output = intra_output.reshape(B, S, K, N).transpose(1, -1)
        intra_output = self.intra_norm(intra_output)


        intra_output = intra_output + x


        inter_output = intra_output.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        inter_output, _ = self.inter_rnn(inter_output)
        inter_output = self.inter_linear(inter_output)
        inter_output = inter_output.view(B, K, S, N).permute(0, 3, 1, 2).contiguous()
        inter_output = self.inter_norm(inter_output)

        output = inter_output + intra_output

        return output


class SpeakerEmbeddedDPRNNBlock(torch.nn.Module):
    def __init__(self, input_size, hidden_size, spk_embedding=100, dropout=0, 
                 num_rnn_layers=1, bidirectional=True, RNN=torch.nn.LSTM):
        '''
        A single DPRNN block.
        Input
        int input_size --- number of input channels
        int hidden_size --- number of features in the hidden state
        float dropout [0; 1] --- dropout ratio (if non-zero, introduces a Dropout layer; default is 0)
        int num_rnn_layers --- number of layers used in RNN (default is 1)
                                            (note: unrelated to the number of DPRNN blocks)
        bool bidirectional --- whether the rnn layers are bidirectional (default is True)
        callable RNN --- RNN achitecture to use (default is pytorch implementation of LSTM)
                                            (can be a custom RNN inherited from BaseSingleRNN or torch.nn.Module)
        '''
        super(DPRNNBlock, self).__init__()

        self.intra_rnn = RNN(
            input_size, 
            hidden_size, 
            num_rnn_layers, 
            batch_first=True, 
            dropout=dropout, 
            bidirectional=True #intra-chunk rnn MUST be bidirectional
        )

        self.intra_linear = torch.nn.Linear(
            hidden_size * 2, #2 is number of directions
            input_size
        )

        self.intra_norm = torch.nn.GroupNorm(
            1, 
            input_size, 
            eps=1e-8
        )

        self.inter_rnn = RNN(
            input_size,
            hidden_size,
            num_rnn_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional #inter-chunk rnn can be unidirectional
        )

        self.inter_linear = torch.nn.Linear(
            hidden_size * (1 + int(bidirectional)), 
            input_size
        )

        self.inter_norm = torch.nn.GroupNorm(
            1, 
            input_size, 
            eps=1e-8
        )

    def forward(self, x):
        '''
        B --- batch size
        N --- num features
        K --- chunk size
        S --- number of chunks
        input shape [B, N, K, S] --> output shape [B, N, K, S]
        '''
        B, N, K, S = x.size()

        intra_output = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        intra_output, _ = self.intra_rnn(intra_output)
        intra_output = self.intra_linear(intra_output)
        intra_output = intra_output.reshape(B, S, K, N).transpose(1, -1)
        intra_output = self.intra_norm(intra_output)


        intra_output = intra_output + x


        inter_output = intra_output.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        inter_output, _ = self.inter_rnn(inter_output)
        inter_output = self.inter_linear(inter_output)
        inter_output = inter_output.view(B, K, S, N).permute(0, 3, 1, 2).contiguous()
        inter_output = self.inter_norm(inter_output)

        output = inter_output + intra_output

        return output


class Conv1D(torch.nn.Conv1d):
    #https://github.com/xuchenglin28/speaker_extraction_SpEx
    """
    1D Conv based on nn.Conv1d for 2D or 3D tensor
    Input: 2D or 3D tensor with [N, L_in] or [N, C_in, L_in]
    Output: Default 3D tensor with [N, C_out, L_out]
            If C_out=1 and squeeze is true, return 2D tensor
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class ConvTrans1D(torch.nn.ConvTranspose1d):
    #https://github.com/xuchenglin28/speaker_extraction_SpEx
    """
    1D Transposed Conv based on nn.ConvTranspose1d for 2D or 3D tensor
    Input: 2D or 3D tensor with [N, L_in] or [N, C_in, L_in]
    Output: 2D tensor with [N, L_out]
    """

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        return torch.squeeze(x, 1)


class ResBlock(torch.nn.Module):
    #https://github.com/xuchenglin28/speaker_extraction_SpEx
    """
    Resnet block for speaker encoder to obtain speaker embedding
    ref to 
        https://github.com/fatchord/WaveRNN/blob/master/models/fatchord_version.py
        and
        https://github.com/Jungjee/RawNet/blob/master/PyTorch/model_RawNet.py
    """
    def __init__(self, in_dims, out_dims):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        self.conv2 = torch.nn.Conv1d(out_dims, out_dims, kernel_size=1, bias=False)
        self.batch_norm1 = torch.nn.BatchNorm1d(out_dims)
        self.batch_norm2 = torch.nn.BatchNorm1d(out_dims)
        self.prelu1 = torch.nn.PReLU()
        self.prelu2 = torch.nn.PReLU()
        self.maxpool = torch.nn.MaxPool1d(3)
        if in_dims != out_dims:
            self.downsample = True
            self.conv_downsample = torch.nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        else:
            self.downsample = False

    def forward(self, x):
        y = self.conv1(x)
        y = self.batch_norm1(y)
        y = self.prelu1(y)
        y = self.conv2(y)
        y = self.batch_norm2(y)
        if self.downsample:
            y += self.conv_downsample(x)
        else:
            y += x
        y = self.prelu2(y)
        return self.maxpool(y)
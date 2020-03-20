import torch
import pyro
import pyro.distributions as dist
from model.seq2seq import Encoder, Decoder
from const import *

class AutoEncoderHandler():
    def __init__(self, input: list, hidden_sz: int, output: list, embed_sz: int = 4, num_layers: int = 4):
        super(AutoEncoderHandler, self).__init__()
        self.input = input
        self.output = output
        self.input_sz = len(input)
        self.output_sz = len(output)
        self.hidden_sz = hidden_sz
        self.embed_sz = embed_sz
        self.num_layers = num_layers
        self.encoder = Encoder(self.input_sz, self.hidden_sz, self.embed_sz, num_layers)
        self.decoder = Decoder(self.output_sz, self.hidden_sz, self.output_sz, self.embed_sz, num_layers)
    
    def encode_decode(self, input: torch.Tensor, address: str, break_on_eos: bool = True):
        hidden = None
        input_len = len(input)
        for i in range(len(input)):
            _, hidden = self.encoder.forward(input[i].unsqueeze(0), hidden)
        
        input = torch.LongTensor([self.output.index(SOS)]).to(DEVICE)
        sample = self.output.index(SOS)
        ret = []
        samples = []

        if break_on_eos:    
            while sample != self.output.index(EOS):
                output, hidden = self.decoder.forward(input, hidden)
                sample = pyro.sample(f"{address}_{i}", dist.Categorical(output)).item()
                ret.append(self.output[sample])
                samples.append(sample)
                input = torch.LongTensor(sample).to(DEVICE)
        else:
            for i in range(input_len):
                output, hidden = self.decoder.forward(input, hidden)
                sample = pyro.sample(f"{address}_{i}", dist.Categorical(output)).item()
                ret.append(self.output[sample])
                samples.append(sample)
                input = torch.LongTensor([sample]).to(DEVICE)
        
        return samples, ret

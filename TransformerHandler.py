import torch

from const import *
from model.Transformer import Transformer


class TransformerHandler():
    def __init__(self, encoder_vocab: list, decoder_vocab: list, decoder_sos_idx: int, decoder_eos_idx: int = None):
        super(TransformerHandler, self).__init__()
        self.input = encoder_vocab
        self.output = decoder_vocab
        self.encoder_dim = len(encoder_vocab)
        self.decoder_dim = len(decoder_vocab)
        self.decoder_sos_idx = decoder_sos_idx
        self.decoder_pad_idx = decoder_vocab.index(PAD)
        self.encoder_pad_idx = encoder_vocab.index(PAD)

        if decoder_eos_idx is None:
            self.decoder_eos_idx = decoder_vocab.index(EOS)
        else:
            self.decoder_eos_idx = decoder_eos_idx
        self.transformer = Transformer(self.encoder_dim, self.decoder_dim, self.encoder_pad_idx, self.decoder_pad_idx)

    def forward(self, src: torch.Tensor, trg: torch.Tensor):
        src_mask = self.get_pad_mask(src, self.encoder_pad_idx)
        trg_mask = self.get_pad_mask(trg, self.decoder_pad_idx)
        output = self.transformer.forward(src.unsqueeze(1), trg.unsqueeze(1), src_mask, trg_mask)

        return output

    def get_pad_mask(self, seq, pad_idx):
        return (seq != pad_idx)

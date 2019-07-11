# encoding: utf-8

from torch import nn

# switch the comment between the following two lines to choose standard decoder or average decoder
from Transference.Decoder import Decoder
# import Encoder and Decoder from transformer.DREncoder and transformer.DRDecoder/transformer.DRAvgDecoder to enable feature combination between layers
from Transference.PEncoder import PEncoder


# from transformer.AvgDecoder import Decoder


class NMT(nn.Module):

    # isize: size of word embedding
    # num_src_word: number of words for enc_{src}
    # num_tgt_word/: number of words for enc_{src->mt}

    # num_tgt_word: number of words for Decoder
    # num_src_layer: number of encoder layers
    # num_mt_layer: number of encoder layers
    # num_pe_layer: number of encoder layers
    # fhsize: number of hidden units for PositionwiseFeedForward
    # attn_drop: dropout for MultiHeadAttention
    # global_emb: Sharing the embedding between encoder and decoder, which means you should have a same vocabulary for source and target language
    # num_head: number of heads in MultiHeadAttention
    # xseql: maxmimum length of sequence
    # ahsize: number of hidden units for MultiHeadAttention

    def __init__(self, isize, num_src_word, num_tgt_word, num_src_layer, num_mt_layer, num_pe_layer, fhsize=None,
                 dropout=0.0, attn_drop=0.0, global_emb=False,
                 num_head=8, xseql=512, ahsize=None, norm_output=True, bindDecoderEmb=False, forbidden_index=None):
        super(NMT, self).__init__()

        self.enc = PEncoder(isize, num_src_word, num_tgt_word, num_src_layer, num_mt_layer, fhsize, dropout, attn_drop,
                            num_head, xseql, ahsize,
                            norm_output, global_emb)

        emb_w = self.enc.enc2.wemb.weight

        self.dec = Decoder(isize, num_tgt_word, num_pe_layer, fhsize, dropout, attn_drop, emb_w, num_head, xseql,
                           ahsize,
                           norm_output, bindDecoderEmb, forbidden_index)

    # inpute: source sentences from encoder (bsize, seql)
    # inputo: decoded translation (bsize, nquery)
    # mask: user specified mask, otherwise it will be:
    #	inpute.eq(0).unsqueeze(1)

    def forward(self, inpute, inputo, stdo, src_mask=None, tgt_mask=None):
        _src_mask = inpute.eq(0).unsqueeze(1) if src_mask is None else src_mask
        _tgt_mask = inputo.eq(0).unsqueeze(1) if tgt_mask is None else tgt_mask

        return self.dec(self.enc(inpute, inputo, _src_mask, _tgt_mask), stdo, _tgt_mask)

    # inpute: source sentences from encoder (bsize, seql)
    # beam_size: the beam size for beam search
    # max_len: maximum length to generate

    def decode(self, inpute, inputo, beam_size=1, max_len=None, length_penalty=0.6):
        mask = inpute.eq(0).unsqueeze(1)
        tgt_mask = inputo.eq(0).unsqueeze(1)

        _max_len = inpute.size(1) + max(64, inpute.size(1) // 4) if max_len is None else max_len

        return self.dec.decode(self.enc(inpute, inputo, mask, tgt_mask), tgt_mask, beam_size, _max_len, length_penalty)

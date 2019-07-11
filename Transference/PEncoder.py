# encoding: utf-8

from Transference.Encoder import Encoder as EncoderBase
from modules import *


class DecoderLayer(nn.Module):

    # isize: input size
    # fhsize: hidden size of PositionwiseFeedForward
    # attn_drop: dropout for MultiHeadAttention
    # num_head: number of heads in MultiHeadAttention
    # ahsize: hidden size of MultiHeadAttention

    def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, ahsize=None):

        super(DecoderLayer, self).__init__()

        _ahsize = isize if ahsize is None else ahsize

        _fhsize = _ahsize * 4 if fhsize is None else fhsize

        self.self_attn = SelfAttn(isize, _ahsize, isize, num_head, dropout=attn_drop)  # enc_mt
        self.cross_attn = CrossAttn(isize, _ahsize, isize, num_head,
                                    dropout=attn_drop)  # May be idea bidirectional cross-attention

        self.ff = PositionwiseFF(isize, _fhsize, dropout, True)

        self.layer_normer1 = nn.LayerNorm(isize, eps=1e-06)
        self.layer_normer2 = nn.LayerNorm(isize, eps=1e-06)

        if dropout > 0:
            self.d1 = nn.Dropout(dropout, inplace=True)
            self.d2 = nn.Dropout(dropout, inplace=True)
        else:
            self.d1 = None
            self.d2 = None

    # inpute: encoded representation from encoder (bsize, seql, isize)
    # inputo: embedding of decoded translation (bsize, nquery, isize)
    # src_pad_mask: mask for given encoding source sentence (bsize, nquery, seql), see Encoder, expanded after generated with:
    #	src_pad_mask = input.eq(0).unsqueeze(1)
    # tgt_pad_mask: mask to hide the future input
    # query_unit: single query to decode, used to support decoding for given step

    def forward(self, inpute, inputo, src_pad_mask=None, tgt_pad_mask=None):

        _inputo = self.layer_normer1(inputo)

        context = self.self_attn(_inputo, mask=tgt_pad_mask)

        if self.d1 is not None:
            context = self.d1(context)

        context = context + inputo

        _context = self.layer_normer2(context)
        _context = self.cross_attn(_context, inpute, mask=src_pad_mask)

        if self.d2 is not None:
            _context = self.d2(_context)

        context = context + _context

        _context = self.ff(context)

        return _context + context


class Decoder(nn.Module):

    # isize: size of word embedding
    # nwd: number of words
    # num_layer: number of encoder layers
    # fhsize: number of hidden units for PositionwiseFeedForward
    # attn_drop: dropout for MultiHeadAttention
    # emb_w: weight for embedding. Use only when the encoder and decoder share a same dictionary
    # num_head: number of heads in MultiHeadAttention
    # xseql: maxmimum length of sequence
    # ahsize: number of hidden units for MultiHeadAttention
    # bindemb: bind embedding and classifier weight

    def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, emb_w=None, num_head=8,
                 xseql=512, ahsize=None, norm_output=True):

        super(Decoder, self).__init__()

        _ahsize = isize if ahsize is None else ahsize

        _fhsize = _ahsize if fhsize is None else fhsize

        self.drop = nn.Dropout(dropout) if dropout > 0.0 else None

        self.xseql = xseql
        self.register_buffer('mask', torch.triu(torch.ones(xseql, xseql, dtype=torch.uint8), 1).unsqueeze(
            0))  # is it really working here?

        self.wemb = nn.Embedding(nwd, isize, padding_idx=0)
        if emb_w is not None:
            self.wemb.weight = emb_w

        self.pemb = PositionalEmb(isize, xseql, 0, 0)
        self.nets = nn.ModuleList(
            [DecoderLayer(isize, _fhsize, dropout, attn_drop, num_head, _ahsize) for i in range(num_layer)])

        self.out_normer = nn.LayerNorm(isize, eps=1e-06) if norm_output else None

    # inpute: encoded representation from encoder (bsize, seql, isize)
    # inputo: decoded translation (bsize, nquery)
    # src_pad_mask: mask for given encoding source sentence (bsize, 1, seql), see Encoder, generated with:
    #	src_pad_mask = input.eq(0).unsqueeze(1)

    def forward(self, inpute, inputo, src_pad_mask=None, tgt_pad_mask=None):

        bsize, nquery = inputo.size()

        out = self.wemb(inputo)

        out = out * sqrt(out.size(-1)) + self.pemb(inputo, expand=False)

        if self.drop is not None:
            out = self.drop(out)

        for net in self.nets:
            out = net(inpute, out, src_pad_mask, tgt_pad_mask)

        if self.out_normer is not None:
            out = self.out_normer(out)

        return out


class PEncoder(nn.Module):

    # isize: size of word embedding
    # nwd: number of words
    # num_layer: number of encoder layers
    # fhsize: number of hidden units for PositionwiseFeedForward
    # attn_drop: dropout for MultiHeadAttention
    # num_head: number of heads in MultiHeadAttention
    # xseql: maxmimum length of sequence
    # ahsize: number of hidden units for MultiHeadAttention

    def __init__(self, isize, nwd, nwdt, num_src_layer, num_mt_layer, fhsize=None, dropout=0.0, attn_drop=0.0,
                 num_head=8, xseql=512,
                 ahsize=None, norm_output=True, global_emb=False):
        super(PEncoder, self).__init__()

        self.enc1 = EncoderBase(isize, nwd, num_src_layer, fhsize, dropout, attn_drop, num_head, xseql, ahsize,
                                norm_output)

        emb_w = self.enc1.wemb.weight if global_emb else None

        self.enc2 = Decoder(isize, nwdt, num_mt_layer, fhsize, dropout, attn_drop, emb_w, num_head, xseql, ahsize,
                            norm_output)

    # inputs: (bsize, seql)
    # mask: (bsize, 1, seql), generated with:
    #	mask = inputs.eq(0).unsqueeze(1)

    def forward(self, inpute, inputo, src_mask=None, tgt_mask=None):
        return self.enc2(self.enc1(inpute, src_mask), inputo, src_mask, tgt_mask)

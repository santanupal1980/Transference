# encoding: utf-8

import sys

import h5py
import numpy

has_unk = True


def list_reader(fname):
    def clear_list(lin):
        rs = []
        for tmpu in lin:
            if tmpu:
                rs.append(tmpu)
        return rs

    with open(fname, "rb") as frd:
        for line in frd:
            tmp = line.strip()
            if tmp:
                tmp = clear_list(tmp.decode("utf-8").split())
                yield tmp


def line_reader(fname):
    with open(fname, "rb") as frd:
        for line in frd:
            tmp = line.strip()
            if tmp:
                yield tmp.decode("utf-8")


def ldvocab(vfile, minf=False, omit_vsize=False):
    global has_unk
    if has_unk:
        rs = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        cwd = 4
    else:
        rs = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
        cwd = 3
    if omit_vsize:
        vsize = omit_vsize
    else:
        vsize = False
    for data in list_reader(vfile):
        freq = int(data[0])
        if (not minf) or freq > minf:
            if vsize:
                ndata = len(data) - 1
                if vsize >= ndata:
                    for wd in data[1:]:
                        rs[wd] = cwd
                        cwd += 1
                else:
                    for wd in data[1:vsize + 1]:
                        rs[wd] = cwd
                        cwd += 1
                        ndata = vsize
                    break
                vsize -= ndata
                if vsize <= 0:
                    break
            else:
                for wd in data[1:]:
                    rs[wd] = cwd
                    cwd += 1
        else:
            break
    return rs, cwd


def batch_loader(finput, fmt, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):
    def get_bsize(maxlen, maxtoken, maxbsize):
        rs = max(maxtoken // maxlen, 1)
        if (rs % 2 == 1) and (rs > 1):
            rs -= 1
        return min(rs, maxbsize)

    rsi = []
    rsm = []
    rst = []
    nd = 0
    maxlen = 0
    _bsize = bsize
    mlen_i = 0
    mlen_m = 0
    mlen_t = 0
    for i_d, m_d, td in zip(list_reader(finput), list_reader(fmt), list_reader(ftarget)):
        lid = len(i_d)
        lmd = len(m_d)
        ltd = len(td)
        lgth = lid + ltd
        if maxlen == 0:
            maxlen = lgth + min(maxpad, lgth // maxpart + 1)
            _bsize = get_bsize(maxlen, maxtoken, bsize)
        if (nd < minbsize) or (lgth <= maxlen and nd < _bsize):
            rsi.append(i_d)
            rsm.append(m_d)
            rst.append(td)
            if lid > mlen_i:
                mlen_i = lid
            if ltd > mlen_t:
                mlen_t = ltd
            if lmd > mlen_m:
                mlen_m = lmd
            nd += 1
        else:
            yield rsi, rsm, rst, mlen_i, mlen_m, mlen_t
            rsi = [i_d]
            rsm = [m_d]
            rst = [td]
            mlen_i = lid
            mlen_m = lmd
            mlen_t = ltd
            maxlen = lgth + min(maxpad, lgth // maxpart + 1)
            _bsize = get_bsize(maxlen, maxtoken, bsize)
            nd = 1
    if rsi:
        yield rsi, rsm, rst, mlen_i, mlen_m, mlen_t


def batch_mapper(finput, fmt, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize):
    def no_unk_mapper(vcb, ltm):
        rs = []
        for wd in ltm:
            if wd in vcb:
                rs.append(vcb[wd])
            else:
                print("Error mapping: " + wd)
        return rs

    for i_d, m_d, td, mlen_i, mlen_m, mlen_t in batch_loader(finput, fmt, ftarget, bsize, maxpad, maxpart, maxtoken,
                                                             minbsize):
        rsi = []
        for lined in i_d:
            tmp = [1]
            tmp.extend([vocabi.get(wd, 1) for wd in lined] if has_unk else no_unk_mapper(vocabi, lined))
            tmp.append(2)
            rsi.append(tmp)
        rst = []
        for lined in td:
            tmp = [1]
            tmp.extend([vocabt.get(wd, 1) for wd in lined] if has_unk else no_unk_mapper(vocabt, lined))
            tmp.append(2)
            rst.append(tmp)
        rsm = []
        for lined in m_d:
            tmp = [1]
            tmp.extend([vocabt.get(wd, 1) for wd in lined] if has_unk else no_unk_mapper(vocabt, lined))
            tmp.append(2)
            rsm.append(tmp)
        yield rsi, rsm, rst, mlen_i + 2, mlen_m + 2, mlen_t + 2


def batch_padder(finput, fmt, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize):
    for src, mt, pe, mlen_src, mlen_mt, mlen_pe in batch_mapper(finput, fmt, ftarget, vocabi, vocabt, bsize, maxpad,
                                                                maxpart, maxtoken, minbsize):
        # ld = []
        rid = []
        for lined in src:
            curlen = len(lined)
            # ld.append(curlen)
            if curlen < mlen_src:
                lined.extend([0 for i in range(mlen_src - curlen)])
            rid.append(lined)
        rmd = []
        for lined in mt:
            curlen = len(lined)
            # ld.append(curlen)
            if curlen < mlen_mt:
                lined.extend([0 for i in range(mlen_mt - curlen)])
            rmd.append(lined)
        rtd = []
        for lined in pe:
            curlen = len(lined)
            if curlen < mlen_pe:
                lined.extend([0 for i in range(mlen_pe - curlen)])
            rtd.append(lined)
        rid.reverse()
        rmd.reverse()
        rtd.reverse()
        # ld.reverse()
        yield rid, rmd, rtd


def handle(source_file, mt_file, pe_file, src_vocab_file, pe_vocab_file, tensor_file, minbsize=1,
           expand_for_mulgpu=True, bsize=672, maxpad=16,
           maxpart=4, maxtoken=3256, minfreq=False, vsize=False):
    vcbi, nwordi = ldvocab(src_vocab_file, minfreq,
                           vsize)  # vcbi = dictionary mapping word to index, nwordi = word index in vcb
    vcbt, nwordt = ldvocab(pe_vocab_file, minfreq, vsize)

    if expand_for_mulgpu:
        _bsize = bsize * minbsize
        _maxtoken = maxtoken * minbsize
    else:
        _bsize = bsize
        _maxtoken = maxtoken
    rsf = h5py.File(tensor_file, 'w')
    curd = 0
    for i_d, m_d, td in batch_padder(source_file, mt_file, pe_file, vcbi, vcbt, _bsize, maxpad, maxpart, _maxtoken,
                                     minbsize):
        rid = numpy.array(i_d, dtype=numpy.int32)  # src
        rmd = numpy.array(m_d, dtype=numpy.int32)  # mt
        rtd = numpy.array(td, dtype=numpy.int32)  # pe
        # rld = numpy.array(ld, dtype = numpy.int32)
        wid = str(curd)
        rsf["i" + wid] = rid
        rsf["m" + wid] = rmd
        rsf["t" + wid] = rtd
        # rsf["l" + wid] = rld
        curd += 1
    rsf["ndata"] = numpy.array([curd], dtype=numpy.int32)  # number of parallel data
    rsf["nwordi"] = numpy.array([nwordi],
                                dtype=numpy.int32)  # number of source word IN VOCAB [TODO: NEED TO CREATE VOCAB FOR MT]
    rsf["nwordt"] = numpy.array([nwordt], dtype=numpy.int32)  # number of target word IN VOCAB
    rsf.close()
    print("Number of batches: %d\nSource Vocabulary Size: %d\nTarget Vocabulary Size: %d" % (curd, nwordi, nwordt))


if __name__ == "__main__":
    handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], int(sys.argv[7]))

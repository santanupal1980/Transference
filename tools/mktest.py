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


def batch_loader(finput, fmt, bsize, maxpad, maxpart, maxtoken, minbsize):
    def get_bsize(maxlen, maxtoken, maxbsize):
        rs = max(maxtoken // maxlen, 1)
        if (rs % 2 == 1) and (rs > 1):
            rs -= 1
        return min(rs, maxbsize)

    rsi = []
    rsm = []
    nd = 0
    maxlen = 0
    _bsize = bsize
    mlen_i = 0
    mlen_d = 0
    for i_d, m_d in zip(list_reader(finput), list_reader(fmt)):
        lgth = len(i_d)
        lgth_m = len(m_d)
        if maxlen == 0:
            maxlen = lgth + min(maxpad, lgth // maxpart + 1)
            _bsize = get_bsize(maxlen, maxtoken, bsize)
        if (nd < minbsize) or (lgth <= maxlen and nd < _bsize):
            rsi.append(i_d)
            rsm.append(m_d)
            if lgth > mlen_i:
                mlen_i = lgth
            if lgth_m > mlen_d:
                mlen_d = lgth_m
            nd += 1
        else:
            yield rsi, rsm, mlen_i, mlen_d
            rsi = [i_d]
            rsm = [m_d]
            mlen_i = lgth
            mlen_d = lgth_m
            maxlen = lgth + min(maxpad, lgth // maxpart + 1)
            _bsize = get_bsize(maxlen, maxtoken, bsize)
            nd = 1
    if rsi:
        yield rsi, rsm, mlen_i, mlen_d


def batch_mapper(finput, fmt, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize):
    for i_d, m_d, mlen_i, mlen_d in batch_loader(finput, fmt, bsize, maxpad, maxpart, maxtoken, minbsize):
        rsi = []
        for lined in i_d:
            tmp = [1]
            tmp.extend([vocabi.get(wd, 1) for wd in lined])
            tmp.append(2)
            rsi.append(tmp)
        rsm = []
        for lined in m_d:
            tmp = [1]
            tmp.extend([vocabt.get(wd, 1) for wd in lined])
            tmp.append(2)
            rsm.append(tmp)
        yield rsi, rsm, mlen_i + 2, mlen_d + 2


def batch_padder(finput, fmt, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize):
    for i_d, m_d, mlen_i, mlen_d in batch_mapper(finput, fmt, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken,
                                                 minbsize):
        # ld = []
        rid = []
        for lined in i_d:
            curlen = len(lined)
            # ld.append(curlen)
            if curlen < mlen_i:
                lined.extend([0 for i in range(mlen_i - curlen)])
            rid.append(lined)
        rmd = []
        for lined in m_d:
            curlen = len(lined)
            if curlen < mlen_d:
                lined.extend([0 for i in range(mlen_d - curlen)])
            rmd.append(lined)
        yield rid, rmd


# maxtoken should be the maxtoken in mkiodata.py / 2 / beam size roughly, similar for bsize
def handle(finput, fmt, fvocab_i, fvocab_t, frs, minbsize=1, expand_for_mulgpu=True, bsize=128, maxpad=16, maxpart=4,
           maxtoken=1660, minfreq=False, vsize=False):
    vcbi, nwordi = ldvocab(fvocab_i, minfreq, vsize)
    vcbt, nwordt = ldvocab(fvocab_t, minfreq, vsize)
    if expand_for_mulgpu:
        _bsize = bsize * minbsize
        _maxtoken = maxtoken * minbsize
    else:
        _bsize = bsize
        _maxtoken = maxtoken
    rsf = h5py.File(frs, 'w')
    curd = 0
    for i_d, m_d in batch_padder(finput, fmt, vcbi, vcbt, _bsize, maxpad, maxpart, _maxtoken, minbsize):
        rid = numpy.array(i_d, dtype=numpy.int32)
        rmd = numpy.array(m_d, dtype=numpy.int32)
        # rld = numpy.array(ld, dtype = numpy.int32)
        wid = str(curd)
        rsf["i" + wid] = rid
        rsf["m" + wid] = rmd
        # rsf["l" + wid] = rld
        curd += 1
    rsf["ndata"] = numpy.array([curd], dtype=numpy.int32)
    rsf["nwordi"] = numpy.array([nwordi], dtype=numpy.int32)
    rsf.close()
    print("Number of batches: %d\nSource Vocabulary Size: %d" % (curd, nwordi))


if __name__ == "__main__":
    handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]))

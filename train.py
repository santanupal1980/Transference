import logging
import sys
from os import makedirs
from os.path import exists as path_check
from random import seed as rpyseed
from random import shuffle, sample

import h5py
from torch import optim
from torch.nn.init import xavier_uniform_
from tqdm import tqdm

from Transference.NMT import NMT
from config import config as cnfg
from loss import LabelSmoothingLoss
from lrsch import GoogleLR
from modules import *


# train function required training data (td), validation data (ed)
def train(train_data, tl, ed, nd, optm, lrsch, model, lossf, mv_device, logger, done_tokens, multi_gpu,
          tokens_optm=32768,
          nreport=None, save_every=None, chkpf=None, chkpof=None, statesf=None, num_checkpoint=1, cur_checkid=0,
          report_eva=True, remain_steps=None, save_loss=False):
    sum_loss = 0.0  # total loss after itearing through the training data
    sum_wd = 0
    part_loss = 0.0
    part_wd = 0
    _done_tokens = done_tokens
    model.train()
    cur_b = 1
    ndata = len(tl)
    _cur_checkid = cur_checkid
    _cur_rstep = remain_steps
    _ls = {} if save_loss else None

    for i_d, m_d, t_d in tqdm(tl):
        seq_batch_src = torch.from_numpy(train_data[i_d][:]).long()  # tensor for source
        seq_batch_mt = torch.from_numpy(train_data[m_d][:]).long()  # tensor for mt
        seq_batch_pe = torch.from_numpy(train_data[t_d][:]).long()  # tensor for pe
        lo = seq_batch_pe.size(1) - 1  # output sequence length
        if mv_device:  # moving tensors to GPU for fast processing
            seq_batch_src = seq_batch_src.to(mv_device)
            seq_batch_mt = seq_batch_mt.to(mv_device)
            seq_batch_pe = seq_batch_pe.to(mv_device)

        if _done_tokens >= tokens_optm:
            # token optimization condition lets say if 25000 tokens are initialized then after every 25000 token processed
            # gradient will be optimized
            optm.zero_grad()  # [Before training clear the gradient]
            _done_tokens = 0  # RESET

        # INPUT OF THE DECODER, LIKE LM TRAINING THE LAST TOKEN NEED TO GNERT SO WITHOUT LAST TOKEN NEED TO INPUT </S>
        dec_seq_input = seq_batch_pe.narrow(1, 0, lo)
        dec_seq_output = seq_batch_pe.narrow(1, 1, lo).contiguous()  # WITHOUT FIRST TOKEN <S>
        output = model(seq_batch_src, seq_batch_mt, dec_seq_input)  # CALL MODEL
        loss = lossf(output, dec_seq_output)  # CALCULATE LOSS

        if multi_gpu:
            loss = loss.sum()

        loss_add = loss.data.item()
        sum_loss += loss_add  # OVERALL LOSS

        # HOW MANY ELEMENT IN THE OUTPUT TOKENS HAVE BEEN PROCESSED
        wd_add = dec_seq_output.numel() - dec_seq_output.eq(0).sum().item()

        if save_loss:
            _ls[(i_d, m_d, t_d)] = loss_add / wd_add
        sum_wd += wd_add
        _done_tokens += wd_add
        if nreport is not None:
            part_loss += loss_add
            part_wd += wd_add
            if cur_b % nreport == 0:
                if report_eva:
                    _leva, _eeva = eva(ed, nd, model, lossf, mv_device, multi_gpu)
                    logger.info("Average loss over %d tokens: %.3f, valid loss/error: %.3f %.2f" % (
                        part_wd, part_loss / part_wd, _leva, _eeva))
                else:
                    logger.info("Average loss over %d tokens: %.3f" % (part_wd, part_loss / part_wd))
                part_loss = 0.0
                part_wd = 0

        # scale the sum of losses down according to the number of tokens adviced by: https://mp.weixin.qq.com/s/qAHZ4L5qK3rongCIIq5hQw,
        # I think not reasonable.
        # loss /= wd_add
        loss.backward()

        if _done_tokens >= tokens_optm:
            if multi_gpu:
                model.collect_gradients()
                optm.step()
                model.update_replicas()
            else:
                optm.step()
            if _cur_rstep is not None:
                _cur_rstep -= 1
                if _cur_rstep <= 0:
                    break
            lrsch.step()

        if (save_every is not None) and (cur_b % save_every == 0) and (chkpf is not None) and (cur_b < ndata):
            if num_checkpoint > 1:
                _fend = "_%d.t7" % (_cur_checkid)
                _chkpf = chkpf[:-3] + _fend
                if chkpof is not None:
                    _chkpof = chkpof[:-3] + _fend
                _cur_checkid = (_cur_checkid + 1) % num_checkpoint
            else:
                _chkpf = chkpf
                _chkpof = chkpof
            # save_model(model, _chkpf, isinstance(model, nn.DataParallel))
            save_model(model, _chkpf, multi_gpu)
            if chkpof is not None:
                torch.save(optm.state_dict(), _chkpof)
            if statesf is not None:
                save_states(statesf, tl[cur_b - 1:])
        cur_b += 1
    if part_wd != 0.0:
        logger.info("Average loss over %d tokens: %.3f" % (part_wd, part_loss / part_wd))
    return sum_loss / sum_wd, _done_tokens, _cur_checkid, _cur_rstep, _ls


def eva(ed, nd, model, lossf, mv_device, multi_gpu):
    r = 0
    w = 0
    sum_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(nd)):
            bid = str(i)
            seq_batch = torch.from_numpy(ed["i" + bid][:]).long()
            seq_batch_mt = torch.from_numpy(ed["m" + bid][:]).long()
            seq_o = torch.from_numpy(ed["t" + bid][:]).long()
            lo = seq_o.size(1) - 1
            if mv_device:
                seq_batch = seq_batch.to(mv_device)
                seq_batch_mt = seq_batch_mt.to(mv_device)
                seq_o = seq_o.to(mv_device)
            ot = seq_o.narrow(1, 1, lo).contiguous()
            output = model(seq_batch, seq_batch_mt, seq_o.narrow(1, 0, lo))
            loss = lossf(output, ot)
            if multi_gpu:
                loss = loss.sum()
                trans = torch.cat([torch.argmax(outu, -1).to(mv_device) for outu in output], 0)
            else:
                trans = torch.argmax(output, -1)
            sum_loss += loss.data.item()
            data_mask = 1 - ot.eq(0)  # ?
            correct = torch.gt(trans.eq(ot) + data_mask, 1)
            w += data_mask.sum().item()
            r += correct.sum().item()
    w = float(w)
    return sum_loss / w, (w - r) / w * 100.0


# to know learing rates during optimization
def getlr(optm):
    lr = []
    for i, param_group in enumerate(optm.param_groups):
        lr.append(float(param_group['lr']))
    return lr


# this function returns boolean value indicating whether the learning rate is updated or not
def updated_lr(oldlr, newlr):
    rs = False
    for olr, nlr in zip(oldlr, newlr):
        if olr != nlr:
            rs = True
            break
    return rs


# TODO: need to be documented
def hook_lr_update(optm, flags):
    for group in optm.param_groups:
        for p in group['params']:
            state = optm.state[p]
            if len(state) != 0:
                state['step'] = 0
                state['exp_avg'].zero_()
                state['exp_avg_sq'].zero_()
                if flags:
                    state['max_exp_avg_sq'].zero_()


# TODO: need to be documented
def dynamic_sample(incd, dss_ws, dss_rm):
    rd = {}
    for k, v in incd.items():
        if v in rd:
            rd[v].append(k)
        else:
            rd[v] = [k]
    incs = list(rd.keys())
    incs.sort(reverse=True)
    _full_rl = []
    for v in incs:
        _full_rl.extend(rd[v])

    return _full_rl[:dss_ws] + sample(_full_rl[dss_ws:], dss_rm) if dss_rm > 0 else _full_rl[:dss_ws]


def tostr(lin):
    return [str(lu) for lu in lin]


def get_logger(fname):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(fname)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


# TODO: need to be documented
def save_states(fname, stl):
    with open(fname, "wb") as f:
        f.write(" ".join([i[0][1:] for i in stl]).encode("utf-8"))
        f.write("\n".encode("utf-8"))


# TODO: need to be documented
def load_states(fname):
    rs = []
    with open(fname, "rb") as f:
        for line in f:
            tmp = line.strip()
            if tmp:
                for tmpu in tmp.decode("utf-8").split():
                    if tmpu:
                        rs.append(tmpu)
    return [("i" + tmpu, "m" + tmpu, "t" + tmpu) for tmpu in rs]


# TODO: need to be documented
def load_model_cpu(modf, base_model):
    mpg = torch.load(modf, map_location='cpu')

    for para, mp in zip(base_model.parameters(), mpg):
        para.data = mp.data

    return base_model


def save_model(model, fname, sub_module):  # TODO: Need documentation
    if sub_module:
        torch.save([t.data for t in model.module.parameters()], fname)
    else:
        torch.save([t.data for t in model.parameters()], fname)


def init_fixing(module):  # TODO: Need documentation
    if "fix_init" in dir(module):
        module.fix_init()


def main():
    ''' Main function '''
    rid = cnfg.runid  # Get run ID from cnfg file where training files will be stored
    if len(sys.argv) > 1:
        rid = sys.argv[1]  # getting runid from console

    earlystop = cnfg.earlystop  # Get early-stop criteria
    epochs = cnfg.epochs  #

    tokens_optm = cnfg.tokens_optm  # number of tokens

    done_tokens = tokens_optm

    batch_report = cnfg.batch_report
    report_eva = cnfg.report_eva

    use_cuda = cnfg.use_cuda
    gpuid = cnfg.gpuid

    # GPU configuration
    if use_cuda and torch.cuda.is_available():
        use_cuda = True
        if len(gpuid.split(",")) > 1:
            cuda_device = torch.device(gpuid[:gpuid.find(",")].strip())
            cuda_devices = [int(_.strip()) for _ in gpuid[gpuid.find(":") + 1:].split(",")]
            print('[Info] using multiple gpu', cuda_devices)
            multi_gpu = True
        else:
            cuda_device = torch.device(gpuid)
            multi_gpu = False
            print('[Info] using single gpu', cuda_device)
            cuda_devices = None
        torch.cuda.set_device(cuda_device.index)
    else:
        cuda_device = False
        print('using single cpu')
        multi_gpu = False
        cuda_devices = None

    use_ams = cnfg.use_ams  # ?

    save_optm_state = cnfg.save_optm_state

    save_every = cnfg.save_every

    epoch_save = cnfg.epoch_save

    remain_steps = cnfg.training_steps

    wkdir = "".join((cnfg.work_dir, cnfg.data_dir, "/", rid, "/"))  # CREATING MODEL DIRECTORY
    if not path_check(wkdir):
        makedirs(wkdir)

    chkpt = None
    chkptoptf = None
    chkptstatesf = None
    if save_every is not None:
        chkpt = wkdir + "checkpoint.t7"
        if save_optm_state:
            chkptoptf = wkdir + "checkpoint.optm.t7"
            chkptstatesf = wkdir + "checkpoint.states"

    logger = get_logger(wkdir + "train.log")  # Logger object

    train_data = h5py.File(cnfg.train_data, "r")  # training data read from h5 file
    valid_data = h5py.File(cnfg.dev_data, "r")  # validation data read from h5 file

    print('[Info] Training and Validation data are loaded.')

    ntrain = int(train_data["ndata"][:][0])  # number of batches for TRAINING DATA
    nvalid = int(valid_data["ndata"][:][0])  # number of batches for VALIDATION DATA
    nwordi = int(train_data["nwordi"][:][0])  # VOCAB SIZE FOR SOURCE
    nwordt = int(train_data["nwordt"][:][0])  # VOCAB SIZE FOR PE [TODO: SIMILAR FOR MT]

    print('[INFO] number of batches for TRAINING DATA: ', ntrain)
    print('[INFO] number of batches for VALIDATION DATA: ', nvalid)
    print('[INFO] Source vocab size: ', nwordi)
    print('[INFO] Target vocab size: ', nwordt)

    random_seed = torch.initial_seed() if cnfg.seed is None else cnfg.seed

    rpyseed(random_seed)

    if use_cuda:
        torch.cuda.manual_seed_all(random_seed)
        print('[Info] Setting up random seed using CUDA.')
    else:
        torch.manual_seed(random_seed)

    logger.info("Design models with seed: %d" % torch.initial_seed())

    mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.num_src_layer, cnfg.num_mt_layer, cnfg.num_pe_layer, cnfg.ff_hsize,
                  cnfg.drop, cnfg.attn_drop, cnfg.share_emb, cnfg.nhead, cnfg.cache_len, cnfg.attn_hsize,
                  cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)  # TODO NEED DOCUMENTATION

    tl = [("i" + str(i), "m" + str(i), "t" + str(i)) for i in range(ntrain)]  # TRAINING LIST

    fine_tune_m = cnfg.fine_tune_m
    # Fine tune model

    if fine_tune_m is not None:
        logger.info("Load pre-trained model from: " + fine_tune_m)
        mymodel = load_model_cpu(fine_tune_m, mymodel)

    lossf = LabelSmoothingLoss(nwordt, cnfg.label_smoothing, ignore_index=0, reduction='sum',
                               forbidden_index=cnfg.forbidden_indexes)

    if use_cuda:
        mymodel.to(cuda_device)
        lossf.to(cuda_device)

    if fine_tune_m is None:
        for p in mymodel.parameters():
            if p.requires_grad and (p.dim() > 1):
                xavier_uniform_(p)
        if cnfg.src_emb is not None:
            _emb = torch.load(cnfg.src_emb, map_location='cpu')
            if nwordi < _emb.size(0):
                _emb = _emb.narrow(0, 0, nwordi).contiguous()
            if use_cuda:
                _emb = _emb.to(cuda_device)
            mymodel.enc.wemb.weight.data = _emb
            if cnfg.freeze_srcemb:
                mymodel.enc.wemb.weight.requires_grad_(False)
            else:
                mymodel.enc.wemb.weight.requires_grad_(True)
        if cnfg.tgt_emb is not None:
            _emb = torch.load(cnfg.tgt_emb, map_location='cpu')
            if nwordt < _emb.size(0):
                _emb = _emb.narrow(0, 0, nwordt).contiguous()
            if use_cuda:
                _emb = _emb.to(cuda_device)
            mymodel.dec.wemb.weight.data = _emb
            if cnfg.freeze_tgtemb:
                mymodel.dec.wemb.weight.requires_grad_(False)
            else:
                mymodel.dec.wemb.weight.requires_grad_(True)
        mymodel.apply(init_fixing)

    # lr will be over written by GoogleLR before used
    optimizer = optim.Adam(mymodel.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9, weight_decay=cnfg.weight_decay,
                           amsgrad=use_ams)

    # TODO: Need to implement
    '''if multi_gpu:
        # mymodel = nn.DataParallel(mymodel, device_ids=cuda_devices, output_device=cuda_device.index)
        mymodel = DataParallelMT(mymodel, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True,
                                 gather_output=False)
        lossf = DataParallelCriterion(lossf, device_ids=cuda_devices, output_device=cuda_device.index,
                                      replicate_once=True)'''

    # Load fine tune state if declared
    fine_tune_state = cnfg.fine_tune_state
    if fine_tune_state is not None:
        logger.info("Load optimizer state from: " + fine_tune_state)
        optimizer.load_state_dict(torch.load(fine_tune_state))

    lrsch = GoogleLR(optimizer, cnfg.isize, cnfg.warm_step)
    lrsch.step()

    num_checkpoint = cnfg.num_checkpoint
    cur_checkid = 0  # initialized current check point

    tminerr = float("inf")  # minimum error during training

    minloss, minerr = eva(valid_data, nvalid, mymodel, lossf, cuda_device, multi_gpu)
    logger.info(
        "".join(("Init lr: ", ",".join(tostr(getlr(optimizer))), ", Dev Loss/Error: %.3f %.2f" % (minloss, minerr))))

    # if fine_tune_m is None:
    save_model(mymodel, wkdir + "init.t7", multi_gpu)
    logger.info("Initial model saved")
    # ==================================================Fine tune ========================================
    if fine_tune_m is None:
        save_model(mymodel, wkdir + "init.t7", multi_gpu)
        logger.info("Initial model saved")
    else:
        cnt_states = cnfg.train_statesf
        if (cnt_states is not None) and path_check(cnt_states):
            logger.info("Continue last epoch")
            args = {}
            tminerr, done_tokens, cur_checkid, remain_steps, _ = train(train_data, load_states(cnt_states), valid_data,
                                                                       nvalid,
                                                                       optimizer,
                                                                       lrsch, mymodel, lossf, cuda_device, logger,
                                                                       done_tokens, multi_gpu, tokens_optm,
                                                                       batch_report,
                                                                       save_every, chkpt, chkptoptf, chkptstatesf,
                                                                       num_checkpoint,
                                                                       cur_checkid, report_eva, remain_steps, False)
            vloss, vprec = eva(valid_data, nvalid, mymodel, lossf, cuda_device, multi_gpu)
            logger.info("Epoch: 0, train loss: %.3f, valid loss/error: %.3f %.2f" % (tminerr, vloss, vprec))
            save_model(mymodel, wkdir + "train_0_%.3f_%.3f_%.2f.t7" % (tminerr, vloss, vprec), multi_gpu)
            if save_optm_state:
                torch.save(optimizer.state_dict(), wkdir + "train_0_%.3f_%.3f_%.2f.optm.t7" % (tminerr, vloss, vprec))
            logger.info("New best model saved")

        # assume that the continue trained model has already been through sort grad, thus shuffle the training list.
        shuffle(tl)
    # ====================================================================================================

    # ================================Dynamic sentence Sampling =========================================
    if cnfg.dss_ws is not None and cnfg.dss_ws > 0.0 and cnfg.dss_ws < 1.0:
        dss_ws = int(cnfg.dss_ws * ntrain)
        _Dws = {}
        _prev_Dws = {}
        _crit_inc = {}
        if cnfg.dss_rm is not None and cnfg.dss_rm > 0.0 and cnfg.dss_rm < 1.0:
            dss_rm = int(cnfg.dss_rm * ntrain * (1.0 - cnfg.dss_ws))
        else:
            dss_rm = 0
    else:
        dss_ws = 0
        dss_rm = 0
        _Dws = None
    # ====================================================================================================

    namin = 0

    # TRAINING EPOCH STARTS
    for i in range(1, epochs + 1):
        terr, done_tokens, cur_checkid, remain_steps, _Dws = train(train_data, tl, valid_data, nvalid, optimizer, lrsch,
                                                                   mymodel, lossf,
                                                                   cuda_device, logger, done_tokens, multi_gpu,
                                                                   tokens_optm,
                                                                   batch_report, save_every, chkpt, chkptoptf,
                                                                   chkptstatesf,
                                                                   num_checkpoint, cur_checkid, report_eva,
                                                                   remain_steps,
                                                                   dss_ws > 0)
        # VALIDATION
        vloss, vprec = eva(valid_data, nvalid, mymodel, lossf, cuda_device, multi_gpu)
        logger.info("Epoch: %d ||| train loss: %.3f ||| valid loss/error: %.3f/%.2f" % (i, terr, vloss, vprec))

        # CONDITION TO SAVE MODELS
        if (vprec <= minerr) or (vloss <= minloss):
            save_model(mymodel, wkdir + "eva_%d_%.3f_%.3f_%.2f.t7" % (i, terr, vloss, vprec), multi_gpu)
            if save_optm_state:
                torch.save(optimizer.state_dict(), wkdir + "eva_%d_%.3f_%.3f_%.2f.optm.t7" % (i, terr, vloss, vprec))
            logger.info("New best model saved")  # [TODO CALCULATE BLEU FOR VALIDATION SET]

            namin = 0

            if vprec < minerr:
                minerr = vprec
            if vloss < minloss:
                minloss = vloss

        else:
            if terr < tminerr:
                tminerr = terr
                save_model(mymodel, wkdir + "train_%d_%.3f_%.3f_%.2f.t7" % (i, terr, vloss, vprec), multi_gpu)
                if save_optm_state:
                    torch.save(optimizer.state_dict(),
                               wkdir + "train_%d_%.3f_%.3f_%.2f.optm.t7" % (i, terr, vloss, vprec))
            elif epoch_save:
                save_model(mymodel, wkdir + "epoch_%d_%.3f_%.3f_%.2f.t7" % (i, terr, vloss, vprec), multi_gpu)

            namin += 1
            # CONDITIONED TO EARLY STOP
            if namin >= earlystop:
                if done_tokens > 0:
                    if multi_gpu:
                        mymodel.collect_gradients()
                    optimizer.step()
                    # lrsch.step()
                    done_tokens = 0
                # optimizer.zero_grad()
                logger.info("early stop")
                break

        if remain_steps is not None and remain_steps <= 0:
            logger.info("Last training step reached")
            break

        '''if dss_ws > 0:
            if _prev_Dws:
                for _key, _value in _Dws.items():
                    if _key in _prev_Dws:
                        _ploss = _prev_Dws[_key]
                        _crit_inc[_key] = (_ploss - _value) / _ploss
                tl = dynamic_sample(_crit_inc, dss_ws, dss_rm)
            _prev_Dws = _Dws'''

        shuffle(tl)

        '''oldlr = getlr(optimizer)
        lrsch.step(terr)
        newlr = getlr(optimizer)
        if updated_lr(oldlr, newlr):
          logger.info("".join(("lr update from: ", ",".join(tostr(oldlr)), ", to: ", ",".join(tostr(newlr)))))
          hook_lr_update(optimizer, use_ams)'''

    if done_tokens > 0:
        if multi_gpu:
            mymodel.collect_gradients()
        optimizer.step()
    # lrsch.step()
    # done_tokens = 0
    # optimizer.zero_grad()

    save_model(mymodel, wkdir + "last.t7", multi_gpu)
    if save_optm_state:
        torch.save(optimizer.state_dict(), wkdir + "last.optm.t7")
    logger.info("model saved")

    train_data.close()
    valid_data.close()


if __name__ == '__main__':
    main()

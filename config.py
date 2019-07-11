class config:
    data_dir = "ape_data"  # place your data in cache/ape_data
    runid = "exp_ape"  # this is the run id all training check points will be here use eva_less-perplexity
    work_dir = "expm"  # experiment folder as wordking directory

    train_data = "cache/" + data_dir + "/train.h5"  # 'train.h5' prepared after running mktrain.sh same as the followings
    dev_data = "cache/" + data_dir + "/dev.h5"
    test_data = "cache/" + data_dir + "/test.h5"  # 'test.h5' prepared after running mktest.sh but use during inference

    fine_tune_m = None  # If you re-run or fine-tune initialize best eva_* here
    train_statesf = None
    fine_tune_state = None

    earlystop = 8
    maxrun = 128

    tokens_optm = 25000  # see token optimization paper
    done_tokens = tokens_optm

    batch_report = 10000  # after this steps it will report in the console
    report_eva = True

    use_cuda = True
    # enable Data Parallel multi-gpu support with values like: 'cuda:0, 1, 3'.
    gpuid = 'cuda:0'
    # use multi-gpu for translating or not. `predict.py` will take the last gpu rather than the first in case multi_gpu_decoding is set to False to avoid potential break due to out of memory, since the first gpu is the main device by default which takes more jobs.
    multi_gpu_decoding = True

    beam_size = 4

    use_ams = False
    save_optm_state = False
    save_every = None
    num_checkpoint = 10
    epoch_save = True

    training_steps = None

    # Embedding conditions [TODO: Need documentation]
    src_emb = None
    freeze_srcemb = False
    tgt_emb = None
    freeze_tgtemb = False

    # model parameters
    seed = 666666  # initial random seeds

    isize = 512  # input embedding size, hidden layer size
    # nlayer = 6
    num_src_layer = 6
    num_mt_layer = 6
    num_pe_layer = 6
    ff_hsize = 4 * isize
    drop = 0.1
    attn_drop = 0.1
    label_smoothing = 0.1

    share_emb = False

    nhead = 8
    cache_len = 260
    attn_hsize = None
    norm_output = True
    bindDecoderEmb = True
    # non-exist indexes in the classifier.
    # "<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3
    # add 3 to forbidden_indexes if there are <unk> tokens in data
    forbidden_indexes = [0, 1]

    # Optiizer setup like google paper
    weight_decay = 0
    length_penalty = 0.0
    warm_step = 8000

    # to accelerate training through sampling, 0.8 and 0.1 in: Dynamic Sentence Sampling for Efficient Training of Neural Machine Translation
    dss_ws = None
    dss_rm = None

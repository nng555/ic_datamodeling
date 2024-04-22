import os
import itertools
import pickle as pkl
import pathlib
import hydra
import logging
from omegaconf import DictConfig
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from ic_datamodel import ICDatamodel
from utils import PROMPT_MAP, TEMPLATE_MAP, EOS_TOK_MAP, get_gpu_type

log = logging.getLogger("eval")

curr_gpu = get_gpu_type()

# only run backwards pass methods on A40 GPUs
if 'A40' not in curr_gpu:
    METHODS = ['datamodel', 'shapley', 'attention', 'loo']
else:
    METHODS = ['datamodel', 'ig', 'saliency', 'shapley', 'attention', 'loo']

METRICS = ['malds', 'slds'] + ['_'.join(v) for v in itertools.product(['suff', 'comp'], ['remove', 'zero'], ['tok', 'seq'])]

# config path set automatically but can also be manually changed
CONFIG_PATH = os.path.join(pathlib.Path(__file__).parent.resolve())

def eval_datamodel(datamodel, seed=0, nexamples=128):
    malds = []
    slds = []
    res = {}

    def print_lds_res(name, res):
        print(f"{name:20} MaLDS: {res[0]:.5f}\t SLDS: {res[1]:.5f} ")
        malds.append(res[0])
        slds.append(res[1])

    topk_res = datamodel.eval_topk(max_p=0.5)
    print(f"{'Sufficiency Remove':20} Tok: {topk_res['suff_remove_tok']:.5f}\t Seq PPLO: {topk_res['suff_remove_seq']:.5f}")
    print(f"{'Sufficiency Zero':20} Tok: {topk_res['suff_zero_tok']:.5f}\t Seq PPLO: {topk_res['suff_zero_seq']:.5f}")
    print(f"{'Comprehensive Remove':20} Tok: {topk_res['comp_remove_tok']:.5f}\t Seq PPLO: {topk_res['comp_remove_seq']:.5f}")
    print(f"{'Comprehensive Zero':20} Tok: {topk_res['comp_zero_tok']:.5f}\t Seq PPLO: {topk_res['comp_zero_seq']:.5f}")
    for ngram in [1, 5, 10]:
        for mask_p in [0.05, 0.10, 0.15, 0.2, 0.25]:
            print_lds_res(f"{ngram}-gram {int(mask_p*100)}%", datamodel.eval_lds(
                mask_type='ngram_frac',
                seed=(ngram + int(mask_p * 100)),
                ngram=ngram,
                nexamples=nexamples,
                mask_p=mask_p,
            ))
    print('-' * 55)
    print_lds_res("Totals", [np.mean(malds), np.mean(slds)])
    print()

    for m in METRICS[2:]:
        res[m] = topk_res[m]
    res['malds'] = np.mean(malds)
    res['slds'] = np.mean(slds)

    return res

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="slurm_config")
def eval_dataset(cfg: DictConfig):

    assert cfg.dataset in PROMPT_MAP, f"{cfg.dataset} not supported."
    prompt = PROMPT_MAP[cfg.dataset]
    gen_start = prompt[-1]
    prompt = prompt[:-1]
    template = TEMPLATE_MAP[cfg.model]
    eos_tok = EOS_TOK_MAP.get(cfg.model, None)

    if cfg.subset == '':
        dataset = load_dataset(cfg.dataset)
    else:
        dataset = load_dataset(cfg.dataset, cfg.subset)

    split_order = ['test', 'val', 'validation', 'train']
    for split in split_order:
        if split in dataset:
            testset = dataset[split]
            break

    if cfg.shuffle is not None:
        testset = testset.shuffle(cfg.shuffle)

    tokenizer = AutoTokenizer.from_pretrained(f"/fs01/model-weights/{cfg.model}")
    model = AutoModelForCausalLM.from_pretrained(f"/fs01/model-weights/{cfg.model}", torch_dtype=torch.float16, device_map="auto").eval()
    if cfg.model == 'Meta-Llama-3-8B-Instruct':
        tokenizer.unk_token_id = 128255 # manually set UNK to an unused ID (0 embedding)

    datamodel = ICDatamodel(
        model,
        tokenizer,
        template,
        sampling=False,
        alpha=0.001,
        eos_tok=eos_tok,
        bsize=cfg.bsize,
    )

    # initialize results dict
    outpath = 'res.pkl'
    if os.path.exists(outpath):
        res = pkl.load(open(outpath, 'rb'))

        # cut to start at the same example
        nprocessed = np.min([len(res[m]['slds']) for m in METHODS])
    else:
        res = {method: {metric: [] for metric in METRICS} for method in METHODS}
        nprocessed = 0

    start_idx = nprocessed * cfg.num_shards + cfg.shard
    print(f"Skipping to index {start_idx}, {nprocessed} processed in full.")

    for i, test_ex in enumerate(testset):
        if i < start_idx:
            continue

        if cfg.max_idx is not None and i > cfg.max_idx:
            break

        if i % cfg.num_shards != cfg.shard:
            continue

        shard_idx = i // cfg.num_shards

        # preprocess choices and document?
        ex_prompt = [p.format(**test_ex) for p in prompt]
        datamodel.gen(ex_prompt, gen_start)

        for method in METHODS:

            # skip if already done for this method
            if len(res[method]['slds']) > shard_idx:
                continue

            if method == 'datamodel':
                nexamples = cfg.nepochs * datamodel.ntoks
                datamodel.fit(
                    mask_type='random', seed=0, whole_word_masking=False,
                    nexamples=nexamples, lo_mask_p=cfg.lo_mask_p, hi_mask_p=cfg.mask_p
                )
            elif method == 'ig':
                datamodel.fit_grads(nsteps=100)
            elif method == 'saliency':
                datamodel.fit_grads(nsteps=1)
            elif method == 'shapley':
                datamodel.fit_shap(nepochs=cfg.nepochs)
            elif method == 'attention':
                datamodel.fit_attention()
            elif method == 'loo':
                datamodel.fit(mask_type='tok')

            eval_res = eval_datamodel(datamodel, nexamples=cfg.nexamples_eval)
            for k, v in eval_res.items():
                res[method][k].append(v)

            with open('tmp.pkl', 'wb') as of:
                pkl.dump(res, of)

            os.rename('tmp.pkl', 'res.pkl')

if __name__ == "__main__":
    eval_dataset()


import os
import pathlib
import hydra
import logging
from omegaconf import DictConfig
import pickle as pkl
from torch import float16
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from ic_datamodel import ICDatamodel
from utils import PROMPT_MAP, TEMPLATE_MAP, EOS_TOK_MAP

P_CHOICES = [0.05 * i for i in range(20)]

log = logging.getLogger("sweep_p")

# config path set automatically but can also be manually changed
CONFIG_PATH = os.path.join(pathlib.Path(__file__).parent.resolve())

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="slurm_config")
def sweep_dataset(cfg: DictConfig):

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

    tokenizer = AutoTokenizer.from_pretrained(f"/fs01/model-weights/{cfg.model}")
    model = AutoModelForCausalLM.from_pretrained(f"/fs01/model-weights/{cfg.model}", torch_dtype=float16, device_map="auto").eval()
    if cfg.model == 'Meta-Llama-3-8B-Instruct':
        tokenizer.unk_token_id = 128255

    datamodel = ICDatamodel(
        model,
        tokenizer,
        template,
        sampling=True,
        alpha=0.001,
        eos_tok=eos_tok,
    )

    split_order = ['test', 'val', 'validation', 'train']
    for split in split_order:
        if split in dataset:
            testset = dataset[split]
            break

    res = []
    for i, test_ex in enumerate(testset):
        #if i < 9:
        #    continue
        ex_prompt = [p.format(**test_ex) for p in prompt]

        res.append({})
        datamodel.gen(ex_prompt, gen_start)
        nexamples_train = cfg.nepochs * datamodel.ntoks
        for train_p in P_CHOICES:
            if train_p == 0.0:
                datamodel.fit(mask_type='tok')
            else:
                datamodel.fit(
                    mask_type='random',
                    hi_mask_p=train_p,
                    seed=0,
                    whole_word_masking=False,
                    nexamples=nexamples_train,
                )
            #topk_res = datamodel.eval_topk()
            #print(f"{'Sufficiency Remove':20} Tok: {topk_res['suff_remove_tok']:.5f}\t Seq PPLO: {topk_res['suff_remove_seq']:.5f}")
            #print(f"{'Sufficiency Zero':20} Tok: {topk_res['suff_zero_tok']:.5f}\t Seq PPLO: {topk_res['suff_zero_seq']:.5f}")
            #print(f"{'Comprehensive Remove':20} Tok: {topk_res['comp_remove_tok']:.5f}\t Seq PPLO: {topk_res['comp_remove_seq']:.5f}")
            #print(f"{'Comprehensive Zero':20} Tok: {topk_res['comp_zero_tok']:.5f}\t Seq PPLO: {topk_res['comp_zero_seq']:.5f}")
            for ngram in [1, 5, 10]:
                for eval_p in P_CHOICES:
                    if eval_p == 0.0:
                        lds_res = datamodel.eval_lds(mask_type='ngram', seed=0, ngram=ngram)
                    else:
                        lds_res = datamodel.eval_lds(
                            mask_type='ngram_frac',
                            seed=(ngram + int(eval_p * 100)),
                            ngram=ngram,
                            nexamples=cfg.nexamples_eval,
                            mask_p=eval_p,
                        )
                    name = f"{ngram}-gram {int(eval_p*100)}%"
                    print(f"{name:20} MaLDS: {lds_res[0]:.5f}\t SLDS: {lds_res[1]:.5f} ", flush=True)
                    res[-1][(int(train_p * 100), int(100 * eval_p), ngram)] = lds_res

        with open("res.pkl", 'wb') as of:
            pkl.dump(res, of)

if __name__ == "__main__":
    sweep_dataset()

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='dataset name')
    parser.add_argument('-s', '--subset', type=str, help='subset name')
    parser.add_argument('-n', '--nepochs', type=int, help='training epochs')
    args = parser.parse_args()

    if args.subset == '':
        dataset = load_dataset(args.dataset)
    else:
        dataset = load_dataset(args.dataset, args.subset)

    if args.dataset not in PROMPT_MAP:
        raise Exception(f"{args.dataset} not supported.")

    sweep_dataset(dataset, PROMPT_MAP[args.dataset][:-1], PROMPT_MAP[args.dataset][-1], args.nepochs)
"""

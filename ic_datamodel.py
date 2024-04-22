import torch.nn.functional as F
from transformers import StoppingCriteriaList
from collections import defaultdict
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from scipy.stats import spearmanr
import numpy as np
import torch
from utils import StoppingCriteriaSub, WORD_STARTS

#from datamodel_mistral import DatamodelMistralForCausalLM

class ICDatamodel:

    def __init__(
        self,
        model,
        tokenizer,
        template,
        bsize=8,
        use_unk=True,
        sampling=False,
        top_p=0.9,
        temperature=1.5,
        top_k=100,
        max_gen_len=200,
        eps=1e-30,
        alpha=0.0,
        eos_tok=None
    ):
        self.model = model
        self.eos_tok = eos_tok
        self.tokenizer = tokenizer
        self.template = template
        self.pstart_len = len(self.tokenizer(template[0])['input_ids'])
        self.bsize = bsize
        self.sampling = sampling
        self.device = self.model.device
        self.top_p = top_p
        self.temperature = temperature
        self.top_k = top_k
        self.max_gen_len = max_gen_len
        self.eps = eps
        self.alpha = alpha
        self.eval_cache = {}

    def generate(
        self,
        prompt, # assume [prompt start, context, prompt end]
        gen_start,
        max_gen_len,
        stop_words=[]
    ):
        if self.eos_tok is not None:
            stop_words.append(self.eos_tok)

        stop_token_ids = [self.tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_token_ids)])

        self.eval_cache = {}
        prompt_lens = [len(self.tokenizer.tokenize(p)) for p in prompt]

        # add a space if needed for proper word splitting
        if gen_start != '':
            gen_start = ' ' + gen_start

        context = self.template[0] + ''.join(prompt) + self.template[1] + gen_start
        context_inputs = self.tokenizer([context], return_tensors='pt').to(self.device)
        context_len = len(context_inputs[0])

        if not self.sampling:
            generated_ids = self.model.generate(
                **context_inputs,
                max_length=len(context_inputs[0]) + max_gen_len,
                do_sample=False,
                stopping_criteria=stopping_criteria,
            )
        else:
            generated_ids = self.model.generate(
                **context_inputs,
                max_length=len(context_inputs[0]) + max_gen_len,
                do_sample=True,
                top_p=self.top_p,
                temperature=self.temperature,
                top_k=self.top_k,
                stopping_criteria=stopping_criteria,
            )

        return generated_ids, prompt_lens, context_len

    def build_zero_mask(self, generated_ids, prompt_lens):

        self.toks = self.tokenizer.convert_ids_to_tokens(generated_ids[0])
        #is_punct = torch.Tensor([any([punct in tok for punct in PUNCTUATION]) for tok in toks]).bool().cuda()
        is_punct = torch.zeros_like(generated_ids[0])

        # get whole word masks
        # (BLANK or PUNCTUATION or NEWLINES are starts of words)
        word_begins_mask = torch.Tensor([any([ws in t for ws in WORD_STARTS]) for t in self.toks]).cuda()
        word_begins_mask[0] = 1  # <s> first word default
        word_begins_mask[-1] = 1 # </s> last word if exists
        tok_to_word = (torch.cumsum(word_begins_mask, -1) - 1).int()

        word_begins_idx = word_begins_mask.nonzero().view(-1)
        words = np.split(word_begins_mask, word_begins_idx)[1:]
        word_lens = torch.Tensor(list(map(len, words))).cuda().long()

        content_start = self.pstart_len + self.p_lens[0]
        content_end = content_start + self.p_lens[1]

        word_can_zero_mask = torch.zeros(len(words)).cuda()
        wcontent_start = tok_to_word[content_start]
        wcontent_end = tok_to_word[content_end]
        word_can_zero_mask[wcontent_start:wcontent_end] = 1
        #for p_idx in is_punct.nonzero()[:, 0]:
        #    word_can_zero_mask[tok_to_word[p_idx]] = 0 # don't mask punctuation
        print(f"{word_can_zero_mask.sum().int()} words")

        tok_can_zero_mask = torch.repeat_interleave(word_can_zero_mask, word_lens)

        tok_can_zero_mask = torch.zeros(self.full_len).cuda()
        tok_can_zero_mask[content_start:content_end] = 1
        #tok_can_zero_mask *= ~is_punct # don't mask punctuation

        return tok_can_zero_mask, word_can_zero_mask, word_lens, tok_to_word

    def _build_random_mask_input(
        self,
        nexamples,
        whole_word_masking=True,
        hi_mask_p=None,
        lo_mask_p=None,
        nmask=None,
    ):
        assert (nmask is None) != (hi_mask_p is None), "Must set one of mask_p or nmask"
        masked_inputs = self.gen_ids.clone().repeat(nexamples, 1)

        if whole_word_masking:
            can_zero_mask = self.word_can_zero_mask
            word_lens = self.word_lens
        else:
            can_zero_mask = self.tok_can_zero_mask
            word_lens = torch.ones_like(can_zero_mask).int()

        mask_choices = can_zero_mask.nonzero()[:, 0].cpu().numpy()

        for j in range(nexamples):
            if nmask is None:
                if lo_mask_p is None:
                    mask_p = hi_mask_p
                else:
                    mask_p = np.random.uniform(lo_mask_p, hi_mask_p)

                nmask = int(np.ceil(mask_p * can_zero_mask.sum().item()))

                # must mask at least one thing
                if nmask == 0:
                    nmask += 1

            mask_ids = torch.zeros_like(can_zero_mask).bool()
            to_mask = np.random.choice(mask_choices, size=int(nmask), replace=False)
            mask_ids[to_mask] = 1
            mask_ids = torch.repeat_interleave(mask_ids, word_lens)
            masked_inputs[j][mask_ids] = self.tokenizer.unk_token_id
        return masked_inputs

    def _build_tok_mask_input(self):
        nexamples = self.ntoks
        masked_inputs = self.gen_ids.clone().repeat(nexamples, 1)
        for i in range(nexamples):
            masked_inputs[i][self.trange[0] + i] = self.tokenizer.unk_token_id
        return masked_inputs

    def _build_ngram_mask_input(self, ngram=1):
        nexamples = self.nwords - ngram + 1
        masked_inputs = self.gen_ids.clone().repeat(nexamples, 1)
        can_mask_idxs = self.word_can_zero_mask.nonzero()[:, 0]
        for i in range(nexamples):
            mask_ids = torch.zeros_like(self.word_can_zero_mask).bool()
            mask_ids[can_mask_idxs[i:i+ngram]] = 1
            mask_ids = torch.repeat_interleave(mask_ids, self.word_lens)
            masked_inputs[i][mask_ids] = self.tokenizer.unk_token_id

        return masked_inputs

    def _build_sentence_mask_input(self):
        raise NotImplementedError

    def _build_ngram_pair_mask_input(self, nexamples, ngram=1):
        raise NotImplementedError
        """
        start_ids = np.random.randint(
            low=self.wrange[0],
            high=self.wrange[1],
            size=nexamples,
        )
        offset_ids = np.random.randint(
            low=0,
            high=self.nwords - ngram * 2,
            size=nexamples,
        )
        end_ids = (start_ids + offset_ids) % self.nwords + self.wrange[0]

        masked_inputs = self.gen_ids.clone().repeat(nexamples, 1)
        for i in range(nexamples):
            mask_ids =  torch.zeros_like(self.word_can_zero_mask).bool()
            mask_ids[start_ids[i]:start_ids[i] + ngram] = 1
            mask_ids[end_ids[i]:end_ids[i] + ngram] = 1
            mask_ids = torch.repeat_interleave(mask_ids, self.word_lens)
            masked_inputs[i][mask_ids] = self.tokenizer.unk_token_id

        return masked_inputs
        """

    def _build_ngram_frac_mask_input(self, nexamples, mask_p, ngram=1):
        nmasked_toks = int(np.ceil(self.ntoks * mask_p))

        masked_inputs = self.gen_ids.clone().repeat(nexamples, 1)
        can_mask_idxs = self.word_can_zero_mask.nonzero()[:, 0].cpu().numpy()

        for i in range(nexamples):
            while ((masked_inputs[i] == self.tokenizer.unk_token_id).sum() < nmasked_toks):
                nmasked_curr = (masked_inputs[i] == self.tokenizer.unk_token_id).sum()
                start_idx = np.random.choice(len(can_mask_idxs))

                can_mask_ngram = torch.zeros_like(self.word_can_zero_mask)
                can_mask_ngram[can_mask_idxs[start_idx:start_idx + ngram]] = 1
                can_mask_ngram = torch.repeat_interleave(can_mask_ngram, self.word_lens)

                for j in can_mask_ngram.nonzero()[:, 0]:
                    masked_inputs[i][j] = self.tokenizer.unk_token_id
                    if ((masked_inputs[i] == self.tokenizer.unk_token_id).sum() == nmasked_toks):
                        break

        return masked_inputs

    def _build_shapley_mask_input(self, whole_word_masking=True):
        if whole_word_masking:
            nexamples = self.nwords
            offset = self.wrange[0]
            can_zero_mask = self.word_can_zero_mask
            word_lens = self.word_lens
        else:
            nexamples = self.ntoks
            offset = self.trange[0]
            can_zero_mask = self.tok_can_zero_mask
            word_lens = torch.ones_like(zero_mask)

        masked_inputs = self.gen_ids.clone().repeat(nexamples + 1, 1)

        can_mask_idxs = can_zero_mask.nonzero()[:, 0].cpu().numpy()
        shap_perm = np.random.permutation(can_mask_idxs)

        # mask all tokens for first example
        curr_mask = can_zero_mask.clone().bool()
        mask_ids = torch.repeat_interleave(curr_mask, word_lens)
        masked_inputs[0][mask_ids] = self.tokenizer.unk_token_id

        for i, mask_idx in enumerate(shap_perm):
            # iteratively unmask each token
            curr_mask[mask_idx] = 0
            mask_ids = torch.repeat_interleave(curr_mask, word_lens)
            masked_inputs[i + 1][mask_ids] = self.tokenizer.unk_token_id

        return masked_inputs, shap_perm

    def _build_topk_mask_input(
        self,
        mask_p,
        keep_top=False,
        whole_word_masking=True,
        remove=False,
    ):
        coefs = [self.seq_datamodel.coef_.copy()] + [dmodel.coef_.copy() for dmodel in self.datamodels]
        masked_inputs = self.gen_ids.clone().repeat(len(coefs), 1)

        for i, coef in enumerate(coefs):
            coef = torch.Tensor(coef).cuda()
            if whole_word_masking:
                coef = torch.repeat_interleave(
                    torch.zeros(len(self.word_lens)).cuda().scatter_add_(
                        0,
                        self.tok_to_word.long()[:len(coef)],
                        coef,
                    ) / self.word_lens,
                    self.word_lens
                ) # combine and rescatter weights across words

            nmask = int(np.ceil(mask_p * self.ntoks))
            mask_order = coef.sort()[1].cpu().numpy()
            mask_order = [o for o in mask_order if o in self.tok_can_zero_mask.nonzero()[:, 0].cpu().numpy()]
            mask_order = np.array(mask_order)

            if keep_top:
                masked_inputs[i][mask_order[:-nmask]] = self.tokenizer.unk_token_id
            else:
                masked_inputs[i][mask_order[-nmask:]] = self.tokenizer.unk_token_id

        if remove:
            masked_inputs = masked_inputs[masked_inputs != self.tokenizer.unk_token_id]
            masked_inputs = masked_inputs.view(len(coefs), -1)

        return masked_inputs

    def build_mask_inputs(
        self,
        mask_type,
        seed=None,
        **kwargs,
    ):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        if mask_type == 'ngram':
            return self._build_ngram_mask_input(**kwargs)
        elif mask_type == 'ngram_pair':
            return self._build_ngram_pair_mask_input(**kwargs)
        elif mask_type == 'tok':
            return self._build_tok_mask_input()
        elif mask_type == 'random':
            return self._build_random_mask_input(**kwargs)
        elif mask_type == 'shapley':
            return self._build_shapley_mask_input(**kwargs)
        elif mask_type == 'ngram_frac':
            return self._build_ngram_frac_mask_input(**kwargs)
        elif mask_type == 'topk':
            return self._build_topk_mask_input(**kwargs)
        else:
            raise NotImplementedError(f"Mask type {mask_type} not supported")

    @torch.no_grad
    def gen_logits(self, masks, silent=True):
        nbatches = int(np.ceil(len(masks) / self.bsize))
        seq_logits = []
        logits = []

        for i in tqdm(range(nbatches), disable=silent):

            batch = masks[i * self.bsize: (i + 1) * self.bsize]
            batch_ids = self.gen_ids.expand(len(batch), -1)
            outs = self.model(batch)

            seq_log_probs = torch.take_along_dim(F.log_softmax(outs.logits, -1)[:, :-1], batch_ids[:, 1:, None], dim=-1)[..., 0]
            seq_probs = torch.take_along_dim(F.softmax(outs.logits, -1)[:, :-1], batch_ids[:, 1:, None], dim=-1)[..., 0]
            seq_logp = seq_log_probs[:, -self.gen_len:].sum(-1)
            seq_logits.append(seq_logp - torch.log(1 - torch.exp(seq_logp) + self.eps))

            out_logits = seq_log_probs - \
                    torch.log(1 - seq_probs + self.eps) # prevent overflow
            logits.append(out_logits)

        logits = torch.concatenate(logits)
        seq_logits = torch.concatenate(seq_logits)

        return logits, seq_logits

    def fit(
        self,
        mask_type='random',
        **kwargs,
    ):
        print(f"Training datamodel with {mask_type} mask")

        # build training data
        train_masks = self.build_mask_inputs(
            mask_type,
            **kwargs,
        )
        train_logits, train_seq_logits = self.gen_logits(train_masks, silent=False)
        train_masks = (train_masks != self.tokenizer.unk_token_id).int()

        # move to CPU for sklearn
        train_masks = train_masks.cpu().numpy()
        train_logits = train_logits.cpu().numpy()
        train_seq_logits = train_seq_logits.cpu().numpy()

        self.datamodels = []
        for i in range(self.c_len, self.full_len):
            self.datamodels.append(Lasso(alpha=self.alpha).fit(
                train_masks[:, :i],
                train_logits[:, i-1],
            ))

        self.seq_datamodel = Lasso(alpha=self.alpha).fit(train_masks[:, :self.c_len], train_seq_logits)

    def gen(self, prompt, gen_start):
        gen_res = self.generate(prompt, gen_start, self.max_gen_len)
        self.gen_ids, self.p_lens, self.c_len = gen_res
        self.full_len = len(self.gen_ids[0])
        self.gen_len = self.full_len - self.c_len

        print("=====================PROMPT=====================")
        print(self.tokenizer.batch_decode(self.gen_ids[..., :self.c_len])[0])
        print("===================GENERATION===================")
        print(self.tokenizer.batch_decode(self.gen_ids[..., self.c_len:])[0])
        print("================================================")

        zero_mask_res = self.build_zero_mask(self.gen_ids, self.p_lens)
        self.tok_can_zero_mask, self.word_can_zero_mask, self.word_lens, self.tok_to_word = zero_mask_res

        self.trange = [self.pstart_len + self.p_lens[0], self.pstart_len + self.p_lens[0] + self.p_lens[1]]
        self.ntoks = self.tok_can_zero_mask.sum().int().cpu().item()

        wcontent_start = self.tok_to_word[self.trange[0]].cpu().item()
        wcontent_end = self.tok_to_word[self.trange[1]].cpu().item()
        self.wrange = [wcontent_start, wcontent_end]
        self.nwords = self.word_can_zero_mask.sum().int().cpu().item()

    def gen_and_fit(
        self,
        prompt,
        gen_start,
        mask_type='random',
        nepochs=10,
        **kwargs,
    ):

        self.gen(prompt, gen_start)

        if mask_type == 'random':
            if 'whole_word_masking' in kwargs and not kwargs['whole_word_masking']:
                nexamples = nepochs * self.ntoks
            else:
                nexamples = nepochs * self.nwords
            kwargs['nexamples'] = nexamples

        self.fit(mask_type, **kwargs)

    @torch.no_grad
    def eval_topk(
        self,
        topks=[0.05, 0.1, 0.15, 0.2, 0.25],
        max_p=1.0,
        whole_word_masking=False,
    ):
        base_out = self.model(self.gen_ids).logits
        base_probs = torch.take_along_dim(
            F.softmax(base_out, -1)[:, :-1],
            self.gen_ids[:, 1:, None],
            dim=-1,
        )[0, :, 0][-self.gen_len:].detach()

        scores = defaultdict(list)

        for k in topks:
            for keep_top in [True, False]:
                for remove in [True, False]:
                    if keep_top:
                        fname = 'suff'
                        mask_p = 1 - max_p + k
                    else:
                        fname = 'comp'
                        mask_p = k
                    ftype = 'remove' if remove else 'zero'
                    masks = self.build_mask_inputs(
                        'topk',
                        mask_p=mask_p,
                        keep_top=keep_top,
                        whole_word_masking=whole_word_masking,
                        remove=remove,
                    )

                    nbatches = int(np.ceil(len(masks) / self.bsize))
                    for i in range(nbatches):
                        batch = masks[i * self.bsize:(i + 1) * self.bsize]
                        out_logits = self.model(batch).logits
                        probs = torch.take_along_dim(
                            F.softmax(out_logits, -1)[:, :-1],
                            batch[:, 1:, None],
                            dim=-1,
                        )[:, -self.gen_len:, 0]
                        if i == 0:
                            seq_score = (2**(base_probs.log2().mean()) - 2**(probs[0].log2().mean(-1)))
                            seq_score = seq_score.detach().item()
                            scores[f"{fname}_{ftype}_seq"] = seq_score
                            probs = probs[1:]
                            start_idx = 0
                        else:
                            start_idx = i * self.bsize - 1
                        tok_scores = (base_probs - probs).detach().cpu().numpy()[:, start_idx:].diagonal()
                        scores[f"{fname}_{ftype}_tok"].append(tok_scores)

        for k, v in scores.items():
            if isinstance (v, list):
                scores[k] = np.mean(np.concatenate(v))

        return scores

    @torch.no_grad
    def eval_lds(
        self,
        mask_type='random',
        seed=None,
        cache=True,
        **kwargs,
    ):
        mask_key = (mask_type, frozenset(kwargs.items()))
        if mask_key in self.eval_cache:
            test_masks, test_logits, test_seq_logits = self.eval_cache[mask_key]
        else:
            test_masks = self.build_mask_inputs(mask_type, seed, **kwargs)
            test_logits, test_seq_logits = self.gen_logits(test_masks)

            test_masks = (test_masks != self.tokenizer.unk_token_id).int().cpu().numpy()
            test_logits = test_logits.cpu().numpy()
            test_seq_logits = test_seq_logits.cpu().numpy()
            self.eval_cache[mask_key] = [test_masks, test_logits, test_seq_logits]

        seq_pred = self.seq_datamodel.predict(test_masks[:, :self.c_len])

        spearmans = []
        for i in range(self.c_len, self.full_len):
            pred = self.datamodels[i - self.c_len].predict(test_masks[:, :i])
            lds = spearmanr(pred, test_logits[:, i-1])[0]
            if np.isnan(lds):
                lds = 0
            spearmans.append(lds)

        seq_spearman = spearmanr(seq_pred, test_seq_logits)[0]
        if np.isnan(seq_spearman):
            seq_spearman = 0

        return np.mean(spearmans), seq_spearman

    def fit_shap(self, nepochs=1, whole_word_masking=True):
        print(f"Calculating sampled Shapley values")

        # num context x num generated
        if whole_word_masking:
            nshap = len(self.word_can_zero_mask)
        else:
            nshap = len(self.tok_can_zero_mask)

        shap_values = torch.zeros(self.full_len, self.gen_len)
        seq_shap_value = torch.zeros(self.c_len)

        for i in tqdm(range(nepochs)):
            shap_masks, shap_perm = self.build_mask_inputs(
                'shapley',
                whole_word_masking=whole_word_masking
            )

            shap_logits, shap_seq_logits = self.gen_logits(shap_masks, silent=True)
            shap_logits = shap_logits[:, -self.gen_len:] # off by 1 since we don't calculate p(<s>)

            base_shap = torch.zeros(nshap, self.gen_len)
            base_seq_shap = torch.zeros(nshap)
            for i in range(len(shap_logits) - 1):
                base_shap[shap_perm[i]] = shap_logits[i + 1] - shap_logits[i]
                base_seq_shap[shap_perm[i]] = shap_seq_logits[i + 1] - shap_seq_logits[i]

            if whole_word_masking:
                base_shap = torch.repeat_interleave(base_shap, self.word_lens.cpu(), dim=0)
                base_seq_shap = torch.repeat_interleave(base_seq_shap, self.word_lens.cpu(), dim=0)

            shap_values += base_shap
            seq_shap_value += base_seq_shap[:self.c_len]

        shap_values /= nepochs
        seq_shap_value /= nepochs

        self.datamodels = []
        for i in range(self.gen_len):
            self.datamodels.append(Lasso())

            # build coefficients
            base_coef = shap_values[:self.c_len + i, i].cpu().numpy()

            self.datamodels[-1].coef_ = base_coef
            self.datamodels[-1].intercept_ = 0

        self.seq_datamodel = Lasso()
        self.seq_datamodel.coef_ = seq_shap_value.cpu().numpy()
        self.seq_datamodel.intercept_ = 0

    def fit_attention(self, layer=None):
        print("Calculating attention weights")

        out = self.model(self.gen_ids, output_attentions=True)

        # average attn across all layers and heads
        if layer is not None:
            attns = out.attentions[layer][0]
        else:
            attns = torch.stack(out.attentions).mean(0)[0]
        attns = attns.mean(0).detach().cpu().numpy()

        self.datamodels = []
        for i in range(self.gen_len):
            self.datamodels.append(Lasso())
            a_idx = self.c_len + i
            self.datamodels[-1].coef_ = attns[a_idx, :a_idx]
            self.datamodels[-1].intercept_ = 0

        self.seq_datamodel = Lasso()
        self.seq_datamodel.coef_ = attns[self.c_len:, :self.c_len].mean(0)
        self.seq_datamodel.intercept_ = 0

    def fit_grads(self, nsteps=1, sequential=False):

        # initialize datamodels
        self.datamodels = [Lasso() for _ in range(self.gen_len)]
        self.seq_datamodel = Lasso()
        for g_idx in range(self.gen_len):
            self.datamodels[g_idx].coef_ = np.zeros(self.c_len + g_idx)
            self.datamodels[g_idx].intercept_ = 0
        self.seq_datamodel.coef_ = np.zeros(self.c_len)
        self.seq_datamodel.intercept_ = 0

        if sequential:
            print(f"Calculating sequential integrated gradients with {nsteps} steps")
            for t_idx in tqdm(self.tok_can_zero_mask.nonzero()[:, 0]):
                base_embs = self.model.model.embed_tokens(self.gen_ids).detach()
                base_embs[:, t_idx] = 0
                tok_coef, seq_coef = self.get_grad_weights(base_embs, nsteps=nsteps, silent=True)
                for g_idx in range(self.gen_len):
                    self.datamodels[g_idx].coef_[t_idx] = tok_coef[g_idx, t_idx]
                self.seq_datamodel.coef_[t_idx] = seq_coef[t_idx]
        else:
            base_embs = (self.model.model.embed_tokens(self.gen_ids * ~(self.tok_can_zero_mask.bool()))).detach()
            if nsteps == 1:
                print("Calculating saliencey scores")
            else:
                print(f"Calculating integrated gradients with {nsteps} steps")
            tok_coef, seq_coef = self.get_grad_weights(base_embs, nsteps=nsteps)
            for g_idx in range(self.gen_len):
                self.datamodels[g_idx].coef_ = tok_coef[g_idx][:self.c_len + g_idx]
            self.seq_datamodel.coef_ = seq_coef

    def get_grad_weights(self, base_embs, nsteps=1, silent=False):
        tok_coef = np.zeros((self.gen_len, self.full_len))
        seq_coef = np.zeros(self.c_len)

        gen_embs = (self.model.model.embed_tokens(self.gen_ids)).detach()
        emb_diff = (gen_embs - base_embs)

        for n in tqdm(range(nsteps), disable=silent):
            step_embs = base_embs + emb_diff * (n + 1) / nsteps # gen_embs if nsteps=1
            step_embs.requires_grad = True
            step_embs.retain_grad()
            outs = self.model(inputs_embeds=step_embs)

            seq_probs = torch.take_along_dim(
                F.log_softmax(outs.logits, -1)[:, :-1],
                self.gen_ids[:, 1:, None],
                dim=-1,
            )[0, :, 0]

            for g_idx in range(self.gen_len):
                t_idx = self.c_len + g_idx
                loss = seq_probs[t_idx - 1]
                loss.backward(retain_graph=True)
                salience = (step_embs.grad * emb_diff)[0].sum(-1)
                coef = salience * self.tok_can_zero_mask
                tok_coef[g_idx] += coef.detach().cpu().numpy()

            full_loss = torch.sum(seq_probs[-self.gen_len:])
            full_loss.backward()
            full_salience = (step_embs.grad * emb_diff)[0].sum(-1)
            full_coef = full_salience[:self.c_len] * self.tok_can_zero_mask[:self.c_len]
            seq_coef += full_coef.detach().cpu().numpy()

        return tok_coef / nsteps, seq_coef / nsteps

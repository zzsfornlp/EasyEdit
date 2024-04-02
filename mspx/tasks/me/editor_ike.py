#

# helpers for IKE

import os
from typing import Dict, Any, List
import pickle
import torch

from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
from easyeditor.models.ike import IKEHyperParams

from .inst import Item
from .utils import get_first_param, get_logger

LOG = get_logger()

class IKEHelper:
    def __init__(self, hparams: IKEHyperParams, model, train_insts):
        self.hparams = hparams
        self.device = get_first_param(model).device
        self.sentence_model = SentenceTransformer(hparams.sentence_model_name).to(self.device)
        # --
        _file = self.get_cache_filename(train_insts)
        if _file and os.path.exists(_file):  # load from cache
            with open(_file, "rb") as fIn:
                self._cache = pickle.load(fIn)
            LOG.info(f"Load pre-calculated cache from {_file}")
        else:  # calculate
            self._cache = self.encode_ike_facts(train_insts, _file)
        # --

    def get_cache_filename(self, train_insts):
        import hashlib
        m = hashlib.sha256()
        # --
        hparams = self.hparams
        safe_model_name = hparams.sentence_model_name.rsplit('/', 1)[-1]
        _sig0 = '__'.join([train_insts[ii]['edit'][0]['answer'] for ii in [0, -1, len(train_insts)//2]])
        m.update(_sig0.encode('utf-8'))
        _sig = f"{len(train_insts)}_{m.hexdigest()[:8]}"
        ret = f"_A{hparams.alg_name}_M{safe_model_name}_{_sig}.pkl"
        return ret

    def encode_ike_facts(self, train_insts, output_file):
        sentences = []
        for i, _inst in enumerate(train_insts):
            all_facts = [_item.format() for _item in _inst['edit']]
            new_fact = "; ".join(all_facts)
            for kk in ["edit", "rephrase", "locality"]:
                for _item in _inst[kk]:
                    sentences.append(f"New Fact: {new_fact}\nPrompt: {_item.format()}\n\n")
        embeddings = self.sentence_model.encode(sentences)
        ret = (sentences, embeddings)
        if output_file:
            with open(output_file, "wb") as fOut:
                pickle.dump(ret, fOut, protocol=pickle.HIGHEST_PROTOCOL)
        return ret

    def retrieve_ike_facts(self, request: List[Dict]):
        stored_sentences, stored_embeddings = self._cache
        stored_embeddings = torch.tensor(stored_embeddings).to(self.device)
        stored_embeddings = util.normalize_embeddings(stored_embeddings)
        # encode and select; todo(note): simply cat things together!
        all_facts = [Item.format_dict(z) for z in request]
        new_fact = "; ".join(all_facts)
        query_sentence = f"New Fact: {new_fact}\nPrompt: {all_facts[0]}\n\n"  # note: simply use the first one!
        query_embedding = util.normalize_embeddings(torch.tensor(self.sentence_model.encode(
            query_sentence, show_progress_bar=False)).unsqueeze(0).to(self.device))
        hits = util.semantic_search(query_embedding, stored_embeddings, score_function=util.dot_score, top_k=self.hparams.k)
        assert len(hits) == 1
        hit = hits[0]
        icl_examples = [stored_sentences[hit[k]["corpus_id"]] for k in range(len(hit))]
        icl_examples.extend([f"New Fact: {z}\nPrompt: {z}\n\n" for z in all_facts])  # add all original edits
        # --
        _prompt_prefix = "".join(icl_examples) + f"New Fact: {new_fact}\nPrompt: "
        return _prompt_prefix

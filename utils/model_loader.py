from transformers import (
    AutoTokenizer,
    AutoConfig,
    BertConfig,
    DistilBertConfig,
    XLNetForSequenceClassification,
    XLNetConfig,
)

from pathlib import Path
from safetensors.torch import load_file as st_load
from typing import Dict
import json
import torch



# GPU support.
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = get_device()



# Add top5k tokens from joana study
def add_tokens_from_file(token_file, tokenizer):
    with open(token_file, "r", encoding="utf-8") as f:
        new_tokens = [ln.strip() for ln in f if ln.strip()]
    to_add = [t for t in new_tokens if t not in tokenizer.get_vocab()]
    if to_add:
        tokenizer.add_tokens(to_add)



# ---------- model/tokenizer pretrain pull ----------
def select_tokenizer_and_pretrained_model(repo_root: str, model_name: str, num_labels: int, use_normalised_tokens: bool = False):

    vocab_path = repo_root / "utils" / "vocab" / "CVSS_5k.vocab"
    SPECIAL_TOKENS = {"additional_special_tokens": ["<IP>", "<IPPORT>", "<VER>", "<CVE>", "<PATH>"]}


    if model_name == "distilbert":
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")
        add_tokens_from_file(vocab_path, tok)
        # load PRETRAINED weights
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-cased", num_labels=num_labels
        )
        if use_normalised_tokens:
            tok.add_special_tokens(SPECIAL_TOKENS)
        # resize AFTER adding tokens
        model.resize_token_embeddings(len(tok))
        model.config.vocab_size = len(tok)
        return tok, model
    # elif model_name == "lrp-distilbert":
    #     from transformers import DistilBertConfig, DistilBertTokenizerFast
    #     from transformers import DistilBertForSequenceClassification as HF_DistilCls
    #     from utils.distilbert_lrp.bert_explainability.distilbert.DistilBertForSequenceClassification import (
    #         DistilBertForSequenceClassification as LRPDistilCls,
    #     )

    #     tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")

    #     cfg = DistilBertConfig.from_pretrained("distilbert-base-cased")
    #     cfg.num_labels = num_labels
    #     lrp = LRPDistilCls(cfg)

    #     base_sd = HF_DistilCls.from_pretrained(
    #         "distilbert-base-cased", num_labels=num_labels
    #     ).state_dict()

    #     missing, unexpected = lrp.load_state_dict(base_sd, strict=False)

    #     # Only allow head differences. Everything else should match.
    #     allowed_missing = ("classifier", "pre_classifier")
    #     bad_missing = [k for k in missing if not k.startswith(allowed_missing)]
    #     bad_unexpected = [k for k in unexpected if not k.startswith(allowed_missing)]
    #     if bad_missing or bad_unexpected:
    #         raise RuntimeError(
    #             "LRP DistilBERT backbone mismatch.\n"
    #             f"bad_missing[:10]={bad_missing[:10]}\n"
    #             f"bad_unexpected[:10]={bad_unexpected[:10]}"
    #         )

    #     add_tokens_from_file(vocab_path, tok)
    #     if use_normalised_tokens:
    #         tok.add_special_tokens(SPECIAL_TOKENS)
    #     lrp.resize_token_embeddings(len(tok))
    #     lrp.config.vocab_size = len(tok)

    #     #  Good hygiene
    #     lrp.config.id2label = {i: c for i, c in enumerate(range(num_labels))}
    #     lrp.config.label2id = {str(v): k for k, v in lrp.config.id2label.items()}
    #     lrp.config.problem_type = "single_label_classification"

    #     return tok, lrp

    elif model_name == "lrp-distilbert":
        from transformers import DistilBertTokenizerFast
        from utils.lrp_distilbert_hf import LRPDistilBertForSequenceClassification

        tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")
        add_tokens_from_file(vocab_path, tok)
        if use_normalised_tokens:
            tok.add_special_tokens(SPECIAL_TOKENS)

        model = LRPDistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-cased",
            num_labels=num_labels,
        )

        model.resize_token_embeddings(len(tok))
        model.config.vocab_size = len(tok)

        model.config.problem_type = "single_label_classification"
        model.config.id2label = {i: str(i) for i in range(num_labels)}
        model.config.label2id = {v: k for k, v in model.config.id2label.items()}

        return tok, model
    
    elif model_name == "lrp-bert":
        # Minimal LRP-enabled BERT: initialise custom LRP head, load HF backbone weights, extend vocab, resize
        from transformers import BertConfig, BertTokenizerFast, BertForSequenceClassification as HF_BertCls
        # Adjust the import path below to your LRP BERT class location if different
        from utils.distilbert_lrp.bert_explainability.distilbert.BertForSequenceClassification import (
            BertForSequenceClassification as LRPBertCls,
        )

        cfg = BertConfig.from_pretrained("bert-base-cased")
        cfg.num_labels = num_labels
        model = LRPBertCls(cfg)

        # Load standard BERT classifier weights into the LRP model (classifier differences are expected)
        base_sd = HF_BertCls.from_pretrained("bert-base-cased", num_labels=num_labels).state_dict()
        model.load_state_dict(base_sd, strict=False)

        # Tokeniser + vocab extension
        tok = BertTokenizerFast.from_pretrained("bert-base-cased")
        add_tokens_from_file(vocab_path, tok)
        if use_normalised_tokens:
            tok.add_special_tokens(SPECIAL_TOKENS)

        # Resize embeddings after adding tokens/special tokens
        model.resize_token_embeddings(len(tok))
        model.config.vocab_size = len(tok)

        # Minimal config hygiene
        model.config.problem_type = "single_label_classification"
        model.config.id2label = {i: str(i) for i in range(num_labels)}
        model.config.label2id = {v: k for k, v in model.config.id2label.items()}

        return tok, model

    elif model_name == "xlnet":
        # Hugging Face XLNet (SentencePiece) classifier
        # Notes:
        # - XLNet has a pad token in the vocab; make sure model.config.pad_token_id is set.
        # - Adding tokens works with *Fast tokenizers too; we still resize embeddings after.
        from transformers import XLNetTokenizerFast, XLNetForSequenceClassification

        tok = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
        add_tokens_from_file(vocab_path, tok)
        if use_normalised_tokens:
            tok.add_special_tokens(SPECIAL_TOKENS)

        model = XLNetForSequenceClassification.from_pretrained(
            "xlnet-base-cased", num_labels=num_labels
        )

        # Ensure pad token id is carried to the model config (XLNet sometimes needs this explicit)
        if tok.pad_token_id is not None:
            model.config.pad_token_id = tok.pad_token_id

        # IMPORTANT: resize after adding any tokens/special tokens
        model.resize_token_embeddings(len(tok))
        model.config.vocab_size = len(tok)

        # Minimal config hygiene (keep consistent with your other branches)
        model.config.problem_type = "single_label_classification"
        model.config.id2label = {i: str(i) for i in range(num_labels)}
        model.config.label2id = {v: k for k, v in model.config.id2label.items()}

        return tok, model

    elif model_name == "roberta":
        from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
        tok = RobertaTokenizerFast.from_pretrained("roberta-base")
        add_tokens_from_file(vocab_path, tok)
        if use_normalised_tokens:
            tok.add_special_tokens(SPECIAL_TOKENS)

        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=num_labels
        )
        model.resize_token_embeddings(len(tok))
        model.config.vocab_size = len(tok)
        return tok, model

    elif model_name == "deberta":
        from transformers import DebertaTokenizerFast, DebertaForSequenceClassification
        tok = DebertaTokenizerFast.from_pretrained("microsoft/deberta-base")
        add_tokens_from_file(vocab_path, tok)
        if use_normalised_tokens:
            tok.add_special_tokens(SPECIAL_TOKENS)

        model = DebertaForSequenceClassification.from_pretrained(
            "microsoft/deberta-base", num_labels=num_labels
        )
        model.resize_token_embeddings(len(tok))
        model.config.vocab_size = len(tok)
        return tok, model

    elif model_name == "albert":
        from transformers import AlbertTokenizerFast, AlbertForSequenceClassification
        tok = AlbertTokenizerFast.from_pretrained("albert-base-v2")
        add_tokens_from_file(vocab_path, tok)
        if use_normalised_tokens:
            tok.add_special_tokens(SPECIAL_TOKENS)

        model = AlbertForSequenceClassification.from_pretrained(
            "albert-base-v2", num_labels=num_labels
        )
        model.resize_token_embeddings(len(tok))
        model.config.vocab_size = len(tok)
        return tok, model

    elif model_name == "bert":
        from transformers import BertTokenizerFast, BertForSequenceClassification
        tok = BertTokenizerFast.from_pretrained("bert-base-cased")
        add_tokens_from_file(vocab_path, tok)
        if use_normalised_tokens:
            tok.add_special_tokens(SPECIAL_TOKENS)

        model = BertForSequenceClassification.from_pretrained(
            "bert-base-cased", num_labels=num_labels
        )
        
        model.resize_token_embeddings(len(tok))
        model.config.vocab_size = len(tok)
        return tok, model
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return tok, model




# ====================== FINE TUNED LOADERS ===================================
def _ensure_pad_token(tok, model_type: str):
    if tok.pad_token_id is None:
        if getattr(tok, "eos_token", None):
            tok.pad_token = tok.eos_token
        elif getattr(tok, "sep_token", None):
            tok.pad_token = tok.sep_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
    if model_type == "xlnet":
        tok.padding_side = "left"

def _find_model_weights_strict(run_dir: Path):
    p = run_dir / "model.safetensors"
    if not p.exists():
        raise FileNotFoundError(f"Expected model.safetensors under {run_dir}")
    return st_load(str(p))

# ---------------- strict model loading ----------------
def load_cfg(run_dir: Path) -> Dict:
    p = run_dir / "config.json"
    if not p.exists():
        raise FileNotFoundError(f"config.json not found under {run_dir}")
    return json.loads(p.read_text(encoding="utf-8"))



def _enforce_vocab_alignment(tok, model):
    emb = model.get_input_embeddings()
    if emb is None:
        return
    tok_len = int(len(tok))
    emb_len = int(emb.num_embeddings)
    if emb_len != tok_len:
        model.resize_token_embeddings(tok_len)


def _lrp_bert_load(run_dir: Path, tok):
    from utils.distilbert_lrp.bert_explainability.distilbert.BertForSequenceClassification import (
        BertForSequenceClassification as LRPBertCls
    )

    hf_cfg = BertConfig.from_pretrained(run_dir, local_files_only=True)
    hf_cfg.vocab_size = len(tok)
    model = LRPBertCls(hf_cfg)
    state = _find_model_weights_strict(run_dir)
    model.load_state_dict(state, strict=False)
    model.resize_token_embeddings(len(tok))
    return model


def _lrp_distilbert_load(run_dir: Path, tok):
    from transformers import DistilBertConfig
    from transformers import DistilBertForSequenceClassification
    from utils.lrp_distilbert_hf import LRPDistilBertForSequenceClassification

    hf_cfg = DistilBertConfig.from_pretrained(run_dir, local_files_only=True)
    hf_cfg.vocab_size = len(tok)

    cls = LRPDistilBertForSequenceClassification
    model = cls(hf_cfg)

    state = _find_model_weights_strict(run_dir)
    # If you saved via Trainer.save_model(), this might be pytorch_model.bin not safetensors.
    # You are using model.safetensors, so keep strict file expectations as you do now.
    model.load_state_dict(state, strict=False)

    model.resize_token_embeddings(len(tok))
    return model


def _xlnet_load(run_dir: Path, tok):
    state = _find_model_weights_strict(run_dir)
    emb_w = state["transformer.word_embedding.weight"]
    vocab_size_ckpt, d_model = emb_w.shape

    q_shape = state["transformer.layer.0.rel_attn.q"].shape
    n_head = int(q_shape[1])
    d_head = int(q_shape[2])
    d_inner = int(state["transformer.layer.0.ff.layer_1.weight"].shape[0])

    n_layer = 0
    while f"transformer.layer.{n_layer}.ff.layer_1.weight" in state:
        n_layer += 1

    if "logits_proj.weight" in state:
        num_labels = int(state["logits_proj.weight"].shape[0])
    else:
        num_labels = int(load_cfg(run_dir).get("num_labels", 2))

    hf_cfg = XLNetConfig(
        model_type="xlnet",
        d_model=d_model,
        n_head=n_head,
        d_head=d_head,
        d_inner=d_inner,
        n_layer=n_layer,
        ff_activation="gelu",
        untie_r=True,
        vocab_size=vocab_size_ckpt,
        num_labels=num_labels,
    )
    model = XLNetForSequenceClassification(hf_cfg)
    model.load_state_dict(state, strict=False)
    if len(tok) != vocab_size_ckpt and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tok))
    return model


def load_model_and_tokenizer(run_dir: Path):
    cfg = load_cfg(run_dir)
    hfconf = AutoConfig.from_pretrained(run_dir, local_files_only=True)
    model_type = getattr(hfconf, "model_type", "").lower()
    use_fast = False if model_type == "xlnet" else True
    tok = AutoTokenizer.from_pretrained(run_dir, local_files_only=True, use_fast=use_fast)

    _ensure_pad_token(tok, model_type)
    if model_type == "xlnet":
        if (
            getattr(tok, "model_max_length", None) is None
            or tok.model_max_length <= 0
            or tok.model_max_length > 10000
        ):
            tok.model_max_length = int(cfg.get("max_length", 512))

    if model_type == "xlnet":
        model = _xlnet_load(run_dir, tok)
    elif model_type == "bert":
        model = _lrp_bert_load(run_dir, tok)
    elif model_type == "distilbert":
        model = _lrp_distilbert_load(run_dir, tok)
    else:
        raise RuntimeError(f"Unsupported model_type for strict loader: {model_type}")

    _enforce_vocab_alignment(tok, model)
    model.to(DEVICE).eval()
    return cfg, tok, model, model_type

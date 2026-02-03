# training_scripts/extended_analysis/train.py
from __future__ import annotations

import argparse, json, time
from pathlib import Path

from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, set_seed
from sklearn.model_selection import StratifiedKFold
import numpy as np

# ---- make repo-root imports work when running as a script ----
from pathlib import Path
import sys

# repo_root = <...>/CVSS_THESIS_REPO
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
# ----------------------------------------------------------------

# silence tokenizer warnings
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from utils.eval_metrics import hf_compute_metrics, trainer_predict, compute_all_metrics
from utils.cvss_data_engine_joana.CVSSDataset_modified import CVSSDataset, read_cvss_csv
from utils.text_normalise import normalise_text
from utils.cvss_mappings import METRIC_TO_CLASSES, METRIC_TO_COLIDX, metric_config
from utils.model_loader import select_tokenizer_and_pretrained_model



# ---------- helpers ----------
def build_run_dir(dataset_name: str, seed: int, model_name: str, metric: str) -> Path:
    run = repo_root / "saved_models" / dataset_name / str(seed) / model_name / metric.lower()
    (run / "tb").mkdir(parents=True, exist_ok=True)
    (run / "artifacts").mkdir(parents=True, exist_ok=True)
    return run

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    # required inputs
    ap.add_argument("--data_file", required=True, help="File under cve_data/extended_analysis/")
    ap.add_argument("--metric", required=True,
                    choices=["av", "ac", "pr", "ui", "s", "c", "i", "a"],
                    help="Which CVSS metric column to learn")
    # optional overrides (usually not needed)
    ap.add_argument("--classes_names_override", type=str, help="Comma-separated class names")
    ap.add_argument("--num_labels_override", type=int)
    ap.add_argument("--label_position_override", type=int)

    # split
    ap.add_argument("--val_size", type=float, default=0.0)   # 0.0 => no validation (e.g., 80/20 train/test)
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--kfold", type=int, default=0, help="If >1 and no val split is provided, run stratified K-fold CV on the train pool, report mean/std metrics, then retrain on the full pool and test once.")

    # model/training
    ap.add_argument("--model", default="distilbert",
                    choices=["distilbert", "bert", "roberta", "deberta", "albert", "xlnet", "lrp-distilbert", "lrp-bert"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train_batch", type=int, default=8)
    ap.add_argument("--eval_batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--warmup_ratio", type=float, default=0.0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--metric_for_best", default="f1_weighted",
                    choices=["f1_weighted", "f1_macro", "f1_micro",
                             "accuracy", "balanced_accuracy",
                             "precision_weighted", "recall_weighted"])
    ap.add_argument("--use_normalised_tokens", action="store_true",
                help="Enable regex normalisation (<IP>, <IPPORT>, <VER>, <CVE>, <PATH>) "
                     "and extend tokenizer with these special tokens.")

    
    #sanity check joana
    ap.add_argument("--eval_on_test", action="store_true",
                help="Evaluate on test set during training (parity/sanity check only)")

    # seed + misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_attentions", action="store_true")
    args = ap.parse_args()

    model_dir_name = args.model + ("-normtok" if args.use_normalised_tokens else "")

    set_seed(args.seed)

    # Defaults from mappings (overridable)
    default_classes, default_num_labels, default_label_pos = metric_config(args.metric)
    classes = (args.classes_names_override.split(",")
               if args.classes_names_override else default_classes)
    num_labels = args.num_labels_override if args.num_labels_override else default_num_labels
    label_pos = args.label_position_override if args.label_position_override else default_label_pos

    csv_path = (repo_root / "cve_data" / "extended_analysis" / args.data_file)
    dataset_name = Path(args.data_file).stem

    # load
    all_texts, all_labels = read_cvss_csv(str(csv_path), label_pos, classes)

    # split (stratified)
    has_val = bool(args.val_size and args.val_size > 0.0)

    x_tmp, x_test, y_tmp, y_test = train_test_split(
        all_texts, all_labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=all_labels
    )

    

    if has_val:
        # val fraction is relative to the remaining pool (same as before)
        val_size_adj = args.val_size / (1.0 - args.test_size)
        x_train, x_val, y_train, y_val = train_test_split(
            x_tmp, y_tmp,
            test_size=val_size_adj,
            random_state=args.seed,
            stratify=y_tmp
        )
    else:
        x_train, y_train = x_tmp, y_tmp
        x_val, y_val = [], []  # empty

    # model + tokenizer
    tokenizer, model = select_tokenizer_and_pretrained_model(repo_root, args.model, num_labels, use_normalised_tokens=args.use_normalised_tokens)

    use_norm = args.use_normalised_tokens

    x_train_norm = [normalise_text(t, enabled=use_norm) for t in x_train]
    x_test_norm  = [normalise_text(t, enabled=use_norm) for t in x_test]
    if has_val:
        x_val_norm = [normalise_text(t, enabled=use_norm) for t in x_val]

    train_enc = tokenizer(x_train_norm, truncation=True, padding="max_length", max_length=512)
    test_enc  = tokenizer(x_test_norm,  truncation=True, padding="max_length", max_length=512)
    if has_val:
        val_enc = tokenizer(x_val_norm, truncation=True, padding="max_length", max_length=512)


    # datasets
    train_ds = CVSSDataset(train_enc, y_train)
    test_ds  = CVSSDataset(test_enc,  y_test)
    val_ds   = CVSSDataset(val_enc,   y_val) if has_val else None

    # run dir includes metric
    run_dir = build_run_dir(dataset_name, args.seed, model_dir_name, args.metric)
    
    
    use_kfold = (not has_val) and (not args.eval_on_test) and (args.kfold and args.kfold > 1)

    if use_kfold:
        print(f"[CV] Running Stratified {args.kfold}-fold cross-validation on train pool; "
            f"seed={args.seed}. Test set remains untouched.")

        # Prepare containers
        cv_fold_metrics = []   # list of dicts returned by trainer.evaluate() on each fold's val set
        cv_fold_sizes   = []   # number of validation samples per fold (for sanity)

        # We will (re)tokenise per fold to keep it simple and avoid tricky indexing on encodings
        skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        indices = np.arange(len(x_tmp))
        labels_arr = np.array(y_tmp)

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(indices, labels_arr), start=1):
            x_tr = [x_tmp[i] for i in tr_idx]
            y_tr = [y_tmp[i] for i in tr_idx]
            x_va = [x_tmp[i] for i in va_idx]
            y_va = [y_tmp[i] for i in va_idx]

            # Fresh model + tokenizer per fold (standard CV practice)
            tokenizer_fold, model_fold = select_tokenizer_and_pretrained_model(repo_root, args.model, num_labels, use_normalised_tokens=args.use_normalised_tokens)

            x_tr_norm = [normalise_text(t, enabled=use_norm) for t in x_tr]
            x_va_norm = [normalise_text(t, enabled=use_norm) for t in x_va]
            enc_tr = tokenizer_fold(x_tr_norm, truncation=True, padding="max_length", max_length=512)
            enc_va = tokenizer_fold(x_va_norm, truncation=True, padding="max_length", max_length=512)


            ds_tr = CVSSDataset(enc_tr, y_tr)
            ds_va = CVSSDataset(enc_va, y_va)

            # Each fold gets its own subdir
            fold_dir = build_run_dir(dataset_name, args.seed, model_dir_name, args.metric) / f"cv_fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            ta_cv = TrainingArguments(
                output_dir=str(fold_dir),
                logging_dir=str(fold_dir / "tb"),
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.train_batch,
                per_device_eval_batch_size=args.eval_batch,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                warmup_steps=args.warmup_steps,
                warmup_ratio=args.warmup_ratio,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=1,
                load_best_model_at_end=True,
                metric_for_best_model=f"eval_{args.metric_for_best}",
                greater_is_better=True,
                report_to=["tensorboard"],
                logging_steps=args.logging_steps,
                seed=args.seed,
                data_seed=args.seed,
                fp16=args.fp16,
                dataloader_num_workers=1,
            )

            trainer_cv = Trainer(
                model=model_fold,
                args=ta_cv,
                train_dataset=ds_tr,
                eval_dataset=ds_va,
                compute_metrics=hf_compute_metrics,
            )

            print(f"[CV] Fold {fold_idx}/{args.kfold}: train={len(ds_tr)}, val={len(ds_va)}")
            trainer_cv.train()

            # Evaluate the best checkpoint on this fold's validation set
            fold_metrics = trainer_cv.evaluate(eval_dataset=ds_va)
            cv_fold_metrics.append(fold_metrics)
            cv_fold_sizes.append(len(ds_va))

            # Persist per-fold metrics for provenance
            save_json(
                {"fold_index": fold_idx, "val_metrics": fold_metrics, "val_size": len(ds_va)},
                fold_dir / "cv_val_metrics.json"
            )

        # Aggregate CV metrics (mean/std for numeric items)
        numeric_keys = sorted(
            k for k in set().union(*[m.keys() for m in cv_fold_metrics])
            if isinstance(cv_fold_metrics[0].get(k, None), (int, float))
        )
        cv_mean = {k: float(np.mean([m.get(k, np.nan) for m in cv_fold_metrics])) for k in numeric_keys}
        cv_std  = {k: float(np.std([m.get(k, np.nan) for m in cv_fold_metrics], ddof=1)) for k in numeric_keys}

        # Save CV summary at the run root
        run_dir = build_run_dir(dataset_name, args.seed, model_dir_name, args.metric)

        save_json(
            {
                "kfold": args.kfold,
                "val_sizes": cv_fold_sizes,
                "per_fold_metrics": cv_fold_metrics,
                "mean": cv_mean,
                "std": cv_std,
            },
            run_dir / "cv_metrics.json",
        )

        print("[CV] Summary (mean):", {k: round(v, 4) for k, v in cv_mean.items()
            if any(s in k for s in ["f1", "accuracy", "balanced_accuracy", "precision", "recall"])})
        print("[CV] Summary (std):",  {k: round(v, 4) for k, v in cv_std.items()
            if any(s in k for s in ["f1", "accuracy", "balanced_accuracy", "precision", "recall"])})

        # --- Final retrain on the entire train pool (x_tmp/y_tmp), then one clean test ---
        tokenizer_full, model_full = select_tokenizer_and_pretrained_model(repo_root, args.model, num_labels, use_normalised_tokens=args.use_normalised_tokens)
        # normalise both the full-train pool and the test set using the same flag
        x_full_norm      = [normalise_text(t, enabled=use_norm) for t in x_tmp]
        x_test_full_norm = [normalise_text(t, enabled=use_norm) for t in x_test]

        # Tokenise the normalised text
        enc_full = tokenizer_full(x_full_norm, truncation=True, padding="max_length", max_length=512)
        ds_full  = CVSSDataset(enc_full, y_tmp)

        ta_full = TrainingArguments(
            output_dir=str(run_dir),
            logging_dir=str(run_dir / "tb"),
            num_train_epochs=args.epochs,                 # keep as requested; CV already validated viability
            per_device_train_batch_size=args.train_batch,
            per_device_eval_batch_size=args.eval_batch,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            warmup_ratio=args.warmup_ratio,
            evaluation_strategy="no",                     # no peeking during final fit
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=False,
            report_to=["tensorboard"],
            logging_steps=args.logging_steps,
            seed=args.seed,
            data_seed=args.seed,
            fp16=args.fp16,
            dataloader_num_workers=1,
        )

        trainer_full = Trainer(
            model=model_full,
            args=ta_full,
            train_dataset=ds_full,
            eval_dataset=None,
            compute_metrics=None,
        )

        print("[CV] Final retrain on full train pool (no validation).")
        trainer_full.train()
        trainer_full.save_model()
        tokenizer_full.save_pretrained(run_dir)

        # Build test set with the full tokenizer to ensure vocab alignment
        test_enc_full = tokenizer_full(x_test_full_norm, truncation=True, padding="max_length", max_length=512)
        test_ds_full  = CVSSDataset(test_enc_full, y_test)

        _ = trainer_full.evaluate(eval_dataset=test_ds_full)  # one-off test
        y_true, logits, probs, y_pred = trainer_predict(trainer_full, test_ds_full)
        test_metrics = compute_all_metrics(y_true, logits)
        save_json({"test_metrics": test_metrics}, run_dir / "metrics.json")

        print("Test metrics:", {k: round(v, 4) for k, v in test_metrics.items()})
        print("Table Format: ", {k: round(v, 4) for k, v in test_metrics.items()
            if k in ["accuracy", "balanced_accuracy", "f1_weighted", "f1_macro", "f1_micro"]})

        # ---- collect extra provenance from folds for config.json ----
        fold_ckpts = []
        for fold_idx in range(1, args.kfold + 1):
            fold_dir = build_run_dir(dataset_name, args.seed, model_dir_name, args.metric) / f"cv_fold_{fold_idx}"
            try:
                state_path = fold_dir / "trainer_state.json"
                if state_path.exists():
                    with open(state_path, "r") as f:
                        st = json.load(f)
                    fold_ckpts.append(st.get("best_model_checkpoint"))
                else:
                    fold_ckpts.append(None)
            except Exception:
                fold_ckpts.append(None)

        # Final-run splits (no fixed val in final retrain)
        n_total = len(all_labels)
        n_test  = len(y_test)
        n_train = len(y_tmp)
        n_val   = 0

        splits = {
            "counts": {"train": n_train, "val": n_val, "test": n_test, "total": n_total},
            "fractions": {
                "train": round(n_train / n_total, 6),
                "val":   round(n_val   / n_total, 6),
                "test":  round(n_test  / n_total, 6),
            },
            "requested": {"val_size": args.val_size, "test_size": args.test_size, "kfold": args.kfold},
        }
        

        cfg = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset": dataset_name,
            "data_file": args.data_file,
            "metric": args.metric,
            "classes": classes,
            "num_labels": num_labels,
            "label_position": label_pos,
            "model": args.model,                          
            "artifact_model_dir_name": model_dir_name,
            "seed": args.seed,
            "splits": splits,
            "train_args": {
                "epochs": args.epochs,
                "train_batch": args.train_batch,
                "eval_batch": args.eval_batch,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "warmup_ratio": args.warmup_ratio,
                "metric_for_best": args.metric_for_best,
                "fp16": args.fp16,
            },
            # final retrain doesn’t track "best", so persist the final dir
            "best_model_dir": str(run_dir),
            "cv": {
                "k": args.kfold,
                "per_fold_best_ckpt": fold_ckpts,
                "mean": cv_mean,
                "std": cv_std,
                "val_sizes": cv_fold_sizes,
            },
            "use_normalised_tokens": args.use_normalised_tokens,
            "special_tokens": ["<IP>", "<IPPORT>", "<VER>", "<CVE>", "<PATH>"] if args.use_normalised_tokens else [],
            "tokenizer_added_vocab": list(getattr(tokenizer, "get_added_vocab", lambda: {})().keys()) if not use_kfold else
                                    list(getattr(tokenizer_full, "get_added_vocab", lambda: {})().keys()),
        }
        save_json(cfg, run_dir / "config.json")

        print("Saved to:", run_dir)
        return
        
    else:

        # --- evaluation wiring
        if has_val:
            eval_ds = val_ds
            evaluation_strategy = "epoch"
            load_best = True
            metric_for_best = f"eval_{args.metric_for_best}"
            compute_metrics_fn = hf_compute_metrics
        elif args.eval_on_test:
            # Sanity-check mode: evaluate each epoch on the test set
            eval_ds = test_ds
            evaluation_strategy = "epoch"
            load_best = False              # mimic Costa (no best-model selection on test)
            metric_for_best = None
            compute_metrics_fn = hf_compute_metrics
        else:
            eval_ds = None
            evaluation_strategy = "no"
            load_best = False
            metric_for_best = None
            compute_metrics_fn = None

        ta = TrainingArguments(
            output_dir=str(run_dir),
            logging_dir=str(run_dir / "tb"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.train_batch,
            per_device_eval_batch_size=args.eval_batch,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            warmup_ratio=args.warmup_ratio,
            evaluation_strategy=evaluation_strategy,   # "no" or "epoch"
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=load_best,
            metric_for_best_model=metric_for_best,
            greater_is_better=True,  # <- force a bool; ignored when load_best_model_at_end=False
            report_to=["tensorboard"],
            logging_steps=args.logging_steps,
            seed=args.seed,
            data_seed=args.seed,
            fp16=args.fp16,
            dataloader_num_workers=1,
        )

        trainer = Trainer(
            model=model,
            args=ta,
            train_dataset=train_ds,
            eval_dataset=eval_ds,                # None, val_ds, or test_ds (sanity mode)
            compute_metrics=compute_metrics_fn,  # None unless we’re evaluating
        )

        # train + persist model
        trainer.train()
        trainer.save_model()  # best checkpoint if has_val, else last epoch in run_dir
        tokenizer.save_pretrained(run_dir)

        # final test
        _ = trainer.evaluate(eval_dataset=test_ds)  # one-off test evaluation
        y_true, logits, probs, y_pred = trainer_predict(trainer, test_ds)
        test_metrics = compute_all_metrics(y_true, logits)

        n_total = len(all_labels)
        n_train, n_test = len(y_train), len(y_test)
        n_val = len(y_val) if has_val else 0

        splits = {
            "counts": {"train": n_train, "val": n_val, "test": n_test, "total": n_total},
            "fractions": {
                "train": round(n_train / n_total, 6),
                "val":   round(n_val   / n_total, 6),
                "test":  round(n_test  / n_total, 6),
            },
            # keep the requested targets too, for provenance
            "requested": {"val_size": args.val_size, "test_size": args.test_size},
        }

        # persist run config + metrics
        cfg = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset": dataset_name,
            "data_file": args.data_file,
            "metric": args.metric,
            "classes": classes,
            "num_labels": num_labels,
            "label_position": label_pos,
            "model": args.model, 
            "artifact_model_dir_name": model_dir_name, 
            "seed": args.seed,
            "splits": splits,
            "train_args": {
                "epochs": args.epochs,
                "train_batch": args.train_batch,
                "eval_batch": args.eval_batch,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "warmup_ratio": args.warmup_ratio,
                "metric_for_best": args.metric_for_best,
                "fp16": args.fp16,
            },
            "best_model_dir": str(trainer.state.best_model_checkpoint or run_dir),
            "use_normalised_tokens": args.use_normalised_tokens,
            "special_tokens": ["<IP>", "<IPPORT>", "<VER>", "<CVE>", "<PATH>"] if args.use_normalised_tokens else [],
            "tokenizer_added_vocab": list(getattr(tokenizer, "get_added_vocab", lambda: {})().keys()) if not use_kfold else
                                    list(getattr(tokenizer_full, "get_added_vocab", lambda: {})().keys()),
        }
        save_json(cfg, run_dir / "config.json")
        save_json({"test_metrics": test_metrics}, run_dir / "metrics.json")

        print("Test metrics:", {k: round(v, 4) for k, v in test_metrics.items()})
        print("Table Format: ", {k: round(v, 4) for k, v in test_metrics.items()
            if k in ["accuracy", "balanced_accuracy", "f1_weighted", "f1_macro", "f1_micro"]})
        print("Saved to:", run_dir)

if __name__ == "__main__":
    main()

# utils/lrp_distilbert_hf.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification


class LRPDistilBertForSequenceClassification(DistilBertForSequenceClassification):
    """
    Real HF DistilBERT classifier (same weights, same keys),
    with an LRP-style relprop() that returns token-level relevance.

    Implementation: Gradient × Input at the embedding level.
    """

    def relprop(
        self,
        seed: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        clamp_positive: bool = True,
        normalise: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            seed: shape [B, C] one-hot or weighted seed over classes.
            attention_mask: [B, T]
            input_ids: [B, T] (required)
        Returns:
            relevance: [B, T] token relevance
        """
        if input_ids is None:
            raise ValueError("relprop() requires input_ids (pass the same ids you used for forward).")

        self.eval()

        # Make sure grads are enabled even if caller used inference_mode
        with torch.enable_grad():
            # Build embeddings with grad
            emb_layer = self.get_input_embeddings()
            embeds = emb_layer(input_ids)
            embeds = embeds.detach().requires_grad_(True)

            # Forward using inputs_embeds so we can backprop to embeds
            outputs = self.distilbert(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state  # [B, T, H]

            # HF head: take CLS-like token (position 0) then pre_classifier, dropout, classifier
            pooled = hidden[:, 0]  # [B, H]
            pooled = self.pre_classifier(pooled)
            pooled = nn.ReLU()(pooled)
            pooled = self.dropout(pooled)
            logits = self.classifier(pooled)  # [B, C]

            # Seeded scalar objective: sum(logits * seed)
            if seed.dim() != 2 or seed.shape[0] != logits.shape[0] or seed.shape[1] != logits.shape[1]:
                raise ValueError(f"seed must be [B, C]. Got seed={tuple(seed.shape)} logits={tuple(logits.shape)}")

            objective = (logits * seed).sum()

            # Clear old grads
            self.zero_grad(set_to_none=True)
            if embeds.grad is not None:
                embeds.grad.zero_()

            objective.backward(retain_graph=False)

            grad = embeds.grad  # [B, T, H]
            if grad is None:
                raise RuntimeError("No gradient produced in relprop(); check that inputs are on the same device.")

            # Gradient × Input relevance
            R = (grad * embeds).sum(dim=-1)  # [B, T]

            if attention_mask is not None:
                R = R * attention_mask.to(dtype=R.dtype)

            if clamp_positive:
                R = torch.clamp(R, min=0.0)

            if normalise:
                # Normalise per example to [0,1] max for stability in rendering
                denom = R.max(dim=1, keepdim=True).values.clamp(min=1e-12)
                R = R / denom

            return R

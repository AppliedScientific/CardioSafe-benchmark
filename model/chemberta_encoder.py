"""ChemBERTa sentence-embedding adapter.

Loads `DeepChem/ChemBERTa-77M-MTR` (Chithrananda et al., 2020) and returns
mean-pooled 384-dim sentence embeddings of SMILES. The transformer is
frozen and used purely as a featurizer, exactly as described in the
Methods section of Jovanovic et al. 2026.

Mean pooling is attention-mask-weighted: padding positions are excluded
from the mean. SMILES inputs are tokenised with the model's tokenizer
(max_length=512, truncation enabled, dynamic padding per batch).
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class ChemBERTaEncoder:
    """SMILES -> (N, 384) float32 embeddings using ChemBERTa-77M-MTR."""

    MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
    MAX_LENGTH = 512

    def __init__(self) -> None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self._model = AutoModel.from_pretrained(self.MODEL_NAME).to(device).eval()
        self._device = device

    def encode(self, smiles: list[str]) -> np.ndarray:
        """Encode a list of SMILES into (N, 384) float32 embeddings."""
        tokens = self._tokenizer(
            smiles,
            padding=True,
            truncation=True,
            max_length=self.MAX_LENGTH,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**tokens)

        hidden = outputs.last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        return pooled.cpu().numpy().astype(np.float32)

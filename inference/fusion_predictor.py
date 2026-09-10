"""Experimental phase-4 inference from raw, row-aligned cached feature vectors.

This does not change the published v1.0/v1.1 predictor or download any weights.
Inputs: Morgan 2048, AtomPair 2048, TopTorsion 2048, RAW descriptors 20,
ChemBERTa 384. The checkpoint owns descriptor preprocessing and target units.
For alternative encoder caches, supply the readout identified by
chemberta_feature_contract; a raw matrix cannot prove its representation identity.
"""
from pathlib import Path
import numpy as np
import torch
from model.cardiosafe import ALL_HEADS, REGRESSION_HEADS, CLASSIFICATION_HEADS, load_checkpoint
from model.fusion_factory import build_model
from train.fusion_preprocessing import transform_descriptors


class FusionPredictor:
    def __init__(self, checkpoint: str | Path, device="cpu"):
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if not all(c in ck.get('reg_scalers', {}) for c in (0, 3, 5)):
            raise ValueError('Checkpoint must contain all three regression scalers')
        self.means = np.asarray(ck['descriptor_scaler_means'], dtype=np.float32)
        self.stds = np.asarray(ck['descriptor_scaler_stds'], dtype=np.float32)
        if self.means.shape != (20,) or self.stds.shape != (20,) or not np.all(self.stds > 0):
            raise ValueError('Expected valid training-fold scalers for 20 descriptors')
        self.transform = ck.get('descriptor_transform', 'none')
        self.chemberta_feature_contract = ck.get('chemberta_cache')
        self.token_feature_contract = ck.get('token_cache')
        self.encoder_dim = ck.get('encoder_dim', 384)
        if self.chemberta_feature_contract and self.chemberta_feature_contract.get('width', 384) != self.encoder_dim:
            raise ValueError('Checkpoint encoder width disagrees with its cache contract')
        mode = ck['attention']
        self.model = build_model(mode=mode, slots=ck.get('slots', 4), chemberta=ck.get('chemberta', True), packed_tokens=ck.get('packed_tokens', False), encoder_dim=ck.get('encoder_dim', 384))
        load_checkpoint(self.model, ck)
        self.device = torch.device(device)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_features(self, raw_features: np.ndarray, batch_size=2048, class_logits=False,
                         token_states=None, token_mask=None):
        """Return pChEMBL and probabilities (or optional logits); input is unchanged."""
        x = np.asarray(raw_features)
        if x.ndim != 2 or x.shape[1] != 6164 + self.encoder_dim:
            raise ValueError(f'Expected (molecules, {6164 + self.encoder_dim}) raw feature array')
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        molecular = hasattr(self.model, 'token_pool')
        if molecular:
            if token_states is None or token_mask is None:
                raise ValueError('This checkpoint requires aligned token states and a boolean mask')
            token_states, token_mask = np.asarray(token_states), np.asarray(token_mask)
            if (token_states.ndim != 3 or token_states.shape[0] != len(x) or token_states.shape[2] != 384
                    or token_mask.shape != token_states.shape[:2] or token_mask.dtype != bool):
                raise ValueError('Token states/mask do not match feature rows')
            batch_size = min(batch_size,128)
        elif token_states is not None or token_mask is not None:
            raise ValueError('This checkpoint does not consume token states')
        chunks = {h: [] for h in ALL_HEADS}
        for start in range(0, len(x), batch_size):
            batch = x[start:start+batch_size].astype(np.float32, copy=True)
            z = (batch[:, 6144:6164] - self.means) / self.stds
            batch[:, 6144:6164] = transform_descriptors(z, self.transform)
            tensor = torch.from_numpy(batch).to(self.device)
            extra = ()
            if molecular:
                extra = (torch.as_tensor(token_states[start:start+batch_size],dtype=torch.float32,device=self.device),
                         torch.as_tensor(token_mask[start:start+batch_size],device=self.device))
            if class_logits:
                out = self.model(tensor,*extra)
                for i,h in enumerate(REGRESSION_HEADS):
                    out[h] = out[h]*self.model.reg_std[i]+self.model.reg_mean[i]
            else:
                out = self.model.predict(tensor,*extra)
            for h, value in out.items():
                chunks[h].append(value.cpu().numpy())
        return {h: np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
                for h, parts in chunks.items()}


class FusionEnsemble:
    """Match experimental scoring: mean pChEMBL, sigmoid of mean class logits."""
    def __init__(self, checkpoints, device='cpu'):
        self.predictors = [FusionPredictor(path, device=device) for path in checkpoints]
        if not self.predictors:
            raise ValueError('At least one checkpoint is required')
        contract = self.predictors[0].chemberta_feature_contract
        if any(p.chemberta_feature_contract != contract for p in self.predictors):
            raise ValueError('An ensemble feature matrix must use one encoder readout contract')
        token_contract = self.predictors[0].token_feature_contract
        if any(p.token_feature_contract != token_contract for p in self.predictors):
            raise ValueError('An ensemble must use one token representation contract')

    def predict_features(self, raw_features: np.ndarray, batch_size=2048, class_logits=False,
                         token_states=None, token_mask=None):
        predictions = [p.predict_features(raw_features, batch_size=batch_size, class_logits=True,
                                          token_states=token_states, token_mask=token_mask)
                       for p in self.predictors]
        out = {h:np.mean([p[h] for p in predictions],axis=0) for h in ALL_HEADS}
        if not class_logits:
            for h in CLASSIFICATION_HEADS:
                out[h] = torch.from_numpy(out[h]).sigmoid().numpy()
        return out

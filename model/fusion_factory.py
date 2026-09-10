"""One architecture factory for experimental training, scoring and inference."""
import hashlib
from model.cardiosafe import CardioSafe
from model.token_fusion import TokenFusion
from model.latent_fusion import LatentFusion, CachedConcat
from model.bit_fusion import BitFusion
from model.shared_fusion import SharedFusion
from model.molecular_token_fusion import MolecularTokenFusion

MOLECULAR_MODES = tuple("molecular_"+m for m in ("conditioned", "fixed_query", "mean"))

FUSION_MODES = TokenFusion.MODES + LatentFusion.MODES + BitFusion.MODES + SharedFusion.MODES
BIT_MODES = BitFusion.MODES
ROUTED_MODES = tuple(m for m in FUSION_MODES if m not in ('token_concat', 'token_mlp', 'shared_concat', *BIT_MODES))
SELF_MODES = ('token_self', 'token_self_static', *LatentFusion.MODES)


def parameter_digest(model):
    """Initial SD can be identically zero; verify actual initial parameters too."""
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        digest.update(f'{name}:{tuple(value.shape)}:{value.dtype}\0'.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_model(mode='none', slots=4, chemberta=True, chem_dim=6164, dropout=.3, packed_tokens=False, encoder_dim=384):
    if not isinstance(encoder_dim, int) or encoder_dim < 1:
        raise ValueError('Encoder width must be a positive integer')
    if encoder_dim != 384 and mode != 'none':
        raise ValueError('Alternative encoder widths currently require concat mode')
    if packed_tokens and mode not in MOLECULAR_MODES:
        raise ValueError('Packed tokens require a molecular-token mode')
    if mode in (*FUSION_MODES, *MOLECULAR_MODES) and (not chemberta or chem_dim != 6164):
        raise ValueError('Experimental fusion expects three 2048-bit FPs, 20 descriptors, and ChemBERTa')
    if mode in MOLECULAR_MODES:
        return MolecularTokenFusion(mode=mode.removeprefix("molecular_"), dropout=dropout, packed_tokens=packed_tokens)
    if mode in TokenFusion.MODES:
        return TokenFusion(mode=mode, slots=slots, dropout=dropout)
    if mode in LatentFusion.MODES:
        return LatentFusion(mode=mode, heads=slots, dropout=dropout)
    if mode in BIT_MODES:
        return BitFusion(mode=mode, dropout=dropout)
    if mode in SharedFusion.MODES:
        return SharedFusion(mode=mode, heads=slots, dropout=dropout)
    cls = CachedConcat if mode == 'none' else CardioSafe
    return cls(attention=mode, chemberta=chemberta, chem_dim=chem_dim, dropout=dropout, chemberta_dim=encoder_dim)

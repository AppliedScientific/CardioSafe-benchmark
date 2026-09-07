"""Guards against silent pKa degradation.

The 7 MolGpKa-derived descriptors fail open: when site detection breaks, every
compound gets sentinel pKa and predictions shift with no error raised. These
tests pin both halves of that behaviour.
"""

from __future__ import annotations

import os
import warnings

import pytest

from inference.featurize import (
    PKA_ACIDIC_SENTINEL,
    PKA_BASIC_SENTINEL,
    _ionization_aids,
    _predict_pka,
)

TERFENADINE = "CC(C)(C)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4"
FEXOFENADINE = "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4"

needs_molgpka = pytest.mark.skipif(
    not os.environ.get("MOLGPKA_SRC"),
    reason="MOLGPKA_SRC not set; see inference/README.md",
)


@needs_molgpka
def test_basic_site_is_detected_from_any_cwd():
    """Terfenadine's piperidine nitrogen must be found, not sentinelled.

    Regression test for the CWD-relative smarts_pattern.tsv lookup: this
    returned (14.0, 0.0) for every molecule when run outside $MOLGPKA_SRC.
    """
    acidic, basic = _predict_pka(TERFENADINE)
    assert (acidic, basic) != (PKA_ACIDIC_SENTINEL, PKA_BASIC_SENTINEL)
    assert basic == pytest.approx(9.45, abs=0.5)


@needs_molgpka
def test_carboxylic_acid_lowers_acidic_pka():
    """Fexofenadine differs from terfenadine by a COOH, which must show up."""
    terf_acidic, _ = _predict_pka(TERFENADINE)
    fex_acidic, _ = _predict_pka(FEXOFENADINE)
    assert fex_acidic == pytest.approx(4.42, abs=0.5)
    assert fex_acidic < terf_acidic - 5.0


def test_broken_site_detection_warns_instead_of_failing_silently():
    """An environment fault must be visible, and must warn only once."""

    def broken(mol, acid_or_base=None):
        raise FileNotFoundError("utils/smarts_pattern.tsv")

    import inference.featurize as featurize

    featurize._pka_setup_warned = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _ionization_aids(broken, None, "acid") == []
        assert _ionization_aids(broken, None, "base") == []

    assert len(caught) == 1, "should warn exactly once, not per molecule"
    assert issubclass(caught[0].category, RuntimeWarning)
    assert "sentinel" in str(caught[0].message).lower()

# Note S2 — Activity-cliff curation provenance

## Overview

The Stage 2 fine-tune of CardioSafe trains on a small set of
**cardiac activity-cliff compounds** drawn from the published
literature, organised into therapeutic-class `pair_id` groups in
which clinical hERG-safety divergence is established between a
`blocker` member and a structurally-related `safer` analogue.
This note documents the two-stage curation pipeline that produces
the per-split cliff training set, the 25 literature sources from
which the raw set is drawn, and the per-`pair_id` composition that
determines which pairs contribute (blocker, safer) ranking terms
to the cliff loss.

## Curation pipeline

**Stage A — manual literature curation.** Hand-curated from 25
published cardiac-cliff / hERG-screening papers (Bowes et al. 2012,
Kongsamut et al. 2002, Kang et al. 2001, Kramer et al. 2013,
Katchman et al. 2002, and 20 smaller-source contributions).
Inclusion criterion: a compound with a quantitatively reported hERG
IC50 (or block fraction at 1 µM and 10 µM) that can be grouped with
at least one structurally-related analogue from the same therapeutic
class whose *clinical* cardiac-safety profile diverges (e.g. one is
withdrawn or carries a black-box QT warning while the other is
widely-used and safe). The qualifier is *clinical-safety divergence
within a therapeutic-class group*, **not** a numerical similarity
or |Δ-pIC50| threshold. The output is `cliff_pairs_all.csv` — 53
compounds organised into 30 `pair_id` groups spanning 12 therapeutic
categories (antiarrhythmic, antihistamine, atypical antipsychotic,
beta-blocker, benzofuran, butyrophenone, calcium-channel blocker,
fluoroquinolone, gastroprokinetic, opioid, phenothiazine, and
tricyclic antidepressant).

**Stage B — automated Tanimoto leak-prevention filter.** For the
active split, the filter computes Morgan-r=2 2,048-bit Tanimoto
from each cliff compound to **every** validation + test row and
drops any cliff compound whose maximum Tanimoto to the val/test
holdout exceeds the split cutoff (0.70 for tan70_v2, 0.60 for
tan60_v2). This is the only mechanism preventing cliff-training
information from leaking into the held-out evaluation set.

## Per-split survivors

| split | cliff compounds (post-filter) | `pair_id` groups | both-role pairs (ranking-eligible) |
| :-- | --: | --: | --: |
| tan70 | 48 | 29 | 12 |
| tan60 | 51 | 29 | 13 |

On **tan70**, five compounds drop because they sit in the validation
fold (Tanimoto = 1.0 to a val row): Terfenadine, Fexofenadine,
Ranolazine, Desipramine, and Amitriptyline.

On **tan60** (stricter 0.60 cutoff), only Terfenadine and
Fexofenadine drop — they are the only two compounds force-moved to
the validation fold by `build_tan_splits_v2.py`. Ranolazine,
Desipramine, and Amitriptyline sit in the *training* fold on tan60
under the relaxed cutoff and therefore remain in the cliff set.

The 13th ranking-eligible pair on tan60 (versus 12 on tan70) is
**TC2** (Amitriptyline / Doxepin) — Amitriptyline is in val on
tan70 but in train on tan60.

## Source bibliography (25 entries)

Citation keys are first-author surname + four-digit year, matching
the `source` column in `cliff_pairs_all.csv`. **Full citations
below should be completed by the author at proof stage** — the
entries here capture first-author / year / topic only.

| citation key | n compounds | citation |
| :-- | --: | :-- |
| `Bowes2012` | 10 | Bowes J. et al. (2012). Reducing safety-related drug attrition: the use of in vitro pharmacological profiling. Nature Reviews Drug Discovery 11(12), 909–922. |
| `Kang2001` | 6 | Kang J. et al. (2001). Interactions of a series of fluoroquinolone antibacterials with the human cardiac K+ channel hERG. |
| `Kongsamut2002` | 6 | Kongsamut S. et al. (2002). A comparison of the receptor binding and hERG channel affinities for a series of antipsychotic drugs. |
| `Kramer2013` | 5 | Kramer J. et al. (2013). MICE models: superior to the HERG model in predicting Torsade de Pointes. Scientific Reports 3, 2100. |
| `Katchman2002` | 4 | Katchman A. N. et al. (2002). Influence of opioid agonists on cardiac human ether-à-go-go-related gene K+ currents. |
| `Rampe1997` | 2 | Rampe D. et al. (1997). Effects of the antimalarial / antipsychotic / 5-HT4 agonist agents on hERG block (cisapride / chlorpromazine class). |
| `Zhang1999` | 2 | Zhang S. et al. (1999). Mechanism of block of the hERG potassium channel by dihydropyridines (verapamil-class CCB study). |
| `Carlsson2005` | 1 | Carlsson L. et al. (2005). Proarrhythmic profile of mosapride. |
| `ChavesGonzalez2013` | 1 | Chaves-González et al. (2013). Paliperidone hERG screen. |
| `Drolet1999` | 1 | Drolet B. et al. (1999). Droperidol lengthens cardiac repolarisation. |
| `Guo2005` | 1 | Guo D. et al. (2005). Prochlorperazine and hERG block. |
| `Kang2000` | 1 | Kang J. et al. (2000). Pimozide hERG inhibition. |
| `Katchman2006` | 1 | Katchman A. N. et al. (2006). Buprenorphine and methadone effects on cardiac repolarisation. |
| `Kiesecker2004` | 1 | Kiesecker C. et al. (2004). Amiodarone interaction with hERG. |
| `Mendzelevski2012` | 1 | Mendzelevski B. et al. (2012). Prucalopride and ECG safety. |
| `Numaguchi2000` | 1 | Numaguchi H. et al. (2000). hERG block by class III antiarrhythmics (sotalol). |
| `Paul2002` | 1 | Paul A. A. et al. (2002). Flecainide and hERG. |
| `Rajamani2008` | 1 | Rajamani S. et al. (2008). Ranolazine and multiple cardiac ion channels. |
| `Rampe1998` | 1 | Rampe D. et al. (1998). A mechanism for the proarrhythmic effects of sertindole. |
| `Scherer2002` | 1 | Scherer C. R. et al. (2002). Fexofenadine vs terfenadine on hERG. |
| `Suessbrich1997` | 1 | Suessbrich H. et al. (1997). Haloperidol and hERG. |
| `Taglialatela1998` | 1 | Taglialatela M. et al. (1998). Loratadine on hERG. |
| `Taglialatela2000` | 1 | Taglialatela M. et al. (2000). Second-generation antihistamines on hERG (cetirizine). |
| `Teschemacher1999` | 1 | Teschemacher A. G. et al. (1999). Antidepressants and hERG (imipramine). |
| `Yang1995` | 1 | Yang T. et al. (1995). Ibutilide and the human cardiac IKr current. |

Total: 53 compounds drawn from 25 distinct sources; 5 sources
(Bowes 2012, Kang 2001, Kongsamut 2002, Kramer 2013, Katchman 2002)
contribute ≥ 4 compounds each; the remaining 20 sources contribute
1–2 compounds each.

## `pair_id` composition (30 groups)

| pair_id | therapeutic class | blockers | safer analogues | sources | tan70 RP? | tan60 RP? |
| :-- | :-- | :-- | :-- | :-- | :--: | :--: |
| AA1 | Antiarrhythmic | — | Mexiletine; Ranolazine | Kramer2013; Rajamani2008 | — | — |
| AA2 | Antiarrhythmic | Flecainide | Lidocaine | Kramer2013; Paul2002 | ✓ | ✓ |
| AA3 | Antiarrhythmic | Sotalol; Dofetilide | — | Kramer2013; Numaguchi2000 | — | — |
| AA4 | Antiarrhythmic | Ibutilide | — | Yang1995 | — | — |
| AH1 | Antihistamine | Terfenadine | Fexofenadine | Bowes2012; Scherer2002 | — | — |
| AH2 | Antihistamine | Astemizole | Desloratadine | Bowes2012; Kongsamut2002 | ✓ | ✓ |
| AH3 | Antihistamine | — | Loratadine; Cetirizine | Taglialatela1998; Taglialatela2000 | — | — |
| AP1 | Atypical antipsychotic | Ziprasidone | Aripiprazole | Kongsamut2002 | ✓ | ✓ |
| AP2 | Atypical antipsychotic | — | Olanzapine | Kongsamut2002 | — | — |
| AP3 | Atypical antipsychotic | Risperidone | Paliperidone | ChavesGonzalez2013; Kongsamut2002 | ✓ | ✓ |
| AP4 | Atypical antipsychotic | Sertindole | Quetiapine | Kongsamut2002; Rampe1998 | ✓ | ✓ |
| BB1 | Beta-blocker | — | Atenolol; Metoprolol | Bowes2012 | — | — |
| BB2 | Beta-blocker | — | Propranolol | Bowes2012 | — | — |
| BF1 | Benzofuran | — | Amiodarone | Kiesecker2004 | — | — |
| BP1 | Butyrophenone | Haloperidol; Droperidol | — | Drolet1999; Suessbrich1997 | — | — |
| BP2 | Butyrophenone | Pimozide | — | Kang2000 | — | — |
| CCB1 | Calcium-channel blocker | — | Verapamil; Diltiazem | Kramer2013 | — | — |
| CCB2 | Calcium-channel blocker | — | Nifedipine; Amlodipine | Zhang1999 | — | — |
| FQ1 | Fluoroquinolone | Sparfloxacin | Ciprofloxacin | Kang2001 | ✓ | ✓ |
| FQ2 | Fluoroquinolone | Moxifloxacin | Levofloxacin | Kang2001 | ✓ | ✓ |
| FQ3 | Fluoroquinolone | Grepafloxacin | Norfloxacin | Kang2001 | ✓ | ✓ |
| GP1 | Gastroprokinetic (5-HT4) | Cisapride | Mosapride | Carlsson2005; Rampe1997 | ✓ | ✓ |
| GP2 | Gastroprokinetic (5-HT4) | — | Prucalopride | Mendzelevski2012 | — | — |
| OP1 | Opioid | Methadone | Buprenorphine | Katchman2002; Katchman2006 | ✓ | ✓ |
| OP2 | Opioid | — | Morphine; Oxycodone | Katchman2002 | — | — |
| OP3 | Opioid | — | Fentanyl | Katchman2002 | — | — |
| PH1 | Phenothiazine | Thioridazine | Prochlorperazine | Bowes2012; Guo2005 | ✓ | ✓ |
| PH2 | Phenothiazine | Chlorpromazine | Trifluoperazine | Bowes2012; Rampe1997 | ✓ | ✓ |
| TC1 | Tricyclic antidepressant | Imipramine; Desipramine | — | Bowes2012; Teschemacher1999 | — | — |
| TC2 | Tricyclic antidepressant | Amitriptyline | Doxepin | Bowes2012 | — | ✓ |

*RP = ranking pair*. ✓ means the `pair_id` has at least one
`blocker` member and at least one `safer` member surviving the
Tanimoto leak-prevention filter, and therefore contributes a
(blocker, safer) hERG-logit ranking term to the Stage 2 loss on
that split.

## Companion CSVs

- `note_s2_pair_id_composition.csv` — the table above, machine-readable.
- `note_s2_per_split_manifest.csv` — one row per compound × split, with
  the per-split include / exclude status and the literature IC50,
  herg_1um_block, and herg_10um_block fields verbatim.
- `note_s2_source_bibliography.csv` — the 25 source citation keys with
  per-source compound counts and citation stub for the author to expand.

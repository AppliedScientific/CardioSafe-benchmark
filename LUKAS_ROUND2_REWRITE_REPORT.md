# CardioSafe round 2 — rewrite handoff for Lukas

Prepared 12 November 2026 against `codex/lukas-final-architecture` at `398731342ded43488c2359d840cf22bad8105918` and the supplied reviewer-comments/manuscript copies.

**Lukas: use this document to revise the manuscript and assemble the response letter.** It covers all 11 reviewer comments, supplies replacement prose, and adds verified evidence from the existing final models. It does not mean the Google manuscript has already been edited or that the revision is ready to submit.

The first reviewer is not numbered in the supplied comments document; “Reviewer A” is a working label. Reviewer 4 is explicitly identified. Response-letter paragraphs below are drafts for use **after** the associated changes and supplements are integrated. Add final manuscript page/line references at that stage.

## What is ready, and what remains unresolved

- [x] Inspected the final branch, split reconstruction, feature extraction and saved final-run evidence.
- [x] Matched the saved ChemBERTa weight hash to the published checkpoint.
- [x] Recomputed both predicted-property scalers from TRAIN; both match exactly.
- [x] Recomputed descriptor/target scalers and class weights for all ten final checkpoints; all match TRAIN-only calculations within the recorded numerical tolerance.
- [x] Evaluated all 31 nonempty subsets of five seeds on each split; reproduced the saved five-seed ensembles.
- [x] Added exploratory error summaries across six physicochemical descriptors, with bins defined from TRAIN.
- [x] Prepared the full v3 compound/label/split appendix and verified identity alignment and an exact parsed-data round trip.
- [ ] Integrate the replacement prose, tables, citations and supplements into the manuscript.
- [ ] Resolve or explicitly retain the randomized-control limitation. Its cause is **not established**.
- [ ] Do not claim that IKs improves other tasks: a matched with/without-IKs experiment has **not** been performed.
- [ ] Finalize page/line references and journal-specific supplementary numbering.

No models were retrained for this handoff. All new numerical analyses use the existing five final v3 fits per split. The companion data are a join/export of the frozen labels, not a label revision.

## Priority corrections before the point-by-point reply

1. **Correct the architecture.** The manuscript currently assigns the three projections to the wrong inputs. The actual branches are fingerprints → 256, descriptors → 64, and combined CLS384 + MTR199 → 128.
2. **Restore the splitting paragraph.** “Primary split (tan70)” currently repeats training-procedure text.
3. **Remove the established-cause explanation for randomized-label residual ranking.** The handoff and saved findings explicitly leave its cause unresolved.
4. **Use the supplied v3 appendix in the availability statement.** The earlier code branch supplied v1 labels, with v3 in separate archives.
5. **Correct the training environment.** All ten final-run provenance records identify Linux, Python 3.12.3 and PyTorch 2.10.0+cpu. The manuscript's blanket M4 Pro/MPS claim does not describe these final fits.

---

## Reviewer A, comment 1 — Practical usefulness

**Request:** substantiate “most useful for prioritizing compounds for experimental IC50 measurement,” or moderate the claim.

**Lukas action**

- [ ] Replace the claim in Classification performance.
- [ ] Qualify “acceptable trade-off”; its acceptability depends on the screening objective.
- [ ] Apply the same scope to Abstract, Discussion and Conclusions.

**Replacement manuscript text**

> These retrospective results suggest that CardioSafe may be useful for prioritizing compounds for experimental IC50 measurement. Its suitability depends on the target channel, chemical domain and operating threshold. The observed false-positive and false-negative rates should therefore be considered in relation to the objectives and costs of the intended screening campaign. The present evaluation does not establish prospective screening utility or clinical safety.

**Response-letter draft**

> We agree that “most useful” overstates the evidence. We have moderated the statement to indicate potential usefulness for experimental prioritization and made its dependence on channel, chemical domain and operating threshold explicit. We do not claim prospective screening utility or universal superiority over the comparator models.

**Closure:** editorial change; no new prospective experiment is needed for the moderated claim.

## Reviewer A, comment 2 — Applicability domain and failure cases

**Request:** explain where predictions fail, including similarity, scaffolds, substructures or physicochemical properties.

**Lukas action**

- [ ] Retain the similarity-stratified results and their counts/intervals.
- [ ] Add the descriptor analysis below as an explicitly exploratory complement.
- [ ] Include the complete descriptor table, rather than only selected favorable comparisons.
- [ ] Keep the limitations on sparse channels and avoid claiming a validated domain cutoff or a causal substructure mechanism.

**Replacement Methods addition**

> We characterized prediction errors across six physicochemical descriptors: molecular weight, clogP, topological polar surface area (TPSA), aromatic ring count, predicted basic pKa and predicted cation fraction at pH 7.4. For each split, the 25th, 50th and 75th percentiles of each descriptor in the global training partition defined four test-set strata. Ties can produce unequal or empty strata. We report sample sizes, positive counts, classification discrimination, false-negative and false-positive rates at a probability threshold of 0.5, and regression errors where observed labels are available. Cells with fewer than five observations are not scored. These analyses are descriptive, involve correlated descriptors and overlapping compounds, and do not provide adjusted hypothesis tests or validated deployment boundaries.

**Replacement Results/Discussion addition**

> Prediction reliability varies within the test population. For tan70 hERG at 10 µM, AUROC decreases from 0.910 in the 0.5–0.7 nearest-training-similarity stratum to 0.739 below 0.3; MCC decreases from 0.476 to 0.156. For tan70 Cav1.2, AUROC is 0.939 in the 0.5–0.7 stratum and 0.574 in the 0.3–0.5 stratum. The confidence intervals and class counts are reported alongside these estimates because the lowest-similarity small-channel strata are sparse.
>
> Physicochemical stratification provides a complementary view of the errors. For hERG at 10 µM, compounds in the upper TRAIN-defined TPSA stratum have lower AUROC and higher false-negative rates than those in the lowest stratum on both splits. On tan70, the respective AUROCs are 0.849 and 0.919, with false-negative rates of 51.3% and 24.2%; on tan60 they are 0.836 and 0.915, with false-negative rates of 57.4% and 28.3%. These are exploratory associations, not isolated effects of polarity: source composition, structural similarity and other molecular properties may differ between strata. The complete six-descriptor analysis is supplied as a supplementary table. We have not established reliable scaffold-specific exclusion rules or a calibrated per-compound applicability threshold.

| hERG 10 µM group | Test n | Positives | False negatives | AUROC |
|---|---:|---:|---:|---:|
| tan70 lowest TPSA stratum | 15,949 | 1,148 | 278 / 1,148 | 0.919 |
| tan70 highest TPSA stratum | 8,921 | 308 | 158 / 308 | 0.849 |
| tan60 lowest TPSA stratum | 5,296 | 360 | 102 / 360 | 0.915 |
| tan60 highest TPSA stratum | 2,565 | 108 | 62 / 108 | 0.836 |

TPSA lower/upper quartile cutpoints are approximately 66.02/108.00 Å² on tan70 and 64.36/106.88 Å² on tan60. The calculations use the unrounded stored values; the CSV contains the actual cutpoints. These new subgroup point estimates have no newly calculated confidence intervals and should not be described as statistically significant differences.

**Response-letter draft**

> We have clarified the applicability-domain results and added a physicochemical error analysis. Performance is lower in parts of chemical space poorly represented by nearby training compounds; the additional descriptor stratification identifies groups with different error profiles, including higher false-negative rates among high-TPSA hERG compounds. We report subgroup sizes and class counts, retain the confidence intervals for the similarity analysis, and describe the new descriptor summaries as exploratory. The sparse small-channel strata do not support reliable chemical-class exclusion rules.

**Evidence:** [Descriptor groups](reports/reviewer_round2/descriptor_error_groups.csv), [tan70 similarity/potency results](reports/reviewer_round2/phase4_domain_tan70_v1_1.csv), [tan60 results](reports/reviewer_round2/phase4_domain_tan60_v1_1.csv), [potency-band error counts](reports/reviewer_round2/phase4_potency_error_counts_both.csv).

**Important denominator:** 339/610 tan70 hERG errors with observed potency lie at 1–10 µM, but there are 3,297 classification errors overall. Do not write that most errors overall lie in this potency band.

## Reviewer A, comment 3 — Repeated Introduction text

- [ ] Delete the duplicate nearest-neighbour phrase.
- [ ] Check captions and the rest of the manuscript for the same duplication.

**Replacement**

> …benchmarked against nine alternative architectures and a parameter-free nearest-neighbour retrieval floor, and compared against…

**Response-letter draft**

> Thank you for identifying this duplication. We have removed the repeated phrase.

## Reviewer A, comment 4 — Channel nomenclature

- [ ] Use **Nav1.5** and **Cav1.2** consistently in manuscript prose, tables, figures and supplements.
- [ ] Preserve exact code column names, gene symbols and bibliographic titles.

**Response-letter draft**

> We have standardized the channel notation to Nav1.5 and Cav1.2 throughout the manuscript and accompanying display material.

---

## Reviewer 4, major comment 1 — Pretrained-property leakage

**Request:** provide evidence or rationale addressing direct leakage through the pretrained property predictions; the reverse-leak audit does not address this.

**Lukas action**

- [ ] Add a dedicated pretraining/feature-provenance paragraph.
- [ ] Remove the claim that an exact property list is supplied in the model card.
- [ ] Distinguish molecular pretraining exposure, cardiac-label exposure, downstream preprocessing and comparator overlap.
- [ ] Integrate the unresolved-control limitation below; do not present these checks as proof that every possible leakage mechanism has been excluded.

**Verified evidence**

| Check | Result |
|---|---|
| Published checkpoint revision inspected | `66b895cab8adebea0cb59a8effa66b2020f204ca` |
| Published weight SHA-256 versus saved feature contract | Exact match: `7882205587d3985c07e425632d0e57ba0fc2fc778e3b47aab02aef9b7961407f` |
| Released regression output width | 199; final hidden-state width 384 |
| Feature extraction | Frozen pretrained weights; saved extraction manifests record no CardioSafe label reads |
| tan70 predicted-property scaler | Recomputed from 241,790 TRAIN rows; mean and scale differences both zero |
| tan60 predicted-property scaler | Recomputed from 306,662 TRAIN rows; mean and scale differences both zero |
| Ten final checkpoints | Descriptor scalers, regression scalers and class weights match TRAIN-only recalculation |
| Exact named mapping of all 199 outputs | Not independently reconstructed |
| Original upstream training logs / full molecule membership | Not audited in this handoff |

**Replacement manuscript text**

> The pretrained features are derived from the released DeepChem/ChemBERTa-77M-MTR checkpoint. The published MTR objective predicts molecular descriptors calculated from SMILES with RDKit, rather than experimental cardiac ion-channel activities [Ahmad et al., 2022]. We verified that the weight-file SHA-256 recorded in our extraction contract matches the published checkpoint identified in the supplementary provenance record. The pretrained transformer and property head remain frozen; feature extraction does not consume CardioSafe activity labels. For each downstream split, normalization of the 199 predicted-property coordinates is fitted exclusively on that split's training rows. Independent recalculation reproduces both sets of stored means and scales, and checks of all ten final checkpoints reproduce the training-only descriptor/target scalers and class weights.
>
> The released checkpoint emits 199 values, whereas the pretraining paper describes a nominal set of 200 properties. We retain the released outputs without assigning an independently unverified name to each coordinate. Molecular overlap with the upstream PubChem corpus has not been audited; such structural exposure is distinct from pretraining on held-out cardiac assay outcomes. These provenance checks address the pretrained-feature pathway, whereas the reverse-leak audit addresses overlap with comparator training data. Neither is a blanket proof against every possible source of bias.

**Response-letter draft**

> We agree that the reverse-leak audit does not address the provenance of our pretrained features. We now distinguish those two questions. The published ChemBERTa-2 MTR objective uses structure-computable RDKit properties, not experimental cardiac activities. We verified the exact released weight-file identity used for extraction and independently recalculated the downstream training-only scalers for both splits and all ten final checkpoints. We also clarify that overlap with the upstream molecular corpus was not excluded and that the exact named ordering of all 199 outputs was not independently reconstructed. The manuscript no longer points to a nonexistent property list in the model card or treats the comparator audit as resolving this concern.

**Sources:** [ChemBERTa-2, Section 3.1](https://arxiv.org/html/2209.01712v1), [pinned checkpoint](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR/tree/66b895cab8adebea0cb59a8effa66b2020f204ca), [upstream descriptor code](https://github.com/seyonechithrananda/bert-loves-chemistry/blob/695bc28cbaa0b00410711f1b2ab5953cd668530d/chemberta/utils/compute_norms.py).

**Evidence:** [Checkpoint provenance comparison](reports/reviewer_round2/upstream_provenance.json), [property-scaler audit](reports/reviewer_round2/pretrained_scaler_audit.csv), [checkpoint-scaler audit](reports/reviewer_round2/checkpoint_scaler_audit.json), [extraction code](scripts/build_mtr_predictions.py), [split-specific feature scaling](scripts/revision_features.py).

**Provenance limit:** the upstream example 199-descriptor normalization JSON does not exactly match this checkpoint's normalization vector. Do not use that file as proof of the precise historical output ordering. The inspected descriptor-generation code excludes Ipc, but that fact alone does not identify the original RDKit version or reconstruct the full ordered list.

### Required linked correction: randomized-label controls

The manuscript currently treats averaging/shared feature structure as an established explanation for residual real-label ranking. The existing handoff explicitly calls the cause unresolved.

**Replace that explanation with**

> Under the whole-row randomized-label protocol, the control models perform near chance against the corresponding randomized test labels. When the same fitted ensembles are evaluated against the original test labels, however, hERG 10 µM AUROC is 0.5667 on tan70 and 0.6236 on tan60. These are different evaluation targets. The cause of the residual ranking against the original labels remains unresolved. Ensemble averaging and shared feature structure are hypotheses rather than established explanations. This observation does not by itself identify a leakage mechanism, but it prevents us from presenting the randomized-label controls as a conclusive exclusion of all such mechanisms.

**Still open:** if a stronger claim is needed, investigate the residual ranking with matched diagnostics. Do not imply that authorization to continue a prior run scientifically resolved it.

## Reviewer 4, major comment 2 — Split construction and validation

**Lukas action**

- [ ] Replace the misplaced “Primary split” paragraph.
- [ ] Retain checkpoint-selection details under Training procedure.
- [ ] Add the actual split seed, global assignment, forced validation cases, v1.1 corrections and excluded-row explanation.

**Replacement Methods text**

> We used a single global compound assignment across channels. The archived split builder calculates each compound's maximum leave-one-out Tanimoto similarity to the other compounds using 2,048-bit Morgan fingerprints of radius 2. Compounds with maximum similarity below the selected cutoff form a novelty evaluation pool; the remaining compounds form the initial training pool. The evaluation pool is shuffled with NumPy random seed 42 and divided into approximately equal validation and test sets. Terfenadine and fexofenadine families, identified by the first 14 characters of their InChIKeys, are forced into validation and are exceptions to the blanket global-novelty rule.
>
> The frozen v1.1 assignments additionally move two compounds from training to validation on tan70 and three on tan60 to correct cross-fold near duplicates; no test identities change. Final train/validation/test counts are 241,790/46,328/46,326 for tan70 and 306,662/13,892/13,889 for tan60. One tan60 compound is marked excluded because the archived builder's two equal-half slices omit an odd evaluation-pool remainder. This is not evidence of an unsatisfiable similarity constraint. Every channel-specific cohort is a labelled subset of this global assignment.
>
> Checkpoints are selected on the validation partition using the weighted regression RMSE on the original pChEMBL scale: 45% hERG, 15% Nav1.5 and 40% Cav1.2. Feature/target scaling and class-weight estimation use training rows only. Direct test-to-training similarity calculations give maxima of 0.6988 and 0.5978 for tan70 and tan60, respectively. The source reconstruction and direct similarity check are separate audits.

**Response-letter draft**

> We have rewritten the split description to specify the global novelty criterion, random seed, validation/test allocation, forced validation compounds, v1.1 corrections and final counts. We also state how the validation set selects checkpoints and explain the single excluded tan60 row. The same compound cannot occupy different partitions for different channels.

**Author check before submission:** describe all uses of validation and test results in architecture/model development accurately. Do not add an unsupported claim that the test set was untouched throughout the entire research campaign merely because individual checkpoints were selected on validation.

**Evidence:** [Archived split-provenance audit](reports/reviewer_round2/phase2_split_provenance.json), [audit code](scripts/revision_split_provenance.py), appendix fold columns.

## Reviewer 4, minor comment 1 — Number formatting

- [ ] Add thousands separators to counts in prose, tables and captions: 334,444; 331,180; 46,326.
- [ ] Keep IDs, hashes, years and machine-readable values intact.

**Response-letter draft**

> We have added thousands separators consistently to numerical counts throughout the manuscript, tables and captions.

## Reviewer 4, minor comment 2 — Dataset appendix

**Prepared:** [Additional_file_1_cardiosafe_v3.csv.gz](reports/reviewer_round2/Additional_file_1_cardiosafe_v3.csv.gz), containing all 334,444 standardized compounds.

- [ ] Upload this file as the journal's machine-readable appendix.
- [ ] Use the data dictionary below in the supplementary description.
- [ ] Replace the repository-only availability paragraph and unfinished “Specific contents” sentence.
- [ ] Cite the immutable commit containing this handoff or an archived release. The mutable main-branch label file is not the source of these final results.

**Replacement availability text**

> Additional file 1 provides the standardized SMILES, InChIKeys, version 3 activity labels and global tan70/tan60 split assignments for all 334,444 compounds used in this study. Missing activity values are explicitly distinguished from non-blocker labels. The source label matrix has SHA-256 4492a02c3091f0669949f02bbb044c152a76a126bcb42269d5295310c9a41ecb. The associated repository release supplies the data dictionary, export script and integrity manifest. Earlier label versions are historical and must not be substituted for the version used here.

**Response-letter draft**

> We provide the requested machine-readable appendix, containing standardized SMILES and all five activity-classification endpoints, together with compound identifiers, the three regression labels and both split assignments. The export was checked for one-to-one identity alignment and numerical consistency with the version 3 labels and final test prediction arrays. A data dictionary specifies the label definitions and missing-value convention.

**Data dictionary**

| Columns | Meaning |
|---|---|
| row_idx | Zero-based compound row identifier used in the released arrays |
| smiles, inchikey | Released standardized structure and its InChIKey |
| herg_blocker_10um, herg_blocker_1um | Curated binary hERG labels at the indicated nominal thresholds |
| nav15_blocker, cav12_blocker, iks_blocker | Curated binary labels at the nominal 10 µM threshold |
| herg_pchembl, nav15_pchembl, cav12_pchembl | Pooled pChEMBL potency targets; not uniformly exact experimental pIC50 |
| tan70_v1_1, tan60_v1_1 | train, val, test; tan60 also has one excluded row |

Binary labels are 1 = blocker and 0 = non-blocker under the published evidence-voting policy. Empty label cells mean unavailable, **not negative**. Binary votes and pooled potency targets are separately curated and must not be silently reconciled by thresholding the regression column. The CSV preserves the frozen values; 10 µM and 1 µM are endpoint definitions, not direct clinical-risk labels.

| Endpoint | Labelled compounds | Positive labels |
|---|---:|---:|
| hERG 10 µM | 331,111 | 25,065 |
| hERG 1 µM | 330,525 | 5,457 |
| Nav1.5 | 3,160 | 1,240 |
| Cav1.2 | 1,138 | 548 |
| IKs | 114 | 30 |
| hERG / Nav1.5 / Cav1.2 regression | 23,667 / 2,855 / 997 | Not applicable |

These are endpoint counts; the hERG channel union of 331,180 is a different quantity.

## Reviewer 4, minor comment 3 — Rationale for 1 and 10 µM

- [ ] Add the rationale below and the supporting primary references.
- [ ] Delete “aligned with the CiPA clinical regime.”
- [ ] Keep strict boundary rules explicit and distinguish them from comparator conventions.

**Replacement Methods text**

> We use 10 µM as a conventional broad blockade endpoint and an additional 1 µM endpoint to characterize more potent hERG activity. Both cutoffs have precedent in hERG modelling; published work also shows that model performance depends on the thresholding policy [Siramshetty et al., 2018; Sanches et al., 2024]. The nominal 10 µM endpoint supports comparison with published multichannel models, including CardioGenAI's pIC50 boundary of 5 [Kyro et al., 2025]. In this dataset, both thresholds are represented for hERG, whereas Nav1.5, Cav1.2 and IKs are evaluated at 10 µM only.
>
> Exact-potency votes use pChEMBL > 5 and > 6, with the separately specified rules for censored and inhibition-percentage evidence. Boundary conventions are not universally identical across publications: CardioGenAI describes blocker assignment at pIC50 ≥ 5, whereas our exact-potency rule is strict. Our comparisons evaluate all predictions against the same curated CardioSafe test labels. These potency thresholds do not incorporate clinical exposure and are not proarrhythmic-risk or safety classifications.

**Response-letter draft**

> We have added the operational rationale and references for the 10 µM blockade endpoint and the stricter 1 µM hERG endpoint. We clarify that these choices have precedent in molecular prediction benchmarks, rather than constituting clinical risk thresholds, and make the boundary conventions and common evaluation labels explicit.

**References:** [Siramshetty et al., The Catch-22 of Predicting hERG Blockade Using Publicly Accessible Bioactivity Data](https://pubmed.ncbi.nlm.nih.gov/29772901/); [Sanches et al., Pred-hERG 5.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC11187631/); [Kyro et al., CardioGenAI, Data featurization](https://pmc.ncbi.nlm.nih.gov/articles/PMC11881490/).

**Do not claim:** that the two cutoff choices are uniquely optimal, that our policy replicates the gap-exclusion policy of Siramshetty et al., or that the comparator boundary rules are identical.

## Reviewer 4, minor comment 4 — Why include IKs?

**Recommended disposition:** retain it explicitly as exploratory and answer directly that improvement to the primary endpoints has not been established. Do not manufacture an ablation result.

- [ ] Add the rationale and limitation below.
- [ ] Remove any suggestion of demonstrated positive transfer from IKs.
- [ ] If the editor requires quantitative proof of benefit, commission a matched no-IKs-supervision ablation; the present evidence cannot answer that stronger question.

**Replacement manuscript text**

> The IKs classification head was included as an exploratory extension to an additional cardiac repolarization current. We do not claim that this auxiliary task improves hERG, Nav1.5 or Cav1.2 predictions: the present experiments do not isolate the effect of adding IKs supervision. Only 114 compounds carry IKs labels, and the tan70 and tan60 test sets each contain one positive. These test cohorts are insufficient for a reliable assessment of IKs generalization, and its output should not be used to support the model's primary performance claims.

**Response-letter draft**

> IKs was included for exploratory coverage of an additional cardiac repolarization current. We agree that biological motivation alone does not demonstrate a predictive benefit. We have explicitly stated that improvement to the primary tasks has not been established and that the one-positive IKs test cohorts do not support a reliable performance conclusion. We do not interpret the single-task comparisons as an ablation of IKs supervision.

**Remaining evidence gap:** a direct full-model versus no-IKs-supervision experiment. This handoff supplies an honest bounded response, not the missing experiment.

## Reviewer 4, minor comment 5 — Five seeds and confidence intervals

**New evidence:** all 31 nonempty seed subsets per split, covering ensemble sizes 1–5, with the same sigmoid averaging and original-scale regression averaging as the final model. Four-member rows correspond to all five leave-one-seed-out ensembles. These subsets overlap and are not independent statistical replicates.

- [ ] Separate test-sample bootstrap uncertainty from training-seed variability.
- [ ] Add the stability table below or a condensed version.
- [ ] Include the complete subset table as supplementary evidence.
- [ ] Avoid claiming that five seeds are universally sufficient or that all model-comparison conclusions are unchanged; comparison deltas were not recomputed for every subset.

| Split / endpoint | Single-seed range | Four-seed range | Five-seed result |
|---|---:|---:|---:|
| tan70 hERG 10 µM AUROC | 0.8831–0.8962 | 0.8944–0.9007 | 0.9007 |
| tan70 hERG 1 µM AUROC | 0.9336–0.9382 | 0.9410–0.9420 | 0.9420 |
| tan70 Nav1.5 AUROC | 0.8128–0.8363 | 0.8250–0.8278 | 0.8261 |
| tan70 Cav1.2 AUROC | 0.8056–0.8276 | 0.8199–0.8235 | 0.8217 |
| tan60 hERG 10 µM AUROC | 0.8751–0.8938 | 0.8993–0.9032 | 0.9031 |
| tan60 hERG 1 µM AUROC | 0.9034–0.9240 | 0.9224–0.9262 | 0.9248 |
| tan60 Nav1.5 AUROC | 0.7426–0.8275 | 0.8219–0.8378 | 0.8323 |
| tan60 Cav1.2 AUROC | 0.8121–0.8635 | 0.8227–0.8546 | 0.8457 |
| tan70 weighted RMSE | 0.7202–0.7449 | 0.7151–0.7195 | 0.7163 |
| tan60 weighted RMSE | 0.7792–0.8308 | 0.7783–0.7960 | 0.7845 |

**Replacement Methods/Results text**

> The reported 95% confidence intervals use 1,000 bootstrap resamples of test molecules, conditional on the fitted five-member ensemble; they are not intervals estimated from five seed-level metric values. For paired comparisons, the same resampled molecules are used for both models. We separately assessed seed sensitivity by evaluating every nonempty subset of the five trained models at ensemble sizes one through five. These overlapping subsets provide descriptive sensitivity ranges rather than additional independent fits or confidence intervals.
>
> Omitting one model changes hERG AUROC by at most approximately 0.0063 across the two hERG endpoints and two splits. Greater sensitivity remains for tan60 Cav1.2: the four-member AUROC range is 0.8227–0.8546, compared with 0.8457 for all five models. These results characterize the available ensemble and support reporting its residual seed sensitivity. They do not prove that five initializations capture the full training distribution or establish a universally sufficient ensemble size.

**Response-letter draft**

> We apologize for the ambiguity. The headline intervals arise from test-molecule bootstrap resampling of the fixed ensemble, not five independent observations. We have clarified their conditional interpretation and added an exhaustive analysis of the 31 nonempty subsets of the five fitted models on each split. The leave-one-seed-out analysis shows relatively small changes for hERG and greater sensitivity for tan60 Cav1.2. We report these results rather than claiming universal sufficiency of five seeds, and retain the limitation that the bootstrap intervals do not capture full training variability.

**Evidence:** [All seed subsets](reports/reviewer_round2/seed_subsets.csv), [summary](reports/reviewer_round2/seed_stability_summary.csv).

---

## Mandatory architecture and environment rewrite

### Replace the branch descriptions and Figure 1 caption

> CardioSafe receives three 2,048-bit fingerprints, 20 physicochemical descriptors and 583 frozen ChemBERTa-MTR features. The 6,144 fingerprint inputs pass through a 512-to-256 feed-forward encoder. The 20 descriptors pass through two 64-wide layers after training-partition normalization and the signed-log transformation. The final-layer initial-token representation (384 coordinates) and the pretrained property predictions (199 coordinates) are concatenated and jointly projected from 583 to 128 dimensions. Concatenating these three branch outputs produces a 448-dimensional representation.
>
> Eight task-specific heads read this shared representation: three pChEMBL regression heads for hERG, Nav1.5 and Cav1.2, and five classification heads for hERG at 10 and 1 µM, Nav1.5, Cav1.2 and exploratory IKs. The pretrained transformer/property head remain frozen. The downstream model has 3,820,680 trainable parameters. No query-token or attention module is part of this final concatenation model.

**Figure 1 caption**

> CardioSafe architecture. SMILES-derived fingerprints (6,144 inputs), physicochemical descriptors (20 inputs) and frozen ChemBERTa-MTR features (384 initial-token coordinates plus 199 property predictions) are encoded separately to 256, 64 and 128 dimensions. Their concatenation feeds five classification and three regression heads. The pretrained transformer and property head are frozen; the IKs classifier is exploratory.

- [ ] Correct all matching descriptions, including the graphical abstract and interpretation of fingerprint/descriptor/MTR probes.
- [ ] Remove the query-token exception from the Initialization paragraph.
- [ ] Describe the signed-log transform explicitly; do not say all descriptors are only z-scored.
- [ ] Do not describe the 199 predictions as an independent experimental biological modality: they are deterministic outputs from the retained pretrained representation.

### Replace the blanket hardware/software claim

> The ten final fits (five seeds per split) were run in a Linux environment with Python 3.12.3 and PyTorch 2.10.0+cpu, as recorded in the per-run provenance files. Feature extraction, comparator execution and subsequent analyses used separately recorded environments.

The exact host/CPU model is not established by those provenance records alone. If a detailed hardware specification is required, recover it from the run infrastructure records. Do not substitute the local laptop specification or imply that all final fits used MPS.

## Remaining manuscript consistency edits

- [ ] Correct the domain caption: tan60 contains observations in the common 0.5–0.7 reporting bin, but none at or above 0.60.
- [ ] Distinguish nearest similarity to global TRAIN from read-across coverage among neighbours with a label for a particular endpoint. Those denominators are not interchangeable.
- [ ] Replace `primary_regression_aggregate` with “weighted RMSE score” in Table 3.
- [ ] Correct pChEMBL/pIC50 definitions to **negative** base-10 logarithm of molar potency; use pChEMBL for pooled endpoint types.
- [ ] Correct the prevalence/no-skill AUPRC table cross-reference.
- [ ] Do not imply an IKs regression head in Conclusions.
- [ ] Remove “consistent across all four channels” where it implies validated IKs performance.
- [ ] Keep the no-multiplicity-adjustment qualification beside the paired architecture comparisons.
- [ ] Keep the named-drug panel explicitly illustrative; it contains no test-fold compounds.
- [ ] Replace the incomplete Additional files section with the actual submitted filenames.
- [ ] Reconcile the retained report's branch/code state with the final manuscript; current helper code is not a complete byte-identical historical training snapshot.

## Reproduction and file integrity

Run from a checkout with the companion data, saved final predictions, model checkpoints and feature caches:

```bash
python scripts/reviewer_round2_evidence.py \
  --source /path/to/full/artifact/checkout \
  --out reports/reviewer_round2
```

The script verifies source identities, reproduces the five-seed predictions, recalculates TRAIN-only scaling, writes all seed-subset and descriptor summaries, exports the appendix, and verifies the parsed appendix round trip. It does not retrain models or edit source labels.

[Evidence manifest](reports/reviewer_round2/evidence_manifest.json) records input hashes, output hashes, numerical-library versions and validation results. [Upstream provenance](reports/reviewer_round2/upstream_provenance.json) records the separately fetched checkpoint identity and the limitations of the upstream normalization comparison.

The gzip appendix is a convenient transport format. Decompress it to CSV if required by the journal, without changing its cells. Use the repository's existing data licensing and source-attribution terms.

## Final integration checklist for Lukas

- [ ] All 11 reviewer comments have a direct response.
- [ ] The Google manuscript contains the agreed replacements.
- [ ] The unresolved control result is accurately disclosed.
- [ ] No IKs benefit or universal five-seed sufficiency claim has been introduced.
- [ ] Supplement names and references match the actual submission files.
- [ ] The appendix contains v3 labels and the stated fold assignments.
- [ ] Methods, figures and captions describe the implemented architecture and recorded run environment.
- [ ] Add exact revised page/line references to every response paragraph.
- [ ] Convert draft response wording into the final response letter only after checking the corresponding changes.

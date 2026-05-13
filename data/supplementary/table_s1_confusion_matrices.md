# Table S1 — Per-head confusion matrices

Confusion matrices for the deployed CardioSafe 5-seed ensemble at the
default operating threshold of 0.5 on the sigmoid output, per binary
head per split. Numbers are the ensemble (mean-of-seeds) predictions
evaluated against the curated labels_v1 ground truth on the tan70 and
tan60 test folds.

## Counts

| head | split | n | prev. | TP | FP | TN | FN |
| :-- | :--: | --: | --: | --: | --: | --: | --: |
| hERG 10 µM | tan70 | 46,120 | 2.03% | 600 | 1,070 | 44,113 | 337 |
| hERG 1 µM | tan70 | 46,042 | 0.45% | 116 | 326 | 45,510 | 90 |
| Nav1.5 10 µM | tan70 | 218 | 27.98% | 43 | 35 | 122 | 18 |
| Cav1.2 10 µM | tan70 | 160 | 27.50% | 31 | 22 | 94 | 13 |
| IKs 10 µM | tan70 | 14 | 7.14% | 0 | 1 | 12 | 1 |
| hERG 10 µM | tan60 | 13,831 | 3.14% | 292 | 827 | 12,570 | 142 |
| hERG 1 µM | tan60 | 13,767 | 0.72% | 55 | 250 | 13,418 | 44 |
| Nav1.5 10 µM | tan60 | 86 | 26.74% | 15 | 11 | 52 | 8 |
| Cav1.2 10 µM | tan60 | 59 | 20.34% | 3 | 7 | 40 | 9 |
| IKs 10 µM | tan60 | 11 | 9.09% | 0 | 0 | 10 | 1 |

## Derived metrics

| head | split | sensitivity | specificity | precision | accuracy | MCC | AUC |
| :-- | :--: | --: | --: | --: | --: | --: | --: |
| hERG 10 µM | tan70 | 0.6403 | 0.9763 | 0.3593 | 0.9695 | +0.4657 | 0.9169 |
| hERG 1 µM | tan70 | 0.5631 | 0.9929 | 0.2624 | 0.9910 | +0.3806 | 0.9587 |
| Nav1.5 10 µM | tan70 | 0.7049 | 0.7771 | 0.5513 | 0.7569 | +0.4514 | 0.8306 |
| Cav1.2 10 µM | tan70 | 0.7045 | 0.8103 | 0.5849 | 0.7812 | +0.4885 | 0.8411 |
| IKs 10 µM | tan70 | 0.0000 | 0.9231 | 0.0000 | 0.8571 | -0.0769 | 0.3846 |
| hERG 10 µM | tan60 | 0.6728 | 0.9383 | 0.2609 | 0.9299 | +0.3907 | 0.8983 |
| hERG 1 µM | tan60 | 0.5556 | 0.9817 | 0.1803 | 0.9786 | +0.3084 | 0.9322 |
| Nav1.5 10 µM | tan60 | 0.6522 | 0.8254 | 0.5769 | 0.7791 | +0.4603 | 0.7923 |
| Cav1.2 10 µM | tan60 | 0.2500 | 0.8511 | 0.3000 | 0.7288 | +0.1084 | 0.7660 |
| IKs 10 µM | tan60 | 0.0000 | 1.0000 | nan | 0.9091 | +0.0000 | 0.5000 |

## Notes

- *Threshold:* 0.5 on the sigmoid output of each binary head. Calibrated operating points per head are explored separately in the threshold-sweep analysis; this table is the default-threshold confusion matrix.
- *Ensemble:* 5-seed (seeds 42–46) mean of post-sigmoid probabilities, then thresholded.
- *IKs:* exploratory head — n is small (14 on tan70, 11 on tan60). Confusion-matrix entries are exact integers, not derived from continuous metric collapses.
- *Prevalence* is `n_positives / n` for the listed head on the listed split's test fold (subset of compounds with non-NaN label for that head).
- *AUC* and *MCC* are reproduced here for cross-reference with main-text Tables 2 and 4.

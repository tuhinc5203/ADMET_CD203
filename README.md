# Predicting hERG Blockade from SMILES (CDD 203, Homework 1)

**Group 3:** Suraj Godithi, Olivia Lu, Aditi Shankar, Tuhin Chatterjee

**Web app:** TODO - add the Hugging Face Space URL after deployment

| File | What it is |
|---|---|
| `Part1_Olivia-Lu-Suraj-Godithi-Aditi-Shankar-Tuhin-Chatterjee.py` | Training script (featurizer, model, ONNX export) |
| `Part1_Olivia-Lu-Suraj-Godithi-Aditi-Shankar-Tuhin-Chatterjee.onnx` | Trained model submitted for evaluation |
| `Part 2.zip` | Web app code (`app.py`, Space config, requirements) plus an identical copy of the ONNX model |

Reproduce with `conda env create -f environment.yml && conda activate cdd203-hw1`, then
`python Part1_Olivia-Lu-Suraj-Godithi-Aditi-Shankar-Tuhin-Chatterjee.py`. Training takes about 15 seconds
and, because every random seed is fixed, rewrites the ONNX file byte-for-byte.

## Modeling decisions

**(a) Decision threshold: 0.57.** Raising the threshold above 0.5 means a molecule needs stronger evidence
before it is flagged, so fewer non-blockers are falsely flagged (higher specificity and precision) at the cost
of missing more true blockers (lower sensitivity); lowering it does the opposite. On our 20% held-out test
split, moving from 0.5 to 0.57 raised specificity from 0.819 to 0.902 and precision from 0.833 to 0.893, while
sensitivity fell from 0.830 to 0.752. We chose 0.57 because we see the model as an early triage filter for
large virtual libraries, where every compound flagged is one a chemist sets aside. A flag should therefore be
trustworthy (about 9 in 10 flagged molecules are true blockers), while the model still catches about three
quarters of blockers. Candidates that pass still go on to experimental hERG patch-clamp testing, which catches
the blockers the model misses. We stopped at 0.57 rather than going higher to keep sensitivity above 0.7.

**(b) Feature engineering.** Each SMILES is parsed with RDKit, and salts are stripped by keeping the largest
fragment. We then build one fixed-length vector of 2,056 numbers: a 2,048-bit Morgan *count* fingerprint
(radius 2, ECFP4-like) plus 8 cheap physicochemical descriptors. Those are molecular weight, TPSA, H-bond
donors, aromatic rings, total rings, fraction sp3 carbons, heavy-atom count and a count of basic amines (a
SMARTS pattern that excludes amides, anilines and sulfonamides). The fingerprint hashes substructures into a
fixed number of bins, so the vector length does not depend on SMILES length; we tested 190+ character SMILES
explicitly. The descriptors encode the classic hERG pharmacophore: a lipophilic, aromatic molecule carrying a
protonatable nitrogen. Parsing applies only the sanitization steps the fingerprint needs, which speeds up
featurization, and unparseable SMILES give an all-zero row instead of an error.

**(c) Model and hyperparameters.** We used scikit-learn's `ExtraTreesClassifier` with `n_estimators=300`,
`min_samples_leaf=3`, `max_features="sqrt"` and `random_state=0`, trained on the train and validation splits
(80% of the data). Tree ensembles handle sparse count fingerprints and unscaled descriptors without
preprocessing, capture non-linear substructure interactions, and export cleanly to ONNX. Extra trees edged out
a random forest of the same size in 5-fold cross-validation (AUROC 0.912 vs 0.908). `min_samples_leaf=3`
smooths the predicted probabilities and keeps the ONNX file at 28.5 MB, under the 50 MB limit. On the
held-out test split the model reaches ROC-AUC 0.914 and PR-AUC 0.923.

## Use of LLM / generative AI

We used Claude (Anthropic) through Claude Code, an AI coding assistant, throughout the assignment. For the web
app, Claude found that the first draft of `app.py` computed different features (1,024 bits, different
descriptors) and used a different threshold than our final training script, which would have made the app
disagree with the submitted model. It rewrote the app to import `featurize`, `NAME` and `THRESHOLD` directly
from the training script, so the two cannot drift apart. It then checked that the app and the ONNX model give
identical probabilities on all 10,000 training molecules. It also diagnosed a dependency problem: Gradio 5.35
fails to import alongside `huggingface_hub` 2.x, so we pinned `huggingface_hub<1.0` for the Space. Finally, it
set up the GitHub repository and drafted this README. We reviewed and edited every suggestion and re-ran the
training script and `TestModel.py` ourselves before submitting.

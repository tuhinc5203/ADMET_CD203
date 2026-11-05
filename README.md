# Predicting hERG Blockade from SMILES (CDD 203, Homework 1)

**Group 3:** Suraj Godithi, Olivia Lu, Aditi Shankar, Tuhin Chatterjee

**Web app:** https://herg-blocker-app.onrender.com (hosted on Render's free tier, so the first visit after a period of inactivity can take about a minute to wake up)

| File | What it is |
|---|---|
| `Part1_Olivia-Lu-Suraj-Godithi-Aditi-Shankar-Tuhin-Chatterjee.py` | Training script (featurizer, model, ONNX export) |
| `Part1_Olivia-Lu-Suraj-Godithi-Aditi-Shankar-Tuhin-Chatterjee.onnx` | Trained model submitted for evaluation |
| `Part 2.zip` | Web app code (`app.py`, Space config, requirements) plus an identical copy of the ONNX model |

Reproduce with `conda env create -f environment.yml && conda activate cdd203-hw1`, then
`python Part1_Olivia-Lu-Suraj-Godithi-Aditi-Shankar-Tuhin-Chatterjee.py`. Training takes about 15 seconds
and, because every random seed is fixed, rewrites the ONNX file byte-for-byte.

## Modeling Decisions

**Decision threshold.** We selected a classification threshold of **0.57** rather than the default 0.50 to favor specificity and precision while retaining useful sensitivity. On our 20% held-out test split, increasing the threshold from 0.50 to 0.57 increased specificity from **0.819 to 0.902** and precision from **0.833 to 0.893**, while sensitivity decreased from **0.830 to 0.752**. We chose this operating point because we wanted positive blocker predictions to be relatively selective while still identifying a substantial majority of true blockers. We stopped at 0.57 rather than increasing the threshold further in order to keep sensitivity above 0.70, making the final threshold a deliberate tradeoff between reducing false-positive blocker classifications and retaining blocker detection.

**Feature engineering.** We used RDKit to convert each SMILES string into a fixed-length vector of **2,056 features**: a 2,048-dimensional radius-2 Morgan count fingerprint plus eight physicochemical and structural descriptors—molecular weight, TPSA, hydrogen-bond donor count, aromatic ring count, total ring count, fraction of sp3 carbons, heavy-atom count, and basic amine count. The Morgan count fingerprint represents local molecular environments while retaining their frequency, and the descriptors provide complementary information about molecular size, polarity, hydrogen bonding, ring structure, and composition. For salts or other multi-fragment SMILES, we retained the fragment with the greatest number of heavy atoms. The fixed-length representation accommodates molecules with different SMILES lengths, while unparseable inputs return an all-zero feature vector rather than crashing the pipeline. Because featurization speed was also evaluated, we limited the representation to inexpensive descriptors and the RDKit sanitization operations required for these features.

**Model and hyperparameters.** Our final classifier was an `ExtraTreesClassifier` with **300 trees, `min_samples_leaf=3`, `max_features="sqrt"`, and `random_state=0`**. Tree ensembles were well suited to the combination of fingerprint and descriptor features because they can model nonlinear relationships and feature interactions without requiring feature scaling, provide class probabilities, and export cleanly to ONNX. In 5-fold cross-validation, Extra Trees achieved an AUROC of approximately **0.912**, compared with **0.908** for a Random Forest of the same size, supporting our final model choice. We used `min_samples_leaf=3` to avoid very small terminal leaves and produce less extreme probability estimates while satisfying the assignment's deployment constraints. On the held-out test split, the final model achieved a **ROC-AUC of 0.914** and **PR-AUC of 0.923**, providing threshold-independent evidence that the model could distinguish blockers from non-blockers before applying our selected 0.57 decision threshold.

## Use of LLM / Generative AI

We used **Claude, primarily through Claude Code**, as a coding, experimentation, and debugging assistant throughout the project. It helped us write code to systematically compare feature representations, classifiers, hyperparameters, and decision thresholds, as well as troubleshoot implementation and deployment issues. One particularly useful example was identifying that an early version of `app.py` used different features and a different threshold from our final training pipeline, which could have caused the deployed application to disagree with the model we evaluated; after correcting this, we compared prediction outputs to verify consistency with the exported ONNX model. Claude also helped us interpret dependency and deployment errors and draft initial documentation. However, we found that LLM-generated code can appear reasonable and run successfully without necessarily being appropriate for the specific modeling pipeline, and suggested parameters did not always improve performance. We therefore treated LLM outputs as **suggestions to test rather than answers to accept**: we reviewed generated code, reran experiments, compared validation and cross-validation metrics, checked model predictions, and used `TestModel.py` to verify the final implementation. Overall, Claude accelerated experimentation and debugging, while final modeling decisions remained based on our own quantitative evaluation and testing.

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

### Modeling Decisions

## Decision threshold
We selected a classification threshold of 0.57 rather than using the default threshold of 0.50. The threshold determines the tradeoff between sensitivity and specificity: increasing it requires stronger model evidence before a compound is labeled a blocker, which generally reduces false positives but increases false negatives.

On our 20% held-out test split, increasing the threshold from 0.50 to 0.57 increased specificity from 0.819 to 0.902 and precision from 0.833 to 0.893, while sensitivity decreased from 0.830 to 0.752. We chose this operating point because we wanted positive predictions to be relatively selective: at 0.57, approximately 89% of compounds predicted as blockers were true blockers in our held-out data. At the same time, sensitivity remained above 0.70, so the classifier continued to identify a substantial majority of blockers.

This threshold therefore represents a deliberate tradeoff rather than an attempt to maximize every metric simultaneously. We prioritized reducing false-positive blocker classifications while retaining useful sensitivity.

## Feature engineering
We converted each SMILES string into a fixed-length numerical representation using RDKit. For multi-fragment molecules such as salts, we retained the fragment containing the greatest number of heavy atoms before calculating features.

Each molecule was represented by 2,056 features, a 2,048-dimensional radius-2 Morgan count fingerprint plus eight physicochemical and structural descriptors. The descriptors were molecular weight, topological polar surface area (TPSA), hydrogen-bond donor count, aromatic ring count, total ring count, fraction of sp3 carbons, heavy-atom count, and basic amine count.

The Morgan fingerprint represents local molecular environments, while the additional descriptors provide complementary global information about molecular size, polarity, hydrogen bonding, ring structure, and composition. We used a count fingerprint so that the representation retains information about the frequency of hashed molecular environments rather than only their presence or absence.

The fixed-length representation also allows molecules with different SMILES lengths to be processed by the same classifier. Our implementation handles salts and long SMILES and returns an all-zero feature vector for an input that cannot be parsed, preventing a malformed molecule from crashing the prediction pipeline.

We also considered computational efficiency because the assignment evaluates prediction speed including featurization. The final implementation therefore used a limited set of inexpensive descriptors and only the RDKit sanitization operations required by the selected representation.

## Model and hyperparameters
Our final classifier was an ExtraTreesClassifier with 300 trees, min_samples_leaf=3, max_features="sqrt", and random_state=0.

Tree ensembles were well suited to our feature representation because they can capture nonlinear relationships and interactions among fingerprint and descriptor features without requiring feature scaling. They also provide class probabilities and can be exported to ONNX for deployment.

We compared Extra Trees with a Random Forest of the same size using 5-fold cross-validation. Extra Trees achieved an AUROC of approximately 0.912, compared with 0.908 for Random Forest, which supported our selection of Extra Trees. We used min_samples_leaf=3 to avoid very small terminal leaves and produce less extreme probability estimates. The final model also satisfied the assignment's deployment constraints, including the ONNX model-size limit.

On our held-out test split, the final model achieved a ROC-AUC of 0.914 and a PR-AUC of 0.923. Because these metrics evaluate the probability predictions independently of the final classification threshold, they provided evidence that the model could distinguish blockers from non-blockers before we selected the 0.57 operating threshold.

## Use of LLM / generative AI

We used Claude, primarily through Claude Code, as a coding, experimentation, and debugging assistant throughout the project. It was especially useful for speeding up repetitive development tasks and helping us explore alternatives, while we remained responsible for evaluating whether its suggestions were appropriate for our model.

One major use was during model development and hyperparameter testing. Claude helped us write and modify code to compare different feature representations, classifiers, and parameter settings, including Random Forest versus Extra Trees, tree and leaf parameters, fingerprint configurations, and classification thresholds. This allowed us to test alternatives more systematically rather than changing parameters manually one at a time. However, we did not choose a model simply because the LLM suggested it. We evaluated the resulting models using validation and cross-validation metrics such as AUROC, sensitivity, specificity, and precision, and based our final modeling decisions on those measured results.

Claude was also useful for checking consistency across the pipeline. During development, it identified that an early version of app.py used different features and a different threshold from the final training script. Although the application could still run, this would have caused the deployed app to behave differently from the model we had evaluated. After correcting the inconsistency, we compared prediction outputs to confirm that the application and exported ONNX model agreed. We also used Claude to help interpret dependency and deployment errors involving Gradio and Hugging Face and to create an initial draft of the project documentation.

A key limitation we observed was that LLM-generated code can appear reasonable and run successfully without necessarily being correct for the specific modeling pipeline. A suggested hyperparameter may not improve performance, and seemingly minor changes to preprocessing or feature definitions can alter model behavior. We therefore treated LLM outputs as suggestions to test rather than answers to accept. We reviewed generated code, reran experiments, compared quantitative results, checked model predictions, and used the provided TestModel.py tests to verify the final implementation.

Overall, Claude was most valuable as an iterative development assistant. It accelerated experimentation, debugging, and code review, but empirical evaluation remained essential for deciding which suggestions were actually useful.

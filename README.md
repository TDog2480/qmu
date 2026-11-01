# Auditing Hybrid Quantum Machine Unlearning for Membership Attacks

Current structure:

```
qmu_inf/
├── README.md
├── .gitignore
├── .vscode/
│
├── HQNN.py                     model + original training (imported by everything)
├── attack_scoring.py               shared pool / split / 4-group report
├── roc_utils.py                ROC plotting
│
├── unlearning/
│   └── QMU_gradient_ascent.py
├── attacks/
│   ├── metric_attack.py
│   ├── mlp_attack.py          
│   ├── vqc_attack.py
│   └── qsvm_attack.py
├── reference-comparison/       
│   ├── HQNN_target.py
│   └── unlearned_vs_reference.py
│
├── checkpoints
│   ├── attack-models/
│   ├── original/
│   ├── reference/
│   └── unlearned/
├── roc_curve_output/
└── data/                       MNIST
```
## Attack outputs

Every trained attack (`{name}` = `mlp`, `vqc`, `qsvm`) writes the same set of files. The attack model, threshold and scaler are fit on the original model's attack-train rows, then frozen and reused unchanged on the unlearned model.

| Role | Location | Metric | MLP | VQC | QSVM |
|---|---|---|---|---|---|
| Fitted attack model | `checkpoints/attack-models/` | — | `attack_model_mlp.pth` (state_dict) | `attack_model_vqc.pth` (state_dict) | `attack_model_qsvm.pkl` (SVC + reference set) |
| Config: scaler, threshold, metadata | `checkpoints/attack-models/` | — (thresholds printed only) | `attack_scaler_mlp.pkl` | `attack_scaler_vqc.pkl` | `attack_config_qsvm.pkl` |
| Per-row scores, both models | `checkpoints/attack-models/` | — | `scores_mlp.npz` | `scores_vqc.npz` | `scores_qsvm.npz` |
| ROC, original model | `roc_curve_output/` | `metric_attack_roc.npz` (all rows) | `mlp_attack_roc.npz` | `vqc_attack_roc.npz` | `qsvm_attack_roc.npz` |
| ROC, unlearned model | `roc_curve_output/` | — (original only) | `mlp_attack_roc_unlearned.npz` | `vqc_attack_roc_unlearned.npz` | `qsvm_attack_roc_unlearned.npz` |

The trained attacks' ROC files use held-out rows only, since the attack model never saw them. The metric attacks have no trained model, so their ROC uses all rows.

### Config keys (per `{name}` in `checkpoints/attack-models/`)

| Key | Meaning |
|---|---|
| `attack` | Attack name |
| `scaler` | `StandardScaler` fit on attack-train rows |
| `threshold` | Frozen decision threshold, in `score_space` units |
| `score_space` | MLP: `logit`; VQC: `probability`; QSVM: `svm_decision_function` |
| `heldout_auc` | Original-model AUC on held-out rows (comparable across attacks) |
| `target_checkpoint`, `attack_split_seed`, `forget_class` | Provenance |
| Attack-specific | MLP: `input_dim`, `epochs`; VQC: `bandwidth`; QSVM: `bandwidth`, `C`, `n_qubits`, `selection_auc` |

Scores are only comparable with a threshold in the same `score_space`. For example, the MLP threshold must be applied to logits from `mlp_scores`, not to `model(x)` probabilities.

To regenerate any ROC or 4-group table without retraining or recomputing the quantum kernel, use `scores_{name}.npz`.


Run HQNN.py to train model

Run unlearning/QMU_gradient_ascent.py to run unlearning loop

Run each attack under attacks/ 

Run roc_utils.py after running attacks to see ROC curves

Run HQNN_target.py to train reference model and then unlearned_vs_reference.py to see per sample breakdown

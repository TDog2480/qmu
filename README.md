# Auditing Hybrid Quantum Machine Unlearning for Membership Attacks

Current structure:

```
qmu_inf/
├── README.md
├── .gitignore
├── .vscode/
│
├── HQNN.py                     model + original training (imported by everything)
├── attack_scoring.py               shared pool / split / 4-group report (was mia_common.py)
├── roc_utils.py                ROC plotting
│
├── unlearning/
│   └── MU_gradient.py
├── attacks/
│   ├── metric_attack.py
│   ├── metric_roc_unlearned_calc.py
│   ├── mlp_attack.py           (was MIA.py)
│   ├── vqc_attack.py
│   └── qsvm_attack.py
├── reference_comparison/       
│   ├── HQNN_target.py
│   └── unlearned_vs_reference.py
│
├── checkpoints
│   ├── attack-models/
│   ├── original/
│   └── unlearned/
├── roc_curve_output/
└── data/                       MNIST
```

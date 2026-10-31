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
│   └── QMU_gradient_ascent.py
├── attacks/
│   ├── metric_attack.py
│   ├── mlp_attack.py          
│   ├── vqc_attack.py
│   └── qsvm_attack.py
├── reference_comparison/       
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

Run HQNN.py to train model

Run unlearning/QMU_gradient_ascent.py to run unlearning loop

Run each attack under attacks/ 

Run roc_utils.py after running attacks to see ROC curves

Run HQNN_target.py to train reference model and then unlearned_vs_reference.py to see per sample breakdown

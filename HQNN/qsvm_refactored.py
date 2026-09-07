"""
Quantum-kernel SVM (QSVM) membership inference attack for evaluating HQNN unlearning.

Same attack model as before -- a classical SVM on a fidelity kernel over an 8-qubit
angle embedding -- but rebuilt on the shared evaluation pool (Part 4) so its numbers
line up with the threshold attacks, the MLP, and the VQC attack.

What changed from the previous version:

  * The member side is no longer capped at indices[0][:500] and no longer filtered
    to exclude class 4. Members and non-members are the target's full saved train /
    test indices (500/500 after the Part 1 retrain), class 4 present on both sides.
    The old asymmetry meant "looks like a 4" was itself evidence of non-membership,
    which is exactly the confound the 4-group framework removes.

  * The single accuracy number on the ~88-sample label-4 mix is replaced by the
    4-group table: member/non-4 (retain), member/class-4 (forget), non-member/non-4,
    non-member/class-4, reported for the original and the unlearned model.

  * The SVM's own 800/200 split is unchanged in spirit but is now stratified by
    group and shared with the other attacks (mia_common.ATTACK_SPLIT_SEED). It is
    the attack model's train/validation split and carries no membership semantics --
    both halves contain members and non-members.

  * The bandwidth x C sweep is scored by held-out AUC rather than accuracy. Accuracy
    on a balanced pool still rewards a degenerate majority-class SVM whenever the
    kernel concentrates; AUC does not.

  * The decision threshold is calibrated once on the original model and frozen, then
    applied unchanged to the unlearned model, so the before/after comparison is a
    change in the model rather than a change in the attack.

Cost: the fidelity kernel is O(N*M) circuit evaluations. Scoring 1000 rows against a
200-row reference set costs 200k circuits *per model*, plus the sweep. The script
prints its circuit budget before starting; lower --n-samples or --n-val if that is
too slow.

Usage:
    python qsvm_attack.py --original model_0324_original_5_0.1_8.pth \
                          --unlearned result/seed/model_MU_gradient_U8R1.pth
    python qsvm_attack.py --original orig.pth --unlearned unl.pth --n-samples 150 --n-val 150
"""

import argparse
import pickle

import numpy as np
import pennylane as qml
import torch
from pennylane.templates.embeddings import AngleEmbedding
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import mia_common as mc
import roc_utils

N_QUBITS = 8
WIRES = range(N_QUBITS)

# Embedding-bandwidth candidates for pi * tanh(bandwidth * z). Small bandwidth keeps
# angles close to linear in z; large bandwidth pushes more mass toward +-pi, which is
# more expressive per feature but concentrates the kernel faster. Swept, not guessed.
BANDWIDTHS = [0.1, 0.3, 1.0]
C_GRID = [0.1, 1.0, 10.0]

dev = qml.device("default.qubit", wires=N_QUBITS)


def feature_map(x, wires):
    """Compact two-axis angle embedding + one fixed entangling CNOT ring."""
    AngleEmbedding(x[:8], wires=wires, rotation="X")
    AngleEmbedding(x[8:16], wires=wires, rotation="Y")
    wires = list(wires)
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    qml.CNOT(wires=[wires[-1], wires[0]])


adjoint_feature_map = qml.adjoint(feature_map)


@qml.qnode(dev)
def kernel_circuit(x1, x2):
    feature_map(x1, wires=WIRES)
    adjoint_feature_map(x2, wires=WIRES)
    return qml.probs(wires=WIRES)


def quantum_kernel(x1, x2):
    return kernel_circuit(x1, x2)[0]


def prepare_inputs(x_raw, scaler, bandwidth):
    """Standardize with a pre-fit scaler, smoothly squash into rotation range, zero-pad 13 -> 16."""
    x_std = scaler.transform(x_raw)
    angles = np.pi * np.tanh(bandwidth * x_std)
    pad = np.zeros((angles.shape[0], 16 - angles.shape[1]))
    return np.hstack([angles, pad]).astype(np.float32)


def stratified_subsample(group, n_samples, rng):
    """Pick n_samples row positions, keeping the 4 groups' proportions.

    Uniform sampling of 200 from 800 would leave the forget group with whatever
    handful of class-4 members it happened to draw; the SVM needs to have actually
    seen that group to say anything about it.
    """
    n_total = len(group)
    if n_samples >= n_total:
        return np.arange(n_total)

    chosen = []
    for gid in np.unique(group):
        rows = np.flatnonzero(group == gid)
        take = max(1, int(round(len(rows) * n_samples / n_total)))
        take = min(take, len(rows))
        chosen.append(rng.choice(rows, size=take, replace=False))
    chosen = np.concatenate(chosen)

    # Rounding can overshoot or undershoot the requested budget; trim or top up.
    if len(chosen) > n_samples:
        chosen = rng.choice(chosen, size=n_samples, replace=False)
    elif len(chosen) < n_samples:
        remaining = np.setdiff1d(np.arange(n_total), chosen)
        extra = rng.choice(remaining, size=n_samples - len(chosen), replace=False)
        chosen = np.concatenate([chosen, extra])
    return np.sort(chosen)


def kernel_against(x_query, x_reference):
    return qml.kernels.kernel_matrix(x_query, x_reference, quantum_kernel)


def main():
    parser = argparse.ArgumentParser(description="QSVM-based MIA attack, 4-group evaluation")
    parser.add_argument("--original", default="model_0324_original_5_0.1_8.pth",
                        help="target checkpoint the attack model is fit against")
    parser.add_argument("--unlearned", default="result/seed/model_MU_gradient_U8R1.pth",
                        help="unlearned checkpoint to score with the frozen attack model")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="attack-train rows used as the quantum kernel's reference set (O(N^2))")
    parser.add_argument("--n-val", type=int, default=200,
                        help="held-out rows used to pick bandwidth and C")
    parser.add_argument("--n-attack-train", type=int, default=mc.N_ATTACK_TRAIN,
                        help="rows used to fit the attack model; the rest are held out. "
                             "Default 800 assumes a 1000-row pool -- raise it for a larger pool")
    parser.add_argument("--forget-class", type=int, default=mc.FORGET_CLASS)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--artifact", type=str, default="attack_model_qsvm.pkl")
    parser.add_argument("--scores-out", type=str, default="scores_qsvm.npz")
    parser.add_argument("--roc-out", type=str, default="qsvm_attack_roc.png",
                        help="path to save the QSVM attack's ROC plot; pass '' to skip")
    parser.add_argument("--roc-npz-out", type=str, default=None,
                        help="optional path to save raw fpr/tpr/thresholds (npz)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Pool + features
    # ------------------------------------------------------------------
    ckpt = torch.load(args.original, weights_only=False)
    mc.set_seed(int(getattr(ckpt["args"], "seed", 0)))
    model = mc.build_model(ckpt)
    print(f"[setup] target: {args.original}  (HQNN.ConvQNN, amplitude embedding)")

    spec = mc.build_pool_spec(ckpt, data_root=args.data_root, forget_class=args.forget_class,
                              n_attack_train=args.n_attack_train)
    mc.describe_pool(spec)

    X_original = mc.features_for(model, spec)

    fit_rows = np.flatnonzero(spec.attack_train_mask)
    heldout_rows = np.flatnonzero(spec.heldout_mask)

    # Scaler is fit on the attack-train rows only, then frozen -- the held-out rows
    # and the unlearned model's features must not influence the standardization.
    scaler = StandardScaler().fit(X_original[fit_rows])

    # Both subsamples are stratified by group. Slicing rows in index order would
    # take members only: the pool is built members-first, so heldout_rows[:n] is a
    # single-class set, which makes AUC undefined and lets a majority-class SVM
    # score 100% accuracy.
    rng = np.random.default_rng(mc.ATTACK_SPLIT_SEED)
    sub = fit_rows[stratified_subsample(spec.group[fit_rows], min(args.n_samples, len(fit_rows)), rng)]
    val_rows = heldout_rows[
        stratified_subsample(spec.group[heldout_rows], min(args.n_val, len(heldout_rows)), rng)
    ]

    y_sub = spec.membership[sub]
    y_val = spec.membership[val_rows]

    for tag, y in (("reference", y_sub), ("validation", y_val)):
        if len(np.unique(y)) < 2:
            raise RuntimeError(
                f"{tag} set contains only one membership class ({int(y.sum())} members / "
                f"{int((1 - y).sum())} non-members) -- AUC is undefined and the SVM has "
                "nothing to separate. Raise --n-samples/--n-val."
            )
    print(f"[setup] reference set: {int(y_sub.sum())} members / {int((1 - y_sub).sum())} non-members;  "
          f"validation set: {int(y_val.sum())} members / {int((1 - y_val).sum())} non-members")

    budget = len(BANDWIDTHS) * (len(sub) * (len(sub) - 1) // 2 + len(val_rows) * len(sub))
    budget += 2 * spec.n * len(sub)
    print(f"\n[setup] quantum-kernel budget: ~{budget:,} circuit evaluations "
          f"(reference set {len(sub)}, val {len(val_rows)}, scoring {spec.n} rows x 2 models)")

    # ------------------------------------------------------------------
    # Sweep bandwidth (expensive: a fresh kernel per candidate) and C (cheap:
    # a refit on an already-computed Gram matrix), scored by held-out AUC
    # ------------------------------------------------------------------
    best = {"auc": -1.0}
    print(f"\nSweeping bandwidth {BANDWIDTHS} x C {C_GRID} on {len(sub)} fit / {len(val_rows)} held-out rows ...")
    for bandwidth in BANDWIDTHS:
        x_sub_q = prepare_inputs(X_original[sub], scaler, bandwidth)
        x_val_q = prepare_inputs(X_original[val_rows], scaler, bandwidth)

        K_train = qml.kernels.square_kernel_matrix(x_sub_q, quantum_kernel,
                                                   assume_normalized_kernel=True)
        K_val = kernel_against(x_val_q, x_sub_q)

        for C in C_GRID:
            svc = SVC(kernel="precomputed", C=C).fit(K_train, y_sub)
            val_scores = svc.decision_function(K_val)
            val_auc = float(roc_auc_score(y_val, val_scores))
            val_pred = svc.predict(K_val)
            val_acc = float((val_pred == y_val).mean())
            degenerate = " [predicts one class]" if len(np.unique(val_pred)) < 2 else ""
            print(f"  bandwidth={bandwidth:.2f}  C={C:<6g}  val_auc={val_auc:.3f}  "
                  f"val_acc={val_acc * 100:.2f}%{degenerate}")

            # np.isfinite guards the comparison: a NaN would silently lose every
            # comparison and leave `best` unassigned.
            if np.isfinite(val_auc) and val_auc > best["auc"]:
                best = {"auc": val_auc, "acc": val_acc, "bandwidth": bandwidth, "C": C,
                        "svc": svc, "reference": x_sub_q}

    if "bandwidth" not in best:
        raise RuntimeError("no configuration produced a usable held-out AUC -- nothing to select from")

    print(f"Best config: bandwidth={best['bandwidth']}, C={best['C']}, "
          f"held-out AUC={best['auc']:.3f} (acc={best['acc'] * 100:.2f}%)")

    # ------------------------------------------------------------------
    # Score the original model's whole pool; calibrate and freeze the threshold
    # ------------------------------------------------------------------
    x_all_q = prepare_inputs(X_original, scaler, best["bandwidth"])
    K_all = kernel_against(x_all_q, best["reference"])
    scores_original = best["svc"].decision_function(K_all)

    # Calibrated on the attack-train rows only. Calibrating on all 1000 would fold
    # the held-out rows into threshold selection and make the held-out row of the
    # headline table no longer held out.
    threshold = mc.calibrate_threshold(scores_original, spec.membership, spec.attack_train_mask)

    # ------------------------------------------------------------------
    # Apply the frozen attack model + threshold to the unlearned checkpoint
    # ------------------------------------------------------------------
    scores_unlearned = None
    try:
        _, unlearned_model = mc.load_weights_into(args.unlearned, ckpt)
    except FileNotFoundError:
        print(f"\n[warn] unlearned checkpoint not found: {args.unlearned} -- reporting the original model only")
    else:
        print(f"[setup] unlearned: {args.unlearned}")
        X_unlearned = mc.features_for(unlearned_model, spec)
        x_unl_q = prepare_inputs(X_unlearned, scaler, best["bandwidth"])
        K_unl = kernel_against(x_unl_q, best["reference"])
        scores_unlearned = best["svc"].decision_function(K_unl)

    mc.report(spec, scores_original, scores_unlearned, threshold, "QSVM attack")

    # --- ROC curve for the QSVM attack (original model) ---------------------
    roc_data = roc_utils.compute_roc(scores_original, spec.membership, higher_is_member=True)
    if args.roc_out:
        roc_utils.plot_roc(roc_data, args.roc_out,
                            title="QSVM attack ROC (original model)", label="QSVM attack")
    if args.roc_npz_out:
        roc_utils.save_roc_data({"qsvm": roc_data}, args.roc_npz_out)

    with open(args.artifact, "wb") as f:
        pickle.dump({
            "svc": best["svc"],
            "scaler": scaler,
            "reference_inputs": best["reference"],
            "bandwidth": best["bandwidth"],
            "C": best["C"],
            "threshold": threshold,
            "heldout_auc": best["auc"],
            "target_checkpoint": args.original,
            "attack_split_seed": mc.ATTACK_SPLIT_SEED,
            "forget_class": args.forget_class,
        }, f)
    print(f"\n[save] QSVM attack model -> {args.artifact}")
    mc.save_scores(args.scores_out, spec, scores_original, scores_unlearned, threshold, "QSVM")


if __name__ == "__main__":
    main()
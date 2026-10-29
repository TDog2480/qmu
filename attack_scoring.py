"""
Shared member/non-member pool, feature extraction, and 4-group reporting for the
MIA family (threshold attacks, MLP, VQC, QSVM).

Part 4 of the evaluation plan requires all four attack families to be scored on
the *same* balanced pool with the *same* 4-group breakdown, so their numbers are
directly comparable. Everything that defines the pool, the groups, or the
attack-model split lives here, so the attack scripts cannot silently drift apart
the way shadow_attack_train/eval's build_attack_features() could.

Two splits are in play and must not be conflated:

  * membership split -- member (target's train indices) vs non-member (target's
    test indices). This is the *label* the attack is trying to predict, and it
    is what the 4-group table is built from.

  * attack-model split -- 800 rows to fit the attack model / calibrate a
    threshold, 200 rows held out. Both halves contain members and non-members
    and both contain class-4 rows. This is ordinary supervised-learning hygiene
    for the attack model and carries no membership semantics.

The 4 groups (Part 3):

    group                        original     unlearned (success)   measures
    member, non-4 (retain)       member       member                utility
    member, class-4 (forget)     member       non-member            forgetting
    non-member, non-4            non-member   non-member            baseline FPR
    non-member, class-4          non-member   non-member            class-matched FPR

Success criterion for forgetting is the forget row's unlearned member-rate
approaching the *class-matched* non-member rate, not the global FPR. Compare the
two non-member rows first: if they diverge, class 4 is atypical for this model
and the class-matched number is the honest target.

The retain row measures utility, not privacy. It staying member-like only means
unlearning did not break the model.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

FORGET_CLASS = 4
N_ATTACK_TRAIN = 800

# Fixed and shared across every attack script so the 800/200 split is byte-identical
# between the MLP, VQC and QSVM runs. Deliberately independent of the target
# checkpoint's own seed -- the split must not move when the target does.
ATTACK_SPLIT_SEED = 12345

FEATURE_NAMES = [f"prob_{i}" for i in range(10)] + ["max_prob", "correct", "loss"]
FEATURE_DIM = len(FEATURE_NAMES)

# (key, human-readable label, is_member, is_forget_class) -- order defines group ids.
GROUPS = (
    ("member_retain", "member, non-4 (retain)", True, False),
    ("member_forget", "member, class-4 (forget)", True, True),
    ("nonmember_retain", "non-member, non-4", False, False),
    ("nonmember_forget", "non-member, class-4", False, True),
)
GROUP_ID = {key: i for i, (key, _label, _m, _f) in enumerate(GROUPS)}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_mnist(data_root: str = "./data"):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    testset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    return trainset, testset


def build_model(ckpt):
    """
    Instantiate HQNN.ConvQNN and load this checkpoint's weights.

    Every attack in this evaluation targets the base HQNN.py model (amplitude
    embedding, 8 qubits), so the architecture is fixed rather than sniffed from
    the state_dict. A checkpoint from a different encoding will fail loudly here
    on a shape mismatch, which is the wanted behaviour -- silently loading the
    wrong ConvQNN would produce a full 4-group table of meaningless numbers.
    """
    from HQNN import ConvQNN

    model = ConvQNN(ckpt["args"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_weights_into(ckpt_path, reference_ckpt):
    """Build a *second* HQNN.ConvQNN instance and load another checkpoint's weights
    into it, leaving the original model untouched.

    The reference checkpoint supplies the constructor args, so the unlearned model
    is built exactly like the original it is being compared against."""
    other = torch.load(ckpt_path, weights_only=False)
    model = build_model(reference_ckpt)
    model.load_state_dict(other["model_state_dict"])
    model.eval()
    return other, model


# ---------------------------------------------------------------------------
# Pool definition
# ---------------------------------------------------------------------------

@dataclass
class PoolSpec:
    """The member/non-member pool and its group + attack-split structure.

    Model-independent by construction: labels come from MNIST and membership comes
    from the target's saved indices, so the identical spec is reused to score the
    original model, the unlearned model, and a from-scratch reference model.
    """
    train_indices: List[int]
    test_indices: List[int]
    labels: np.ndarray            # (N,) true MNIST class
    membership: np.ndarray        # (N,) 1 = member, 0 = non-member
    group: np.ndarray             # (N,) index into GROUPS
    attack_train_mask: np.ndarray  # (N,) bool: row is in the attack model's 800
    forget_class: int
    trainset: object
    testset: object

    @property
    def n(self) -> int:
        return len(self.labels)

    @property
    def heldout_mask(self) -> np.ndarray:
        return ~self.attack_train_mask


def build_pool_spec(ckpt, data_root: str = "./data", forget_class: int = FORGET_CLASS,
                    n_attack_train: int = N_ATTACK_TRAIN,
                    split_seed: int = ATTACK_SPLIT_SEED) -> PoolSpec:
    """
    Build the full pool from the target checkpoint's own saved indices.

    Note there is no [:500] cap and no `!= forget_class` filter on the member
    side. Both were in the original attack scripts and together they created the
    confound this framework exists to remove: class-4 rows appeared almost only
    among non-members, so "looks like a 4" was a free non-member signal.
    """
    trainset, testset = load_mnist(data_root)

    train_indices = [int(i) for i in ckpt["indices"][0]]
    test_indices = [int(i) for i in ckpt["indices"][1]]

    member_labels = np.array([int(trainset.targets[i]) for i in train_indices])
    nonmember_labels = np.array([int(testset.targets[i]) for i in test_indices])

    labels = np.concatenate([member_labels, nonmember_labels])
    membership = np.concatenate([
        np.ones(len(member_labels), dtype=int),
        np.zeros(len(nonmember_labels), dtype=int),
    ])
    is_forget = labels == forget_class

    group = np.where(
        membership == 1,
        np.where(is_forget, GROUP_ID["member_forget"], GROUP_ID["member_retain"]),
        np.where(is_forget, GROUP_ID["nonmember_forget"], GROUP_ID["nonmember_retain"]),
    )

    if len(member_labels) != len(nonmember_labels):
        print(f"[warn] pool is unbalanced: {len(member_labels)} members vs {len(nonmember_labels)} "
              "non-members. Part 1 assumes -n_train 500 -n_test 500; accuracy will be "
              "misleading, read the AUC and the per-group rates instead.")

    n_total = len(labels)
    n_tr = int(min(n_attack_train, n_total - 1))
    # Stratify on the 4 groups, not just membership, so the ~50 class-4 members
    # are split proportionally instead of landing mostly on one side by luck.
    tr_rows, _ = train_test_split(
        np.arange(n_total), train_size=n_tr, stratify=group, random_state=split_seed
    )
    attack_train_mask = np.zeros(n_total, dtype=bool)
    attack_train_mask[tr_rows] = True

    return PoolSpec(
        train_indices=train_indices,
        test_indices=test_indices,
        labels=labels,
        membership=membership,
        group=group,
        attack_train_mask=attack_train_mask,
        forget_class=forget_class,
        trainset=trainset,
        testset=testset,
    )


def describe_pool(spec: PoolSpec) -> None:
    print(f"\nPool: {spec.n} rows  ({int(spec.membership.sum())} members / "
          f"{int((1 - spec.membership).sum())} non-members), forget class = {spec.forget_class}")
    print(f"Attack-model split: {int(spec.attack_train_mask.sum())} fit / "
          f"{int(spec.heldout_mask.sum())} held out  (seed {ATTACK_SPLIT_SEED}, stratified by group)")
    print(f"{'group':<28}{'n':>6}{'fit':>7}{'held-out':>10}")
    for gid, (_key, label, _m, _f) in enumerate(GROUPS):
        g = spec.group == gid
        print(f"{label:<28}{int(g.sum()):>6}{int((g & spec.attack_train_mask).sum()):>7}"
              f"{int((g & spec.heldout_mask).sum()):>10}")


# ---------------------------------------------------------------------------
# Feature extraction -- the same 13-dim vector MIA.py uses
# ---------------------------------------------------------------------------

def _features_and_labels(model, dataset, indices: Sequence[int], batch_size: int):
    model.eval()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            loss = F.cross_entropy(logits, y, reduction="none")
            max_prob, pred = probs.max(dim=1)
            correct = (pred == y).to(probs.dtype)
            feats.append(
                torch.cat([probs, max_prob[:, None], correct[:, None], loss[:, None]], dim=1).cpu().numpy()
            )
            labels.append(y.cpu().numpy())
    return np.vstack(feats).astype(np.float64), np.concatenate(labels)


def features_for(model, spec: PoolSpec, batch_size: int = 64) -> np.ndarray:
    """(N, 13) features for every pool row, in the spec's row order (members first)."""
    Xm, ym = _features_and_labels(model, spec.trainset, spec.train_indices, batch_size)
    Xn, yn = _features_and_labels(model, spec.testset, spec.test_indices, batch_size)
    X = np.vstack([Xm, Xn])
    # Guard against row-order drift between the spec and the feature matrix; a silent
    # misalignment here would scramble every group in the table below.
    assert np.array_equal(np.concatenate([ym, yn]), spec.labels), \
        "feature row order does not match the pool spec"
    X = np.nan_to_num(X, nan=0.0, posinf=50.0, neginf=-50.0)
    return X


# ---------------------------------------------------------------------------
# Thresholds and metrics
# ---------------------------------------------------------------------------

def calibrate_threshold(scores: np.ndarray, membership: np.ndarray,
                        mask: Optional[np.ndarray] = None) -> float:
    """Threshold maximizing balanced accuracy, swept over the ROC.

    Calibrated once on the original model and then frozen (Part 2). Selection is
    in-sample over whatever rows `mask` covers, so the resulting accuracy/FPR on
    those same rows is optimistic -- the before/after forget comparison is what
    the frozen threshold is really for.
    """
    sel = np.ones(len(scores), dtype=bool) if mask is None else mask
    fpr, tpr, thresholds = roc_curve(membership[sel], scores[sel])
    best = int(np.argmax((tpr + (1 - fpr)) / 2))
    thr = float(thresholds[best])
    if not np.isfinite(thr):  # roc_curve's first threshold is +inf
        finite = thresholds[np.isfinite(thresholds)]
        thr = float(finite.max()) if len(finite) else 0.0
    return thr


def membership_auc(scores: np.ndarray, spec: PoolSpec,
                   mask: Optional[np.ndarray] = None) -> float:
    sel = np.ones(len(scores), dtype=bool) if mask is None else mask
    y = spec.membership[sel]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores[sel]))


def balanced_accuracy(scores: np.ndarray, spec: PoolSpec, threshold: float,
                      mask: Optional[np.ndarray] = None) -> float:
    sel = np.ones(len(scores), dtype=bool) if mask is None else mask
    pred = (scores[sel] >= threshold).astype(int)
    y = spec.membership[sel]
    tpr = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    tnr = float((1 - pred[y == 0]).mean()) if (y == 0).any() else float("nan")
    return (tpr + tnr) / 2


def group_member_rates(spec: PoolSpec, scores: np.ndarray, threshold: float,
                       mask: Optional[np.ndarray] = None):
    """{group_key: (n, member_rate)} -- fraction of the group the attack calls "member"."""
    sel = np.ones(len(scores), dtype=bool) if mask is None else mask
    out = {}
    for gid, (key, _label, _m, _f) in enumerate(GROUPS):
        g = sel & (spec.group == gid)
        n = int(g.sum())
        rate = float((scores[g] >= threshold).mean()) if n else float("nan")
        out[key] = (n, rate)
    return out


# ---------------------------------------------------------------------------
# The 4-group report (Part 3)
# ---------------------------------------------------------------------------

def print_group_table(spec: PoolSpec, scores_original: np.ndarray,
                      scores_unlearned: Optional[np.ndarray], threshold: float,
                      mask: Optional[np.ndarray] = None, title: str = "") -> None:
    before = group_member_rates(spec, scores_original, threshold, mask)
    after = group_member_rates(spec, scores_unlearned, threshold, mask) if scores_unlearned is not None else None

    print(f"\n{'=' * 78}")
    print(title or "Member rate by group")
    print(f"{'=' * 78}")
    if after is not None:
        print(f"{'group':<28}{'n':>5}{'original':>11}{'unlearned':>11}{'delta':>9}   measures")
    else:
        print(f"{'group':<28}{'n':>5}{'original':>11}   measures")
    print("-" * 78)

    measures = ["utility / stability", "forgetting", "baseline FPR", "class-matched FPR"]
    for (key, label, _m, _f), what in zip(GROUPS, measures):
        n, rate_b = before[key]
        if after is not None:
            _, rate_a = after[key]
            delta = rate_a - rate_b
            print(f"{label:<28}{n:>5}{rate_b:>11.3f}{rate_a:>11.3f}{delta:>+9.3f}   {what}")
        else:
            print(f"{label:<28}{n:>5}{rate_b:>11.3f}   {what}")
    print("-" * 78)

    # Is class 4 atypical for this model? Compare the two non-member rows first.
    nm_other = before["nonmember_retain"][1]
    nm_forget = before["nonmember_forget"][1]
    print(f"non-member rows (original): non-4 {nm_other:.3f} vs class-4 {nm_forget:.3f} "
          f"(diff {nm_forget - nm_other:+.3f})")
    if abs(nm_forget - nm_other) > 0.05:
        print("  -> class 4 is atypical here; the class-matched rate, not the global FPR, "
              "is the honest forgetting target")

    if after is not None:
        forget_after = after["member_forget"][1]
        target = after["nonmember_forget"][1]
        print(f"forgetting check (unlearned): forget member-rate {forget_after:.3f} vs "
              f"class-matched non-member rate {target:.3f} (gap {forget_after - target:+.3f})")
        print(f"utility check (unlearned): retain member-rate {after['member_retain'][1]:.3f} "
              f"(was {before['member_retain'][1]:.3f}) -- utility, not privacy")


def print_headline(spec: PoolSpec, scores_original: np.ndarray,
                   scores_unlearned: Optional[np.ndarray], threshold: float,
                   attack_name: str) -> None:
    """AUC + balanced accuracy on the full pool and on the held-out rows only."""
    print(f"\n{attack_name}: threshold = {threshold:.4f} (frozen, calibrated on the original model)")
    print(f"{'rows':<22}{'AUC':>8}{'Bal.Acc':>10}")
    print("-" * 40)
    for tag, mask in (("full pool", None), ("held-out only", spec.heldout_mask)):
        print(f"{tag:<22}{membership_auc(scores_original, spec, mask):>8.3f}"
              f"{balanced_accuracy(scores_original, spec, threshold, mask):>10.3f}")
    print("Held-out is the honest generalization number; the full-pool row includes the "
          "800 rows the attack model was fit on.")
    if scores_unlearned is not None:
        print(f"\nSame attack applied to the unlearned model (full pool): "
              f"AUC {membership_auc(scores_unlearned, spec, None):.3f}, "
              f"Bal.Acc {balanced_accuracy(scores_unlearned, spec, threshold, None):.3f}")


def report(spec: PoolSpec, scores_original: np.ndarray, scores_unlearned: Optional[np.ndarray],
           threshold: float, attack_name: str) -> None:
    """Full Part 3 / Part 4 report for one attack family."""
    print_headline(spec, scores_original, scores_unlearned, threshold, attack_name)
    print_group_table(spec, scores_original, scores_unlearned, threshold, mask=None,
                      title=f"{attack_name} -- member rate by group (full 1000-row pool)")
    print_group_table(spec, scores_original, scores_unlearned, threshold, mask=spec.heldout_mask,
                      title=f"{attack_name} -- member rate by group (held-out rows only; "
                            "small n, wide error bars)")


def save_scores(path, spec: PoolSpec, scores_original, scores_unlearned, threshold, attack_name):
    """Per-row scores, so several attacks' 4-group tables can be merged later."""
    np.savez(
        path,
        attack=attack_name,
        threshold=threshold,
        labels=spec.labels,
        membership=spec.membership,
        group=spec.group,
        attack_train_mask=spec.attack_train_mask,
        scores_original=scores_original,
        scores_unlearned=(scores_unlearned if scores_unlearned is not None
                          else np.array([], dtype=float)),
    )
    print(f"[save] per-row membership scores -> {path}")
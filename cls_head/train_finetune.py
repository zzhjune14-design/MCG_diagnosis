import csv
import os
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from finetune_model import HybridFineTuneModel, MCGFineTuneDataset, MLPSpectralModel, MLPWaveformModel


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "sum":
            return focal_loss.sum()
        if self.reduction == "none":
            return focal_loss
        return focal_loss.mean()


EXPERIMENT_CONFIGS = {
    "BASELINE": {
        "warmup_epochs": 5,
        "joint_epochs": 30,
        "warmup_lr": 1e-3,
        "encoder_lr": 5e-5,
        "head_lr": 5e-4,
        "loss_name": "focal",
        "use_mixup": True,
        "mixup_stop_last_n": 5,
        "use_patient_sampler": False,
        "use_patient_selection": False,
        "early_stopping_patience": None,
        "drop_last": True,
    },
    "A": {
        "warmup_epochs": 5,
        "joint_epochs": 30,
        "warmup_lr": 1e-3,
        "encoder_lr": 5e-5,
        "head_lr": 5e-4,
        "loss_name": "focal",
        "use_mixup": True,
        "mixup_stop_last_n": 5,
        "use_patient_sampler": False,
        "use_patient_selection": True,
        "early_stopping_patience": 5,
        "drop_last": True,
    },
    "B": {
        "warmup_epochs": 5,
        "joint_epochs": 12,
        "warmup_lr": 1e-3,
        "encoder_lr": 1e-5,
        "head_lr": 2e-4,
        "loss_name": "focal",
        "use_mixup": False,
        "mixup_stop_last_n": 3,
        "use_patient_sampler": False,
        "use_patient_selection": True,
        "early_stopping_patience": 5,
        "drop_last": True,
    },
    "C": {
        "warmup_epochs": 5,
        "joint_epochs": 12,
        "warmup_lr": 1e-3,
        "encoder_lr": 1e-5,
        "head_lr": 2e-4,
        "loss_name": "bce",
        "use_mixup": False,
        "mixup_stop_last_n": 3,
        "use_patient_sampler": True,
        "use_patient_selection": True,
        "early_stopping_patience": 5,
        "drop_last": True,
    },
}


def compute_metrics(labels, probs):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.nan_to_num(np.asarray(probs, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    if labels.size == 0:
        return {"acc": 0.0, "sens": 0.0, "spec": 0.0, "f1": 0.0, "auc": 0.5}

    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(labels, probs) if np.unique(labels).size > 1 else 0.5

    return {"acc": acc, "sens": sens, "spec": spec, "f1": f1, "auc": auc}


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aggregate_patient_predictions(subject_ids, labels, probs):
    grouped = defaultdict(lambda: {"label": None, "probs": []})

    for subject_id, label, prob in zip(subject_ids, labels, probs):
        subject_id = int(subject_id)
        label = int(label)
        grouped[subject_id]["probs"].append(float(prob))
        if grouped[subject_id]["label"] is None:
            grouped[subject_id]["label"] = label

    patient_ids = []
    patient_labels = []
    patient_probs = []
    for subject_id in sorted(grouped):
        patient_ids.append(subject_id)
        patient_labels.append(grouped[subject_id]["label"])
        patient_probs.append(float(np.mean(grouped[subject_id]["probs"])))

    return np.array(patient_ids), np.array(patient_labels), np.array(patient_probs)


def parse_priority_weight_map(priority_weights_str):
    if not priority_weights_str:
        return {}

    priority_weight_map = {}
    for item in priority_weights_str.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        priority, weight = item.split("=", 1)
        priority = priority.strip().lower()
        weight = weight.strip()
        if not priority or not weight:
            continue
        priority_weight_map[priority] = float(weight)

    return priority_weight_map


def load_subject_weight_map(review_csv, default_weight, priority_filter, priority_weight_map=None):
    if review_csv is None or not os.path.exists(review_csv):
        return {}

    review_df = pd.read_csv(review_csv)
    review_df = review_df[review_df["xinshuai"].notna()].copy() if "xinshuai" in review_df.columns else review_df.copy()
    if "review_flag" not in review_df.columns:
        return {}

    flagged_df = review_df[review_df["review_flag"] == 1].copy()
    if priority_filter:
        allowed = {item.strip().lower() for item in priority_filter.split(",") if item.strip()}
        if "review_priority" in flagged_df.columns:
            flagged_df = flagged_df[flagged_df["review_priority"].astype(str).str.lower().isin(allowed)]

    priority_weight_map = priority_weight_map or {}
    subject_weight_map = {}
    for row in flagged_df.itertuples():
        subject_id = int(row.subject)
        priority = str(getattr(row, "review_priority", "")).strip().lower()
        subject_weight_map[subject_id] = float(priority_weight_map.get(priority, default_weight))

    print(f"Loaded {len(subject_weight_map)} downweighted subjects from {review_csv}")
    return subject_weight_map


def create_patient_sampler(dataset, train_indices, subject_weight_map=None):
    subject_counts = defaultdict(int)
    subject_labels = {}
    subject_weight_map = subject_weight_map or {}

    for idx in train_indices:
        subject_id = int(dataset.subject_ids[idx])
        subject_counts[subject_id] += 1
        subject_labels[subject_id] = int(dataset.labels[idx])

    pos_subjects = sum(1 for label in subject_labels.values() if label == 1)
    neg_subjects = sum(1 for label in subject_labels.values() if label == 0)
    class_weights = {
        0: 1.0 / max(1, neg_subjects),
        1: 1.0 / max(1, pos_subjects),
    }

    sample_weights = []
    for idx in train_indices:
        subject_id = int(dataset.subject_ids[idx])
        label = int(dataset.labels[idx])
        sample_weight = class_weights[label] / max(1, subject_counts[subject_id])
        sample_weight *= subject_weight_map.get(subject_id, 1.0)
        sample_weights.append(sample_weight)

    return WeightedRandomSampler(sample_weights, num_samples=len(train_indices), replacement=True)


def create_beat_sampler(dataset, train_indices, subject_weight_map=None):
    subject_weight_map = subject_weight_map or {}
    labels = [int(dataset.labels[idx]) for idx in train_indices]
    pos_len = sum(labels)
    neg_len = len(labels) - pos_len
    class_weights = {
        0: 1.0 / max(1, neg_len),
        1: 1.0 / max(1, pos_len),
    }
    sample_weights = []
    for idx, label in zip(train_indices, labels):
        subject_id = int(dataset.subject_ids[idx])
        sample_weight = class_weights[int(label)] * subject_weight_map.get(subject_id, 1.0)
        sample_weights.append(sample_weight)
    return WeightedRandomSampler(sample_weights, num_samples=len(train_indices), replacement=True)


def build_criterion(config, labels, device):
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    if config["loss_name"] == "bce":
        pos_weight = torch.tensor([neg_count / max(1, pos_count)], dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    alpha_weight = neg_count / max(1, pos_count + neg_count)
    return FocalLoss(alpha=alpha_weight, gamma=2.0)


def maybe_apply_mixup(x, y, phase, epoch_idx, epochs, config, device):
    if not config["use_mixup"] or phase != "Joint":
        return x, y, None, 1.0

    if epoch_idx >= max(0, epochs - config["mixup_stop_last_n"]):
        return x, y, None, 1.0

    lam = np.random.beta(0.2, 0.2)
    index = torch.randperm(x.size(0), device=device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def run_phase(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epochs,
    device,
    fold,
    phase,
    log_file,
    save_dir,
    task,
    config,
    best_selection_auc,
    seed,
    model_variant,
):
    patience = config["early_stopping_patience"] if phase == "Joint" else None
    stale_epochs = 0

    for epoch_idx in range(epochs):
        model.train()
        train_loss = 0.0
        valid_batches = 0

        for batch in tqdm(train_loader, desc=f"Fold {fold} {phase} Ep {epoch_idx + 1}/{epochs} [Train]"):
            x, y = batch[0].to(device), batch[1].to(device)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            mixed_x, y_a, y_b, lam = maybe_apply_mixup(x, y, phase, epoch_idx, epochs, config, device)

            optimizer.zero_grad()
            logits = model(mixed_x)
            if y_b is None:
                loss = criterion(logits, y_a)
            else:
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss += loss.item()
            valid_batches += 1

        train_loss = train_loss / max(1, valid_batches)

        model.eval()
        val_loss = 0.0
        beat_probs = []
        beat_labels = []
        beat_subjects = []

        with torch.no_grad():
            for x, y, subject_id in tqdm(val_loader, desc=f"Fold {fold} {phase} Ep {epoch_idx + 1}/{epochs} [Val]"):
                x = x.to(device)
                y = y.to(device)
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                beat_probs.extend(probs.tolist())
                beat_labels.extend(y.cpu().numpy().tolist())
                beat_subjects.extend(subject_id.cpu().numpy().tolist())

        val_loss = val_loss / max(1, len(val_loader))
        beat_metrics = compute_metrics(beat_labels, beat_probs)
        patient_ids, patient_labels, patient_probs = aggregate_patient_predictions(beat_subjects, beat_labels, beat_probs)
        patient_metrics = compute_metrics(patient_labels, patient_probs)

        print(
            f"[Fold {fold}] {phase} Ep {epoch_idx + 1}: "
            f"loss={val_loss:.4f} | beat_auc={beat_metrics['auc']:.4f} | patient_auc={patient_metrics['auc']:.4f}"
        )

        with open(log_file, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                config["name"],
                model_variant,
                fold,
                phase,
                epoch_idx + 1,
                round(train_loss, 4),
                round(val_loss, 4),
                round(beat_metrics["acc"], 4),
                round(beat_metrics["sens"], 4),
                round(beat_metrics["spec"], 4),
                round(beat_metrics["f1"], 4),
                round(beat_metrics["auc"], 4),
                round(patient_metrics["acc"], 4),
                round(patient_metrics["sens"], 4),
                round(patient_metrics["spec"], 4),
                round(patient_metrics["f1"], 4),
                round(patient_metrics["auc"], 4),
            ])

        selection_auc = patient_metrics["auc"] if config["use_patient_selection"] else beat_metrics["auc"]
        if selection_auc > best_selection_auc:
            best_selection_auc = selection_auc
            stale_epochs = 0
            save_path = os.path.join(save_dir, f"best_model_{task}_fold{fold}.pth")
            torch.save(model.state_dict(), save_path)
            prediction_path = os.path.join(save_dir, f"seed_{seed}_fold_{fold}_best_predictions.csv")
            prediction_df = pd.DataFrame(
                {
                    "Seed": seed,
                    "Fold": fold,
                    "Phase": phase,
                    "Epoch": epoch_idx + 1,
                    "Subject": patient_ids,
                    "Label": patient_labels,
                    "Prob": patient_probs,
                }
            )
            prediction_df.to_csv(prediction_path, index=False)
            print(
                f"Saved best checkpoint for fold {fold}: "
                f"selection_auc={selection_auc:.4f}, patient_auc={patient_metrics['auc']:.4f}, beat_auc={beat_metrics['auc']:.4f}"
            )
        elif patience is not None:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping fold {fold} {phase} at epoch {epoch_idx + 1} (patience={patience}).")
                break

    return best_selection_auc


def get_experiment_config(task):
    experiment_name = os.environ.get("MCG_EXPERIMENT")
    if experiment_name is None:
        experiment_name = "C" if task == "xinshuai" else "BASELINE"

    experiment_name = experiment_name.upper()
    if experiment_name not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unsupported experiment '{experiment_name}'. Expected one of {sorted(EXPERIMENT_CONFIGS)}")

    config = dict(EXPERIMENT_CONFIGS[experiment_name])
    config["name"] = experiment_name
    return config


def apply_env_overrides(config):
    overrides = {
        "warmup_epochs": ("MCG_WARMUP_EPOCHS", int),
        "joint_epochs": ("MCG_JOINT_EPOCHS", int),
        "warmup_lr": ("MCG_WARMUP_LR", float),
        "encoder_lr": ("MCG_ENCODER_LR", float),
        "head_lr": ("MCG_HEAD_LR", float),
        "mixup_stop_last_n": ("MCG_MIXUP_STOP_LAST_N", int),
        "early_stopping_patience": ("MCG_EARLY_STOPPING_PATIENCE", int),
    }
    for key, (env_name, caster) in overrides.items():
        env_value = os.environ.get(env_name)
        if env_value is not None:
            config[key] = caster(env_value)

    env_loss = os.environ.get("MCG_LOSS_NAME")
    if env_loss is not None:
        config["loss_name"] = env_loss.lower()

    env_mixup = os.environ.get("MCG_USE_MIXUP")
    if env_mixup is not None:
        config["use_mixup"] = env_mixup.lower() in {"1", "true", "yes", "y"}

    env_sampler = os.environ.get("MCG_USE_PATIENT_SAMPLER")
    if env_sampler is not None:
        config["use_patient_sampler"] = env_sampler.lower() in {"1", "true", "yes", "y"}

    env_selection = os.environ.get("MCG_USE_PATIENT_SELECTION")
    if env_selection is not None:
        config["use_patient_selection"] = env_selection.lower() in {"1", "true", "yes", "y"}

    return config


def init_log(log_file):
    with open(log_file, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "Experiment",
            "ModelVariant",
            "Fold",
            "Phase",
            "Epoch",
            "Train_Loss",
            "Val_Loss",
            "Beat_Val_Acc",
            "Beat_Val_Sens",
            "Beat_Val_Spec",
            "Beat_Val_F1",
            "Beat_Val_AUC",
            "Patient_Val_Acc",
            "Patient_Val_Sens",
            "Patient_Val_Spec",
            "Patient_Val_F1",
            "Patient_Val_AUC",
        ])


def build_model(model_variant, encoder_weight_path, device):
    model_variant = model_variant.lower()
    if model_variant == "hybrid":
        return HybridFineTuneModel(encoder_weight_path=encoder_weight_path).to(device)
    if model_variant == "mlp":
        return MLPWaveformModel().to(device)
    if model_variant == "spectral_mlp":
        return MLPSpectralModel().to(device)
    raise ValueError(f"Unsupported model variant '{model_variant}'. Expected one of ['hybrid', 'mlp', 'spectral_mlp']")


def build_optimizers(model, config, model_variant):
    model_variant = model_variant.lower()

    if model_variant == "hybrid":
        classifier_params = [param for name, param in model.named_parameters() if not name.startswith("encoder.")]
        warmup_optimizer = optim.Adam(classifier_params, lr=config["warmup_lr"], weight_decay=1e-4)
        joint_optimizer = optim.Adam(
            [
                {"params": model.encoder.parameters(), "lr": config["encoder_lr"]},
                {"params": classifier_params, "lr": config["head_lr"]},
            ],
            weight_decay=1e-4,
        )
        return warmup_optimizer, joint_optimizer

    classifier_params = list(model.parameters())
    warmup_optimizer = optim.Adam(classifier_params, lr=config["warmup_lr"], weight_decay=1e-4)
    joint_optimizer = optim.Adam(classifier_params, lr=config["head_lr"], weight_decay=1e-4)
    return warmup_optimizer, joint_optimizer


def train_finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = os.environ.get("MCG_TASK", "Ischemia")
    model_variant = os.environ.get("MCG_MODEL_VARIANT", "hybrid").lower()
    seed = int(os.environ.get("MCG_SEED", "1"))
    set_random_seed(seed)
    config = apply_env_overrides(get_experiment_config(task))
    batch_size = int(os.environ.get("MCG_BATCH_SIZE", "128"))
    n_folds = int(os.environ.get("MCG_N_FOLDS", "5"))
    save_dir = os.environ.get("MCG_SAVE_DIR", "./finetune_checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    label_csv = os.environ.get("MCG_LABEL_CSV", r"E:\Pythonpro\MCG_quexue_xinshuai\label.csv")
    data_dir = os.environ.get("MCG_DATA_DIR", r"E:\Pythonpro\MCG_quexue_xinshuai\all_data_folder")
    encoder_weight_path = os.environ.get(
        "MCG_ENCODER_WEIGHT_PATH",
        r"D:\New_python_project\MCG_diagnosis\cls_head\finetune_checkpoints\best_model_Ischemia_fold5.pth",
    )
    review_csv = os.environ.get("MCG_REVIEW_CSV")
    review_downweight = float(os.environ.get("MCG_REVIEW_DOWNWEIGHT", "1.0"))
    review_priority_filter = os.environ.get("MCG_REVIEW_PRIORITY_FILTER", "")
    review_priority_weights = os.environ.get("MCG_REVIEW_PRIORITY_WEIGHTS", "")
    priority_weight_map = parse_priority_weight_map(review_priority_weights)
    subject_weight_map = load_subject_weight_map(
        review_csv,
        review_downweight,
        review_priority_filter,
        priority_weight_map=priority_weight_map,
    )

    print(f"Running task={task} experiment={config['name']} model={model_variant} seed={seed} on device={device}")
    df = pd.read_csv(label_csv)
    if task == "xinshuai":
        df = df.dropna(subset=["xinshuai"])

    subjects = df["subject"].astype(int).values
    y_patient = (df[task].values == 1.0).astype(int)

    full_dataset = MCGFineTuneDataset(df, data_dir, task=task, return_subject_id=True)
    log_file = os.path.join(save_dir, f"{task}_training_log.csv")
    init_log(log_file)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_best_aucs = []

    for fold_idx, (train_subject_idx, val_subject_idx) in enumerate(skf.split(subjects, y_patient), start=1):
        train_subjects = set(subjects[train_subject_idx])
        val_subjects = set(subjects[val_subject_idx])
        train_indices = [idx for idx, subject_id in enumerate(full_dataset.subject_ids) if subject_id in train_subjects]
        val_indices = [idx for idx, subject_id in enumerate(full_dataset.subject_ids) if subject_id in val_subjects]

        print(f"\n{'=' * 24} Fold {fold_idx} {'=' * 24}")
        print(f"Train beats: {len(train_indices)} | Val beats: {len(val_indices)}")
        print(f"Train patients: {len(train_subjects)} | Val patients: {len(val_subjects)}")

        sampler = (
            create_patient_sampler(full_dataset, train_indices, subject_weight_map=subject_weight_map)
            if config["use_patient_sampler"]
            else create_beat_sampler(full_dataset, train_indices, subject_weight_map=subject_weight_map)
        )
        train_loader = DataLoader(
            Subset(full_dataset, train_indices),
            batch_size=batch_size,
            sampler=sampler,
            drop_last=config["drop_last"],
        )
        val_loader = DataLoader(
            Subset(full_dataset, val_indices),
            batch_size=batch_size,
            shuffle=False,
        )

        model = build_model(model_variant, encoder_weight_path, device)
        best_selection_auc = 0.0

        fold_train_labels = [full_dataset.labels[idx] for idx in train_indices]
        criterion = build_criterion(config, fold_train_labels, device)

        model.freeze_encoder()
        warmup_optimizer, joint_optimizer = build_optimizers(model, config, model_variant)
        best_selection_auc = run_phase(
            model,
            train_loader,
            val_loader,
            warmup_optimizer,
            criterion,
            config["warmup_epochs"],
            device,
            fold_idx,
            "Warmup",
            log_file,
            save_dir,
            task,
            config,
            best_selection_auc,
            seed,
            model_variant,
        )

        model.unfreeze_encoder()
        best_selection_auc = run_phase(
            model,
            train_loader,
            val_loader,
            joint_optimizer,
            criterion,
            config["joint_epochs"],
            device,
            fold_idx,
            "Joint",
            log_file,
            save_dir,
            task,
            config,
            best_selection_auc,
            seed,
            model_variant,
        )

        fold_best_aucs.append(best_selection_auc)

    mean_auc = float(np.mean(fold_best_aucs))
    std_auc = float(np.std(fold_best_aucs))
    print(f"\nExperiment {config['name']} finished. Best selection AUC mean={mean_auc:.4f}, std={std_auc:.4f}")


if __name__ == "__main__":
    train_finetune()

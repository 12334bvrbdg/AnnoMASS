"""Bridge to the standalone ABCoRT project for the AnnoMASS GUI.

ABCoRT (graph-neural-network retention-time predictor, trained on the SMRT
dataset) lives in a sibling project directory, not inside this repo:

    <ANNOMASS_HOME>/external/ABCoRT/model.py         -- MyNet (GNN architecture)
    <ANNOMASS_HOME>/external/ABCoRT/feature_ops.py   -- atom/bond featurizers
    <ANNOMASS_HOME>/external/ABCoRT/model_dict/best_model.pth -- pretrained SMRT checkpoint

This module is the single place that knows that absolute path. If the
ABCoRT project or its dependencies (torch_geometric) are missing,
`is_available()` returns False and callers are expected to degrade
gracefully (disable the relevant UI) rather than fail.

Two GUI-facing capabilities:
  - predict_rt_batch(): run the pretrained (or a fine-tuned) checkpoint on a
    list of SMILES, purely informational -- predicted retention time is
    never used to score or rank candidates (see core/orthogonal_ranker.py
    for why: ranking must stay a transparent rule cascade).
  - finetune(): transfer-learning in the same spirit as the original
    project's train_transfer_FE.py -- freeze the pretrained embedding/conv
    backbone (in_node, in_edge, conv1..conv6) and fine-tune only the
    regression head on the caller's own (SMILES, retention_time) pairs.
    Unlike the original script (which only logs k-fold CV metrics for a
    paper table and never saves a checkpoint), this saves a usable .pth so
    the GUI can select it for future predictions.

Fine-tuning training data comes from an MGF where each spectrum carries a
SMILES field and an RTINMINUTES field (matchms lowercases MGF keys), i.e.
the user's own experimental spectra with known retention times -- the same
RTINMINUTES convention used when we added retention time to the MassBank
standard-reference dataset (scripts/data_prep/add_retention_time_to_standard_reference.py).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from core.paths import external_dir as _external_dir

ABCORT_PROJECT_ROOT = _external_dir("ABCoRT")
PRETRAINED_CHECKPOINT = ABCORT_PROJECT_ROOT / "model_dict" / "best_model.pth"

# Fine-tuned checkpoints are kept in this project (not written back into the
# ABCoRT sibling repo), so they version with the GUI, not with the model code.
FINETUNED_DIR = Path(__file__).resolve().parent.parent / "model" / "abcort_checkpoints"
FINETUNED_DIR.mkdir(parents=True, exist_ok=True)

# The layer names the pretrained model was built with -- frozen during
# fine-tuning so only the regression head adapts to the new RT scale.
# Matches ABCoRT/train_transfer_FE.py's set_freeze_by_names(...) call.
FROZEN_LAYER_NAMES = ["in_node", "in_edge", "conv1", "conv2", "conv3", "conv4", "conv5", "conv6"]

_IMPORT_ERROR = None
if ABCORT_PROJECT_ROOT.exists() and str(ABCORT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(ABCORT_PROJECT_ROOT))

try:
    import torch
    from torch_geometric.data import Data, Batch
    from feature_ops import atom_featurizer, bond_featurizer
    from model import MyNet
except Exception as exc:  # pragma: no cover - environment dependent
    torch = None
    Data = None
    Batch = None
    atom_featurizer = None
    bond_featurizer = None
    MyNet = None
    _IMPORT_ERROR = exc


def is_available() -> bool:
    """Whether the ABCoRT project + its dependencies (torch_geometric) can be used."""
    return _IMPORT_ERROR is None and ABCORT_PROJECT_ROOT.exists() and PRETRAINED_CHECKPOINT.exists()


def unavailable_reason() -> str:
    if not ABCORT_PROJECT_ROOT.exists():
        return f"ABCoRT project not found: {ABCORT_PROJECT_ROOT}"
    if _IMPORT_ERROR is not None:
        return f"ABCoRT import failed: {_IMPORT_ERROR}"
    if not PRETRAINED_CHECKPOINT.exists():
        return f"Pretrained checkpoint not found: {PRETRAINED_CHECKPOINT}"
    return ""


def list_checkpoints():
    """Return [(label, path), ...] -- pretrained first, then fine-tuned checkpoints found in FINETUNED_DIR."""
    items = [("Pretrained (SMRT)", str(PRETRAINED_CHECKPOINT))]
    for p in sorted(FINETUNED_DIR.glob("*.pth")):
        items.append((p.stem, str(p)))
    return items


def checkpoint_metadata(checkpoint_path):
    """Return the sidecar training-metadata dict for a fine-tuned checkpoint, or None."""
    meta_path = Path(checkpoint_path).with_suffix(".json")
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


_MODEL_CACHE = {}


def get_model(checkpoint_path=None, device=None):
    """Cached MyNet singleton, keyed by (resolved checkpoint path, device)."""
    checkpoint_path = checkpoint_path or PRETRAINED_CHECKPOINT
    device = device or get_device()
    key = (str(Path(checkpoint_path).resolve()), str(device))
    if key not in _MODEL_CACHE:
        model = MyNet().to(device)
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def smiles_to_graph(smiles: str):
    """SMILES -> PyG Data molecular graph. Ported from ABCoRT/predict_rt.py::smiles_to_graph."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    node_feats = np.array([atom_featurizer(a) for a in mol.GetAtoms()], dtype="float32")
    x = torch.tensor(node_feats, dtype=torch.float32)

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [s, e]
        col += [e, s]
        bf = bond_featurizer(bond)
        edge_feat.append(bf)
        edge_feat.append(bf)
    if len(edge_feat) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 21), dtype=torch.float32)
    else:
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def predict_rt_batch(smiles_list, checkpoint_path=None, device=None, batch_size=64):
    """Predict retention time for a list of SMILES.

    Returns a list of dicts, one per input (same order/length as
    smiles_list): {"smiles", "canonical_smiles", "predicted_rt_min", "valid"}.
    Unparseable SMILES get valid=False and predicted_rt_min=None -- purely
    informational output, never used to score or rank candidates.
    """
    from rdkit import Chem

    device = device or get_device()
    model = get_model(checkpoint_path, device)

    results = [None] * len(smiles_list)
    graphs, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        g = smiles_to_graph(smi) if smi else None
        if g is None:
            results[i] = {"smiles": smi, "canonical_smiles": None, "predicted_rt_min": None, "valid": False}
            continue
        graphs.append(g)
        valid_idx.append(i)

    for start in range(0, len(graphs), batch_size):
        chunk_graphs = graphs[start:start + batch_size]
        chunk_idx = valid_idx[start:start + batch_size]
        batch = Batch.from_data_list(chunk_graphs).to(device)
        with torch.no_grad():
            out = model(batch)
        out = out.view(-1).cpu().tolist()
        for idx, rt_sec in zip(chunk_idx, out):
            smi = smiles_list[idx]
            mol = Chem.MolFromSmiles(smi)
            canonical = Chem.MolToSmiles(mol) if mol else smi
            results[idx] = {
                "smiles": smi,
                "canonical_smiles": canonical,
                "predicted_rt_min": round(rt_sec / 60.0, 4),
                "valid": True,
            }

    return results


def extract_smiles_rt_pairs_from_mgf(mgf_path):
    """Read (smiles, retention_time_minutes) pairs from an MGF's SMILES/RTINMINUTES fields.

    matchms lowercases MGF field names, so 'SMILES=' -> metadata['smiles']
    and 'RTINMINUTES=' -> metadata['rtinminutes']. Spectra missing either
    field are skipped.
    """
    from matchms.importing import load_from_mgf

    pairs = []
    skipped = 0
    for spectrum in load_from_mgf(str(mgf_path)):
        smiles = spectrum.get("smiles")
        rt = spectrum.get("rtinminutes")
        if not smiles or rt in (None, ""):
            skipped += 1
            continue
        try:
            rt_min = float(rt)
        except (TypeError, ValueError):
            skipped += 1
            continue
        pairs.append((str(smiles), rt_min))
    return pairs, skipped


def set_freeze_all(model, requires_grad=True):
    for _, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = requires_grad


def set_freeze_by_names(model, layer_names, freeze=True):
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze


def finetune(
    train_pairs,
    out_name,
    base_checkpoint_path=None,
    epochs=100,
    lr=1e-4,
    batch_size=8,
    val_frac=0.15,
    seed=42,
    progress_cb=None,
    device=None,
):
    """Fine-tune the regression head on (smiles, retention_time_minutes) pairs.

    Freezes in_node/in_edge/conv1..conv6 (the pretrained SMRT backbone) and
    trains only the head (graphline/trans_graph/trans_add/trans_out/out_lin),
    matching the freezing strategy in ABCoRT/train_transfer_FE.py. Unlike
    that script (5 seeds x 10-fold CV, metrics-only, no saved model), this
    does a single train/val split and saves a usable checkpoint plus a JSON
    sidecar with training metadata.

    Returns {"path", "n_train", "n_val", "final_train_mae_min", "final_val_mae_min"}.
    """
    from torch import optim
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    from torch.utils.data import random_split
    from torch_geometric.loader import DataLoader

    device = device or get_device()
    base_checkpoint_path = base_checkpoint_path or PRETRAINED_CHECKPOINT

    torch.manual_seed(seed)
    np.random.seed(seed)

    data_list = []
    skipped = 0
    for smiles, rt_min in train_pairs:
        g = smiles_to_graph(smiles)
        if g is None:
            skipped += 1
            continue
        g.y = torch.tensor(rt_min * 60.0, dtype=torch.float32)  # model target unit is seconds
        data_list.append(g)

    if len(data_list) < 4:
        raise ValueError(
            f"Only {len(data_list)} usable (SMILES, RT) pairs after parsing "
            f"({skipped} skipped) -- need at least 4 to fine-tune."
        )

    n_val = max(1, int(round(len(data_list) * val_frac))) if len(data_list) >= 8 else 0
    n_train = len(data_list) - n_val
    if n_val > 0:
        train_set, val_set = random_split(
            data_list, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
        )
    else:
        train_set, val_set = data_list, []

    model = MyNet().to(device)
    state = torch.load(base_checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    set_freeze_all(model, requires_grad=True)
    set_freeze_by_names(model, FROZEN_LAYER_NAMES, freeze=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, amsgrad=True, weight_decay=1e-2)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(10, epochs // 5))
    criterion = torch.nn.SmoothL1Loss()

    train_bs = min(batch_size, max(1, len(train_set)))
    train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=train_bs, shuffle=False, num_workers=0) if n_val else None

    final_train_mae = None
    final_val_mae = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_abs_err, total_n = 0.0, 0
        for data in train_loader:
            data = data.to(device)
            y_hat = model(data)
            if y_hat.dim() == 0:
                y_hat = y_hat.view(1)
            loss = criterion(y_hat, data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_abs_err += torch.abs(y_hat.detach() - data.y).sum().item()
            total_n += data.y.numel()
        scheduler.step()
        train_mae_sec = total_abs_err / max(1, total_n)
        final_train_mae = train_mae_sec / 60.0

        val_mae_sec = None
        if val_loader is not None:
            model.eval()
            abs_err, n = 0.0, 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    y_hat = model(data)
                    if y_hat.dim() == 0:
                        y_hat = y_hat.view(1)
                    abs_err += torch.abs(y_hat - data.y).sum().item()
                    n += data.y.numel()
            val_mae_sec = abs_err / max(1, n)
            final_val_mae = val_mae_sec / 60.0

        if progress_cb is not None:
            progress_cb(epoch, epochs, final_train_mae, final_val_mae)

    out_path = FINETUNED_DIR / f"{out_name}.pth"
    torch.save(model.state_dict(), out_path)
    meta = {
        "base_checkpoint": str(base_checkpoint_path),
        "n_train": n_train,
        "n_val": n_val,
        "skipped": skipped,
        "epochs": epochs,
        "final_train_mae_min": final_train_mae,
        "final_val_mae_min": final_val_mae,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {"path": str(out_path), **meta}

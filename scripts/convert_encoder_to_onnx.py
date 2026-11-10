"""Convert both encoders (Ordinal + Target) from TargetEncoderBuilder to ONNX."""

from pathlib import Path
import numpy as np
import pandas as pd
import time

from onnxruntime import InferenceSession
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType

from src.features.target_encoder import TargetEncoderBuilder
from src.agents.tools import _gold

# === 1. Load ===
builder = TargetEncoderBuilder.load("data/models/target_encoder.pkl")
ord_encoder = builder.transformer_.named_transformers_["ord"]
te_transformer = builder.te_transformer_
df = _gold()

# === 2. Convert OrdinalEncoder ===
sample_ord = df[builder.cat_cols_].head(5).astype(str)
ord_onnx = to_onnx(
    ord_encoder,
    initial_types=[("ord_inputs", StringTensorType([None, len(builder.cat_cols_)]))],
    target_opset={"": 17, "ai.onnx.ml": 3},
)
ord_path = Path("data/models/ordinal_encoder.onnx")
ord_path.write_bytes(ord_onnx.SerializeToString())
print(f"OrdinalEncoder ONNX: {ord_path}")

# === 3. Convert TargetEncoder ===
sample_te = df[builder.te_cols].head(5).astype(str)
te_onnx = to_onnx(
    te_transformer,
    initial_types=[("te_inputs", StringTensorType([None, len(builder.te_cols)]))],
    target_opset={"": 17, "ai.onnx.ml": 3},
)
te_path = Path("data/models/target_encoder.onnx")
te_path.write_bytes(te_onnx.SerializeToString())
print(f"TargetEncoder ONNX: {te_path}")

# === 4. Load both encoders
sess_ord = InferenceSession(str(ord_path))
sess_te = InferenceSession(str(te_path))

# === 5. Diagnostic OrdinalEncoder ===
sklearn_ord = ord_encoder.transform(df[builder.cat_cols_].head(5).astype(str)).to_numpy()
onnx_ord = sess_ord.run(None, {"ord_inputs": sample_ord.values})[0]
# Post-processing NaN (le fix)
nan_mask = df[builder.cat_cols_].head(5).isna().to_numpy()
onnx_ord_fixed = onnx_ord.copy()
onnx_ord_fixed[nan_mask] = np.nan

# Validation
match_before = np.allclose(sklearn_ord, onnx_ord, atol=1e-5, equal_nan=True)
match_after = np.allclose(sklearn_ord, onnx_ord_fixed, atol=1e-5, equal_nan=True)
print(f"Match before fix : {match_before}")
print(f"Match after fix : {match_after}")

print(f"\nsklearn_ord: {sklearn_ord.shape} {sklearn_ord.dtype}")
print(f"onnx_ord:    {onnx_ord.shape} {onnx_ord.dtype}")

diff_cols = []
for i, col in enumerate(builder.cat_cols_):
    if not np.allclose(sklearn_ord[:, i], onnx_ord[:, i], atol=1e-5, equal_nan=True):
        diff_cols.append(col)

print(f"Different columns : {len(diff_cols)}/{len(builder.cat_cols_)}")
print(f"List : {diff_cols[:10]}")

if diff_cols:
    col = diff_cols[0]
    idx = builder.cat_cols_.index(col)
    print(f"\nDétail '{col}' :")
    print(f"  Input brut       : {df[col].head(5).tolist()}")
    print(f"  Input astype str : {sample_ord[col].tolist()}")
    print(f"  sklearn          : {sklearn_ord[:, idx].tolist()}")
    print(f"  onnx             : {onnx_ord[:, idx].tolist()}")
    print(f"  NaN count input  : {int(df[col].head(5).isna().sum())}")
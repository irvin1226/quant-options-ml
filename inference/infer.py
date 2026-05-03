import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
from model_nn import ModelNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BLEND_ALPHA       = 0.6
MC_DROPOUT_PASSES = 5

TRADE_MODE = os.environ.get("TRADE_MODE", "dynamic")

GBT_MODEL_PATH      = f"deployment_models/{TRADE_MODE}/gbt/gbt_model_2026.pkl"
GBT_FEATURES_PATH   = f"deployment_models/{TRADE_MODE}/gbt/gbt_features_2026.pkl"
GBT_CALIBRATOR_PATH = f"deployment_models/{TRADE_MODE}/gbt/gbt_calibrator_2026.pkl"
GBT_THRESHOLDS_PATH = f"deployment_models/{TRADE_MODE}/gbt/gbt_thresholds_2026.json"

NN_MODEL_PATH      = f"deployment_models/{TRADE_MODE}/nn/best_nn_2026.pt"
NN_FEATURES_PATH   = f"deployment_models/{TRADE_MODE}/nn/nn_features_2026.pkl"
NN_CALIBRATOR_PATH = f"deployment_models/{TRADE_MODE}/nn/nn_calibrator_2026.pkl"
NN_THRESHOLDS_PATH = f"deployment_models/{TRADE_MODE}/nn/nn_thresholds_2026.json"


def _load_gbt_artifacts():
    model      = joblib.load(GBT_MODEL_PATH)
    features   = joblib.load(GBT_FEATURES_PATH)
    calibrator = joblib.load(GBT_CALIBRATOR_PATH)

    with open(GBT_THRESHOLDS_PATH) as f:
        thresholds = json.load(f)

    return {
        'model':        model,
        'features':     features,
        'calibrator':   calibrator,
        'threshold':    thresholds['threshold'],
        'hc_threshold': thresholds['hc_threshold'],
    }


def _load_nn_artifacts():
    features   = joblib.load(NN_FEATURES_PATH)
    calibrator = joblib.load(NN_CALIBRATOR_PATH)

    inputSize = len(features.COLUMNS)
    model = ModelNN(input_size=inputSize)
    model.load_state_dict(torch.load(NN_MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    with open(NN_THRESHOLDS_PATH) as f:
        thresholds = json.load(f)

    return {
        'model':        model,
        'features':     features,
        'calibrator':   calibrator,
        'threshold':    thresholds['threshold'],
        'hc_threshold': thresholds['hc_threshold'],
    }


def _run_gbt(featureData, artifacts):
    X, cleaned = artifacts['features'].transform(featureData)

    if len(X) == 0:
        return pd.DataFrame()

    X_df          = pd.DataFrame(X, columns=artifacts['features'].COLUMNS)
    rawScores        = artifacts['model'].predict(X_df)
    calibratedScores = artifacts['calibrator'].transform(rawScores)
    blendedScores    = BLEND_ALPHA * calibratedScores + (1 - BLEND_ALPHA) * rawScores

    result = cleaned[['expiration', 'strike', 'right', 'ask', 'dte', 'moneyness', 'underlying_price']].copy()
    result = result.reset_index(drop=True)
    result['gbt_score']              = blendedScores
    result['gbt_passes_threshold']   = result['gbt_score'] >= artifacts['threshold']
    result['gbt_is_high_confidence'] = result['gbt_score'] >= artifacts['hc_threshold']

    return result


def _run_nn(featureData, artifacts):
    X, cleaned = artifacts['features'].transform(featureData)

    if len(X) == 0:
        return pd.DataFrame()

    X_tensor = torch.tensor(X, dtype=torch.float).to(DEVICE)

    model = artifacts['model']
    model.eval()

    with torch.no_grad():
        singleLogits = model(X_tensor)

    # MC dropout blend - replicates train_nn.py inference loop
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    mcPasses = []
    for _ in range(MC_DROPOUT_PASSES):
        with torch.no_grad():
            mcPasses.append(model(X_tensor))

    model.eval()

    mcMean        = torch.stack(mcPasses, dim=0).mean(dim=0)
    blendedLogits = BLEND_ALPHA * singleLogits + (1 - BLEND_ALPHA) * mcMean
    logitsNp      = blendedLogits.cpu().numpy().flatten()
    nnScores      = artifacts['calibrator'].transform(logitsNp)

    # NN abstains when its max prediction never reaches the threshold -
    # matches the 2026 abstention behavior observed during walk-forward.
    nnAbstained = float(nnScores.max()) < artifacts['threshold']

    result = cleaned[['expiration', 'strike', 'right']].copy()
    result = result.reset_index(drop=True)
    result['nn_score']    = nnScores
    result['nn_abstained'] = nnAbstained

    if nnAbstained:
        result['nn_passes_threshold']   = False
        result['nn_is_high_confidence'] = False
    else:
        result['nn_passes_threshold']   = result['nn_score'] >= artifacts['threshold']
        result['nn_is_high_confidence'] = result['nn_score'] >= artifacts['hc_threshold']

    return result


# Loads all model artifacts for GBT and NN from deployment_models/{TRADE_MODE}/.
# Call once at startup from run_dynamic.py or run_fixed.py - not on every inference cycle.
def load_artifacts():
    print(f"[infer] Loading artifacts for mode: {TRADE_MODE}")
    gbtArtifacts = _load_gbt_artifacts()
    nnArtifacts  = _load_nn_artifacts()
    return gbtArtifacts, nnArtifacts


# Runs inference for both models on the prepared feature DataFrame.
# GBT is the primary signal. NN scores are attached for reference.
def get_signals(featureData, gbtArtifacts, nnArtifacts):
    if featureData.empty:
        return pd.DataFrame()

    gbtResult = _run_gbt(featureData, gbtArtifacts)
    nnResult  = _run_nn(featureData, nnArtifacts)

    if gbtResult.empty:
        return pd.DataFrame()

    joinColumns = ['expiration', 'strike', 'right']
    signals = gbtResult.merge(nnResult, on=joinColumns, how='left')

    gbtFired = signals['gbt_passes_threshold']
    signals  = signals[gbtFired].reset_index(drop=True)

    return signals
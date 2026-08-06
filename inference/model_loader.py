import os
import joblib
import xgboost as xgb

def load_stacking_model(model_path: str):
    """
    Loads the trained Hybrid Stacked Ensemble model and applies necessary 
    compatibility patches for XGBoost sub-estimators.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
        
    model = joblib.load(model_path)
    
    # Patch XGBoost sub-estimators for cross-version unpickling compatibility
    xgb_attrs = {
        'use_label_encoder': False,
        'gpu_id': -1,
        'predictor': 'auto',
        'enable_categorical': False,
        'max_cat_to_onehot': 4,
        'eval_metric': 'mlogloss',
        'early_stopping_rounds': None,
        'callbacks': None,
        'tree_method': 'hist'
    }
    
    if hasattr(model, 'estimators_'):
        for est in model.estimators_:
            if isinstance(est, xgb.XGBClassifier):
                for k, v in xgb_attrs.items():
                    if not hasattr(est, k):
                        setattr(est, k, v)
                        
    return model

def load_artifact(artifact_path: str):
    """Generic loader for joblib pickled artifacts (vectorizers, SVD, PCA, Scalers)."""
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Artifact file not found at: {artifact_path}")
    return joblib.load(artifact_path)

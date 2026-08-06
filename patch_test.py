import joblib
import xgboost as xgb
import numpy as np

def patch_xgb(est):
    if isinstance(est, xgb.XGBClassifier):
        attrs = {
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
        for k, v in attrs.items():
            if not hasattr(est, k):
                setattr(est, k, v)

stack_model = joblib.load('Colab/models/stacking/Hybrid_Stacked_Ensemble.pkl')
for est in stack_model.estimators_:
    patch_xgb(est)

X_fusion = joblib.load('Colab/fusion/fingerprint_fusion.pkl')
print('Classes:', stack_model.classes_)
print('Preds:', stack_model.predict(X_fusion[:2]))
print('Probs:', stack_model.predict_proba(X_fusion[:2]))

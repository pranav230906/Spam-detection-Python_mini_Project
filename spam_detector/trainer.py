# Placeholder for trainer.py
# trainer.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import joblib
import os
from pipeline_builder import get_pipeline

def train_and_evaluate(df, model_name="MultinomialNB", test_size=0.2, use_grid=False):
    """
    Train pipeline and return (pipeline, metrics, figs)
    figs: dict with 'confusion', 'roc', 'features' matplotlib.Figure objects
    """
    X = df['text'].values
    y = (df['label'] == 'spam').astype(int).values  # binary 1=spam,0=ham

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    pipeline = get_pipeline(model_name=model_name)

    if use_grid:
        param_grid = {}
        if model_name == "MultinomialNB":
            param_grid = {
                'clf__alpha': [0.1, 0.5, 1.0]
            }
        elif model_name == "LogisticRegression":
            param_grid = {
                'clf__C': [0.01, 0.1, 1.0, 10.0]
            }
        gs = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
        gs.fit(X_train, y_train)
        pipeline = gs.best_estimator_
        best = getattr(gs, 'best_params_', {})
    else:
        pipeline.fit(X_train, y_train)
        best = {}

    y_pred = pipeline.predict(X_test)
    probs = None
    try:
        probs = pipeline.predict_proba(X_test)[:,1]
    except Exception:
        # fallback for estimators without predict_proba
        probs = pipeline.decision_function(X_test)
        # scale to [0,1]
        probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-9)

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
    }
    # ROC AUC
    try:
        fpr, tpr, _ = roc_curve(y_test, probs)
        metrics['auc'] = float(auc(fpr, tpr))
    except Exception:
        metrics['auc'] = None

    # create figures
    figs = {}
    figs['confusion'] = plot_confusion_matrix_fig(y_test, y_pred)
    if metrics['auc'] is not None:
        figs['roc'] = plot_roc_curve_fig(y_test, probs)
    try:
        figs['features'] = plot_top_features_fig(pipeline, n=15)
    except Exception:
        figs['features'] = None

    return pipeline, metrics, figs

def plot_confusion_matrix_fig(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,3))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0,1])
    ax.set_xticklabels(["Ham","Spam"])
    ax.set_yticks([0,1])
    ax.set_yticklabels(["Ham","Spam"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig

def plot_roc_curve_fig(y_true, probs):
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0,1], [0,1], linestyle='--')
    ax.set_xlim([0.0,1.0])
    ax.set_ylim([0.0,1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig

def plot_top_features_fig(pipeline, n=20):
    # Works for linear models (coef_) or MultinomialNB (feature_log_prob_)
    vect = pipeline.named_steps.get('tfidf', None)
    clf = pipeline.named_steps.get('clf', None)
    if vect is None or clf is None:
        raise ValueError("Pipeline missing expected steps.")

    feature_names = vect.get_feature_names_out()
    fig, ax = plt.subplots(figsize=(6,2.5))
    try:
        if hasattr(clf, 'coef_'):
            coefs = clf.coef_[0]
            topn = np.argsort(coefs)[-n:]
            top_features = feature_names[topn]
            top_vals = coefs[topn]
            ax.barh(top_features, top_vals)
            ax.set_title("Top positive coefficients")
            fig.tight_layout()
            return fig
        elif hasattr(clf, 'feature_log_prob_'):
            # For MultinomialNB: log probabilities for classes; class 1 corresponds to spam if trained that way
            logp = clf.feature_log_prob_[1]  # spam class
            topn = np.argsort(logp)[-n:]
            top_features = feature_names[topn]
            top_vals = logp[topn]
            ax.barh(top_features, top_vals)
            ax.set_title("Top features (spam)")
            fig.tight_layout()
            return fig
    except Exception as e:
        raise e
    raise ValueError("Classifier does not expose coefficients or feature_log_prob_")

def get_top_features(pipeline, n=20):
    # convenience wrapper returning list of (feature, score)
    vect = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']
    feature_names = vect.get_feature_names_out()
    if hasattr(clf, 'coef_'):
        coefs = clf.coef_[0]
        topn = np.argsort(coefs)[-n:]
        return [(feature_names[i], float(coefs[i])) for i in reversed(topn)]
    elif hasattr(clf, 'feature_log_prob_'):
        logp = clf.feature_log_prob_[1]
        topn = np.argsort(logp)[-n:]
        return [(feature_names[i], float(logp[i])) for i in reversed(topn)]
    else:
        return []

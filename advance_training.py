# advance_training.py
# Usage:
#   python advance_training.py --data harrypool_training_data.json --out model.pkl

import argparse, json, runpy, random, datetime, joblib
from pathlib import Path
from collections import Counter
import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score

SEED = 42
random.seed(SEED); np.random.seed(SEED)

def _load_json_or_py(path: Path):
    if path.suffix == ".py":
        ns = runpy.run_path(str(path))
        if "training_data" not in ns:
            raise ValueError(f"{path} must define `training_data = {{...}}`")
        return ns["training_data"]
    txt = path.read_text(encoding="utf-8")
    if txt.strip().startswith("training_data"):
        txt = txt[txt.find("{"):]
    return json.loads(txt)

def load_dataset(path_str: str):
    p = Path(path_str)
    if not p.exists():
        for cand in ["harrypool_training_data.json","harrpool_training_data.json","harrypool_training_data.py"]:
            if Path(cand).exists():
                p = Path(cand); break
        else:
            raise FileNotFoundError(f"Dataset not found: {path_str}")
    data = _load_json_or_py(p)
    intents = data.get("intents") or data.get("training_data", {}).get("intents")
    if not intents: raise ValueError("No 'intents' found.")
    X, y, tag2responses = [], [], {}
    for it in intents:
        tag = it["tag"]
        tag2responses[tag] = it.get("responses", [])
        for patt in it.get("patterns", []):
            X.append(patt.strip()); y.append(tag)
    if not X: raise ValueError("No patterns to train on.")
    return X, y, tag2responses

def build_features():
    word = TfidfVectorizer(analyzer="word", ngram_range=(1,3),
                           lowercase=True, strip_accents="unicode",
                           sublinear_tf=True, min_df=1)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3,6),
                           lowercase=False, sublinear_tf=True, min_df=1)
    return FeatureUnion([("w", word), ("c", char)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to training data (.json or .py)")
    ap.add_argument("--out", default="model.pkl", help="Output model path")
    args = ap.parse_args()

    X, y, tag2responses = load_dataset(args.data)

    counts = Counter(y)
    print("Class counts:", dict(counts))
    min_count = min(counts.values())
    if min_count < 2:
        raise SystemExit("Each intent needs at least 2 patterns. Add more examples and retry.")
    cv_splits = max(2, min(5, min_count))
    print(f"Using StratifiedKFold with n_splits={cv_splits}")

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    features = build_features()
    base = Pipeline([("features", features),
                     ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])

    grids = [
        {
            "clf": [LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")],
            "features__w__ngram_range": [(1,2), (1,3)],
            "features__c__ngram_range": [(3,5), (3,6)],
            "clf__C": [0.5, 1.0, 2.0, 4.0],
        },
        {
            
            "clf": [CalibratedClassifierCV(LinearSVC(class_weight="balanced"), method="sigmoid", cv=3)],
            "features__w__ngram_range": [(1,2), (1,3)],
            "features__c__ngram_range": [(3,5)],
            "clf__estimator__C": [0.5, 1.0, 2.0],
        },
        {
            "clf": [ComplementNB()],
            "features__w__ngram_range": [(1,2)],
            "features__c__ngram_range": [(3,5)],
            "clf__alpha": [0.3, 0.7, 1.0],
        },
    ]

    gs = GridSearchCV(
        base, param_grid=grids, scoring="f1_macro",
        cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED),
        n_jobs=-1, verbose=1, refit=True
    )
    gs.fit(X_tr, y_tr)

    best = gs.best_estimator_
    print("\nBest params:", gs.best_params_)
    print("CV best f1_macro:", round(gs.best_score_, 4))

    y_pred = best.predict(X_va)
    print("\nValidation accuracy:", round(accuracy_score(y_va, y_pred), 4))
    print("Validation macro-F1:", round(f1_score(y_va, y_pred, average='macro'), 4))
    print("\nReport:\n", classification_report(y_va, y_pred))

    artifact = {
        "pipeline": best,
        "responses": tag2responses,
        "labels": sorted(set(y)),
        "metadata": {
            "name": "harrypool",
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "framework": "scikit-learn (word+char TF-IDF) + model search",
            "best_params": gs.best_params_,
            "cv_f1_macro": float(gs.best_score_),
        },
    }
    joblib.dump(artifact, args.out)
    print("\nSaved:", args.out)

if __name__ == "__main__":
    main()

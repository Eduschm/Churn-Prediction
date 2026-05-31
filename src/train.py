import os
import json
import ast
import argparse
import pandas as pd
from typing import Dict, Any, Optional

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    average_precision_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier

import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import skops.io as skio


class ChurnTrainer:
    def __init__(
        self,
        df: pd.DataFrame,
        numerical_features=None,
        target_col: str = "Churn",
        random_state: int = 42,
        test_size: float = 0.2,
        model_dir: str = "models",
        results_dir: str = "results",
        config_dir: str = "config",
    ):
        self.df = df.copy()
        self.target_col = target_col
        self.random_state = random_state
        self.test_size = test_size
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.config_dir = config_dir

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)

        # default numerical features if not provided
        self.numerical_features = numerical_features or [
            "MonthlyCharges",
            "tenure",
            "TotalCharges",
            "avg_monthly_spend",
            "contract_value",
        ]
        self.categorical_features = [
            col
            for col in self.df.columns
            if col not in self.numerical_features + [self.target_col]
        ]

        # placeholders for splits and objects
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.preprocessor = None
        self.cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=self.random_state
        )

        # pipelines and param grids
        self._build_preprocessor()
        self._build_pipelines_and_grids()

        # to store results
        self.best_models: Dict[str, Any] = {}
        self.best_params: Dict[str, Dict[str, Any]] = {}
        self.voting_classifier = None
        self.voting_threshold = None

        # loaded saved params (for quick mode)
        self.saved_best_params: Dict[str, Dict[str, str]] = {}

    # Preprocessing

    def _build_preprocessor(self):
        num_pipeline = Pipeline(
            [("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())]
        )
        cat_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first")),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, self.numerical_features),
                ("cat", cat_pipeline, self.categorical_features),
            ],
            remainder="drop",
        )


    # Model pipelines & grids

    def _build_pipelines_and_grids(self):
        # Training pipelines (with SMOTE and ImbPipeline)
        self.pipelines = {
            "xgb": ImbPipeline(
                [
                    ("preprocessor", self.preprocessor),
                    ("smote", SMOTE(random_state=self.random_state)),
                    (
                        "classifier",
                        xgb.XGBClassifier(
                            use_label_encoder=False,
                            eval_metric="logloss",
                            random_state=self.random_state,
                            objective="binary:logistic",
                        ),
                    ),
                ]
            ),
            "lgbm": ImbPipeline(
                [
                    ("preprocessor", self.preprocessor),
                    ("smote", SMOTE(random_state=self.random_state)),
                    ("classifier", lgb.LGBMClassifier(random_state=self.random_state)),
                ]
            ),
            "dt": ImbPipeline(
                [
                    ("preprocessor", self.preprocessor),
                    ("smote", SMOTE(random_state=self.random_state)),
                    ("classifier", DecisionTreeClassifier(random_state=self.random_state)),
                ]
            ),
        }

        # Param grids
        self.param_grids = {
            "xgb": {
                "classifier__n_estimators": [100, 300],
                "classifier__max_depth": [3],
                "classifier__learning_rate": [0.05],
                "classifier__subsample": [0.7, 0.9, 1.0],
                "classifier__colsample_bytree": [0.7, 0.9, 1.0],
                "classifier__gamma": [0.1],
                "classifier__reg_alpha": [0.1],
                "classifier__reg_lambda": [1, 1.5, 2],
                "classifier__scale_pos_weight": [1, 2, 4],
            },
            "lgbm": {
                "classifier__n_estimators": [100, 300],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__subsample": [0.7, 1.0],
                "classifier__colsample_bytree": [0.7, 1.0],
                "classifier__reg_alpha": [0.1, 0.5],
                "classifier__reg_lambda": [1, 1.5, 2],
                "classifier__scale_pos_weight": [1, 4],
            },
            "dt": {
                "classifier__max_depth": [None, 5, 10, 15],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__max_features": ["sqrt", "log2", None],
                "classifier__class_weight": ["balanced", None],
            },
        }


    # Data split

    def split(self):
        X = self.df.drop(self.target_col, axis=1)
        y = self.df[self.target_col].map({"No": 0, "Yes": 1}).astype(int)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )


    # Training

    def _run_grid_search(
        self, name: str, scoring=None, refit: str = "f1", n_jobs: int = -1, verbose: int = 1
    ):
        pipeline = self.pipelines[name]
        grid = GridSearchCV(
            pipeline,
            self.param_grids[name],
            cv=self.cv,
            scoring=scoring
            or ["recall", "roc_auc", "f1", "accuracy", "precision"],
            n_jobs=n_jobs,
            refit=refit,
            verbose=verbose,
        )
        grid.fit(self.X_train, self.y_train)
        self.best_models[name] = grid.best_estimator_
        # convert params to JSON-serializable strings
        self.best_params[name] = {k: str(v) for k, v in grid.best_params_.items()}
        return grid

    def _parse_param_value(self, s: str):
        if s is None:
            return None
        if isinstance(s, (int, float, bool, dict, list)):
            return s
        if s == "None":
            return None
        try:
            return json.loads(s)
        except Exception:
            pass
        try:
            return ast.literal_eval(s)
        except Exception:
            pass
        return s

    def load_saved_params(self, path: Optional[str] = None):
        path = path or os.path.join(self.config_dir, "best_params.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Saved params not found at {path}")
        with open(path, "r") as f:
            self.saved_best_params = json.load(f)
        return self.saved_best_params

    def _apply_saved_params_and_fit(self, name: str):
        """
        Apply saved params (from self.saved_best_params[name]) to the pipeline,
        fit it once on training data (no CV), and store in self.best_models.
        """
        if name not in self.saved_best_params:
            raise ValueError(f"No saved params for model '{name}' in loaded best_params.json")
        pipeline = self.pipelines[name]
        raw_params = self.saved_best_params[name]
        params = {k: self._parse_param_value(v) for k, v in raw_params.items()}
        pipeline.set_params(**params)
        pipeline.fit(self.X_train, self.y_train)
        self.best_models[name] = pipeline
        self.best_params[name] = raw_params
        return pipeline

    # Evaluation

    def evaluate(self, name: str):
        model = self.best_models.get(name)
        if model is None:
            raise ValueError(f"No trained model found for '{name}'")
        y_pred = model.predict(self.X_test)
        print(f"\n{name.upper()} Results:")
        print("Best parameters:", self.best_params[name])
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(self.X_test)[:, 1]
                pr_auc = average_precision_score(self.y_test, y_proba)
                print(f"PR-AUC: {pr_auc:.4f}")
            except Exception:
                pass


    # Deployment-ready saving

    def _get_fitted_components(self, name: str):
        """
        Returns fitted (preprocessor, classifier) from the best model pipeline.
        Assumes self.best_models[name] is an ImbPipeline with steps:
        preprocessor -> smote -> classifier
        """
        model = self.best_models.get(name)
        if model is None:
            raise ValueError(f"No trained model found for '{name}'")

        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        return preprocessor, classifier

    def save_dt_model(self, filename: str = "model_dt.skops"):
        """Save a pure sklearn deployment pipeline (no SMOTE) using skops."""
        preprocessor, classifier = self._get_fitted_components("dt")
        deployment_pipeline = Pipeline(
            [("preprocessor", preprocessor), ("classifier", classifier)]
        )
        path = os.path.join(self.model_dir, filename)
        skio.dump(deployment_pipeline, path)
        print(f"Decision Tree deployment model saved to: {path}")
        return path

    def save_xgb_model(self, filename: str = "model_xgb.skops"):
        preprocessor, classifier = self._get_fitted_components("xgb")
        deployment_pipeline = Pipeline(
            [("preprocessor", preprocessor), ("classifier", classifier)]
        )
        path = os.path.join(self.model_dir, filename)
        skio.dump(deployment_pipeline, path)
        print(f"XGBoost deployment model saved to: {path}")
        return path

    def save_lgbm_model(self, filename: str = "model_lgbm.skops"):
        preprocessor, classifier = self._get_fitted_components("lgbm")
        deployment_pipeline = Pipeline(
            [("preprocessor", preprocessor), ("classifier", classifier)]
        )
        path = os.path.join(self.model_dir, filename)
        skio.dump(deployment_pipeline, path)
        print(f"LightGBM deployment model saved to: {path}")
        return path

    def save_all_params(self):
        path = os.path.join(self.config_dir, "best_params.json")
        with open(path, "w") as f:
            json.dump(self.best_params, f, indent=2)
        return path

    # Voting classifier

    def build_voting_classifier(self, max_fpr: float = 0.1):
        """Build and train a soft-voting ensemble from the best individual models."""
        required_models = ["xgb", "dt", "lgbm"]
        for model_name in required_models:
            if model_name not in self.best_models:
                raise ValueError(
                    f"Model '{model_name}' must be trained before building voting classifier. "
                    f"Run run_all() first or train individual models."
                )

        print("\n" + "=" * 60)
        print("Building Voting Classifier (PREFERRED MODEL)...")
        print("=" * 60)

        self.voting_classifier = VotingClassifier(
            estimators=[
                ("xgb", self.best_models["xgb"]),
                ("dt", self.best_models["dt"]),
                ("lgbm", self.best_models["lgbm"]),
            ],
            voting="soft",
        )

        print("Training voting classifier...")
        self.voting_classifier.fit(self.X_train, self.y_train)

        # Evaluate with default threshold
        y_pred_voting = self.voting_classifier.predict(self.X_test)
        print("\nVoting Classifier Results (default threshold):")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred_voting))
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred_voting))

        # Find optimal threshold using training-set predictions to avoid test leakage.
        # Using training probabilities is a conservative estimate; for a stricter approach
        # hold out a dedicated validation split before calling this method.
        train_proba = self.voting_classifier.predict_proba(self.X_train)[:, 1]
        fpr_train, _tpr_train, thresholds = roc_curve(self.y_train, train_proba)

        threshold_index = next(
            (i for i, fpr in enumerate(fpr_train) if fpr > max_fpr),
            len(fpr_train) - 1,
        )
        self.voting_threshold = thresholds[threshold_index]
        print(
            f"\nVoting Classifier Threshold for Max FPR {max_fpr}: "
            f"{self.voting_threshold:.4f}"
        )

        # Evaluate on test set with chosen threshold
        voting_pred_proba = self.voting_classifier.predict_proba(self.X_test)[:, 1]
        y_pred_voting_thresholded = (voting_pred_proba >= self.voting_threshold).astype(int)
        print("\nClassification Report with Thresholding:")
        print(classification_report(self.y_test, y_pred_voting_thresholded))
        print("\nConfusion Matrix with Thresholding:")
        print(confusion_matrix(self.y_test, y_pred_voting_thresholded))
        pr_auc = average_precision_score(self.y_test, voting_pred_proba)
        print(f"PR-AUC: {pr_auc:.4f}")

        self.save_voting_classifier()

    def save_voting_classifier(self, filename: str = "model.skops"):
        if self.voting_classifier is None:
            raise ValueError(
                "Voting classifier has not been built yet. "
                "Run build_voting_classifier() first."
            )

        path = os.path.join(self.model_dir, filename)
        skio.dump(self.voting_classifier, path)
        print(f"\nPreferred model (Voting Classifier) saved to: {path}")

        threshold_path = os.path.join(self.results_dir, "voting_threshold.json")
        with open(threshold_path, "w") as f:
            json.dump({"threshold": float(self.voting_threshold)}, f, indent=2)
        print(f"Voting threshold saved to: {threshold_path}")

        return path


    # Orchestrator

    def run_all(
        self,
        train_xgb: bool = True,
        train_lgbm: bool = True,
        train_dt: bool = True,
        build_voting: bool = True,
        max_fpr: float = 0.1,
        quick: bool = False,
    ):
        """
        Train all models.

        Args:
            quick: If True, skip grid search and fit using saved params from
                   config/best_params.json. If False (default), run full GridSearchCV.
        """
        self.split()

        if quick:
            self.load_saved_params()

        if train_xgb:
            print("\n" + "=" * 60)
            print("Training XGBoost...")
            print("=" * 60)
            if quick:
                self._apply_saved_params_and_fit("xgb")
            else:
                self._run_grid_search(
                    "xgb",
                    scoring=["recall", "roc_auc", "f1", "accuracy", "precision"],
                )
            self.evaluate("xgb")
            self.save_xgb_model()

        if train_dt:
            print("\n" + "=" * 60)
            print("Training Decision Tree...")
            print("=" * 60)
            if quick:
                self._apply_saved_params_and_fit("dt")
            else:
                self._run_grid_search("dt", scoring=["recall", "roc_auc", "f1"])
            self.evaluate("dt")
            self.save_dt_model()

        if train_lgbm:
            print("\n" + "=" * 60)
            print("Training LightGBM...")
            print("=" * 60)
            if quick:
                self._apply_saved_params_and_fit("lgbm")
            else:
                self._run_grid_search("lgbm", scoring=["recall", "roc_auc", "f1"])
            self.evaluate("lgbm")
            self.save_lgbm_model()

        if build_voting and all([train_xgb, train_lgbm, train_dt]):
            self.build_voting_classifier(max_fpr=max_fpr)
        elif build_voting:
            print(
                "\nWarning: All three models (xgb, lgbm, dt) must be trained "
                "to build voting classifier."
            )

        params_path = self.save_all_params()
        print(f"\nParameters saved to {params_path}")
        print("\nTraining complete!")
        print(f"\n{'=' * 60}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    from src.load_data import DataProcessor

    parser = argparse.ArgumentParser(description="Train churn models")
    parser.add_argument("--data", type=str, default=os.path.join("data", "dataset.csv"), help="path to CSV dataset")
    parser.add_argument("--quick", action="store_true", help="Use saved best params from config/best_params.json (no grid search)")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    proc = DataProcessor(args.data)
    df = proc.preprocess()
    trainer = ChurnTrainer(df, model_dir=args.model_dir, results_dir=args.results_dir)
    trainer.run_all(train_xgb=True, train_lgbm=True, train_dt=True, build_voting=True, quick=args.quick)

#!/usr/bin/env python3
"""

Usage:
  python main.py train    [--quick]
  python main.py evaluate [--model MODELPATH]
  python main.py pytest   [--pytest-args "args"]

Only three high-level commands are supported: train, evaluate, pytest.
"""
import argparse
import sys
import subprocess

from src.load_data import DataProcessor
from src.train import ChurnTrainer
from src.test import ModelEvaluator


def cmd_train(args):
    proc = DataProcessor(args.input)
    df = proc.preprocess()
    trainer = ChurnTrainer(df=df, random_state=args.random_state, test_size=args.test_size)
    trainer.run_all(train_xgb=True, train_lgbm=True, train_dt=True, build_voting=True, quick=args.quick)
    print("Training finished.")


def cmd_evaluate(args):
    proc = DataProcessor(args.input)
    df = proc.preprocess()
    trainer = ChurnTrainer(df=df, random_state=args.random_state, test_size=args.test_size)
    trainer.split()
    X_test, y_test = trainer.X_test, trainer.y_test

    evaluator = ModelEvaluator(args.model)
    evaluator.load_model()
    metrics = evaluator.evaluate(X_test, y_test, plot_cm=False)
    print(metrics.to_string())


def cmd_pytest(args):
    # run pytest; allow passing extra args string
    extra = []
    if args.pytest_args:
        extra = args.pytest_args.split()
    subprocess.run([sys.executable, "-m", "pytest", "-q"] + extra, check=False)


def build_parser():
    p = argparse.ArgumentParser(prog="cicd", description="Minimal churn CLI")
    p.add_argument("--input", default="data/dataset.csv", help="Raw dataset path")
    p.add_argument("--random-state", type=int, default=42, help=argparse.SUPPRESS)
    p.add_argument("--test-size", type=float, default=0.2, help=argparse.SUPPRESS)

    sub = p.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train all models")
    train_p.add_argument("--quick", action="store_true", help="Use smaller grids for faster run")
    train_p.set_defaults(func=cmd_train)

    eval_p = sub.add_parser("evaluate", help="Evaluate saved model on test split")
    eval_p.add_argument("--model", default="models/model.skops", help="Model file (skops)")
    eval_p.set_defaults(func=cmd_evaluate)

    py_p = sub.add_parser("pytest", help="Run test suite with pytest")
    py_p.add_argument("--pytest-args", default="", help="Additional args for pytest (quoted)")
    py_p.set_defaults(func=cmd_pytest)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
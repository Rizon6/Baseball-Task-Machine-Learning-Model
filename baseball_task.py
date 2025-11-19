import numpy as np
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
import sys

def main():
    load_data()
    args = build_parser().parse_args()
    """if args.mode == 'train':
        training_mode(args.x, args.y, args.model, args.max_steps, args.order)
    elif args.mode == 'predict':
        prediction_mode(args.x, args.model, args.order, args.preds)
    elif args.mode == 'eval':
        evaluation_mode(args.x, args.y, args.model, args.order)"""

def _mirror_for_help(action: argparse.Action, *groups: argparse._ArgumentGroup) -> None:
    """Make an existing action appear in additional help groups without re-registering it."""
    for g in groups:
        # Only affect help rendering; don't trigger conflict checks.
        if action not in g._group_actions:  # avoid duplicates if called twice
            g._group_actions.append(action)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # Mutually exclusive mode selection (exactly one)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--train",   dest="mode", action="store_const", const="train",
                       help="Train a model from x.txt and y.txt.")
    modes.add_argument("--predict", dest="mode", action="store_const", const="predict",
                       help="Run prediction on a dev set using a trained model.")
    modes.add_argument("--eval",    dest="mode", action="store_const", const="eval",
                       help="Evaluate a trained model on a dev set (prints MSE).")

    # Define each option once
    order     = p.add_argument("--order", type=int, default=1,
                               help="Polynomial order (1 for linear model, default 1).")
    x_arg     = p.add_argument("--x", help="Path to inputs x.txt.")
    y_arg     = p.add_argument("--y", help="Path to targets y.txt.")
    model     = p.add_argument("--model",  help="Path to model file to be written (train) or read (predict/eval).")
    preds     = p.add_argument("--preds", help="Path to write predictions (predict).")
    lr        = p.add_argument("--lr", type=float,
                               help="Learning rate for gradient descent (train).")
    max_steps = p.add_argument("--max-steps", type=int, default=0,
                               help="Max gradient steps; 0 means analytical solution (train), default 0.")

    # Help-only groups
    g_train = p.add_argument_group("train mode arguments (used with --train)")
    g_pred  = p.add_argument_group("prediction mode arguments (used with --predict)")
    g_eval  = p.add_argument_group("evaluation mode arguments (used with --eval)")

    # Mirror actions into the groups for DISPLAY ONLY
    _mirror_for_help(order, g_train, g_pred, g_eval)
    _mirror_for_help(x_arg,     g_train, g_pred, g_eval)
    _mirror_for_help(y_arg,     g_train, g_eval)
    _mirror_for_help(model,     g_train, g_pred, g_eval)
    _mirror_for_help(preds,     g_pred)
    _mirror_for_help(lr,        g_train)
    _mirror_for_help(max_steps, g_train)

    return p

def build_model(alpha = 1):
    return Pipeline ([
        ("imputer", SimpleImputer(missing_values = -1, strategy = "mean"))
        ("scaler", StandardScaler())
        ("model", Ridge(alpha = alpha))
    ])

def load_data(X_path, Y_path):
    x = np.loadtxt(X_path)
    y = np.loadtxt(Y_path)
    return x, y

def training_mode(x_path, y_path, model_path):
    x_train, y_train = load_data(x_path, y_path)
    model = build_model()
    model.fit(x_train, y_train)
    joblib.dump(model, model_path)

def prediction_mode(x_path, model_path, pred_path = None):
    x, y = load_data(x_path, y_path)
    model = joblib.load(model_path)
    y_predictions = model.predict()
    

if __name__ == "__main__":
    main()

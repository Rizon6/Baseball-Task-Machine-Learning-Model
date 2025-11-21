import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import argparse
import sys

def main():
    args = build_parser().parse_args()
    if args.mode == 'train':
        training_mode(args.trainx, args.trainy, args.model, args.alpha)
    elif args.mode == 'predict':
        prediction_mode(args.devx, args.model, args.preds)
    elif args.mode == 'eval':
        evaluation_mode(args.devx, args.devy, args.model)

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
    trainx = p.add_argument("--trainx", help="Path to training inputs npy file.")
    trainy = p.add_argument("--trainy", help="Path to training targets npy file (class labels).")
    devx   = p.add_argument("--devx", help="Path to dev inputs npy file.")
    devy   = p.add_argument("--devy", help="Path to dev targets npy file (class labels).")
    model  = p.add_argument("--model",  help="Path to model file to be written (train) or read (predict/eval).")
    preds  = p.add_argument("--preds", help="Path to write predictions (predict).")
    alpha  = p.add_argument("--alpha", type=float, default=1.0, help="alpha value to use to train models")

    # Help-only groups
    g_train = p.add_argument_group("train mode arguments (used with --train)")
    g_pred  = p.add_argument_group("prediction mode arguments (used with --predict)")
    g_eval  = p.add_argument_group("evaluation mode arguments (used with --eval)")

    # Mirror actions into the groups for DISPLAY ONLY
    _mirror_for_help(trainx, g_train)
    _mirror_for_help(trainy, g_train)
    _mirror_for_help(devx,   g_train, g_pred, g_eval)
    _mirror_for_help(devy,   g_train, g_eval)
    _mirror_for_help(model,     g_train, g_pred, g_eval)
    _mirror_for_help(preds,     g_pred)
    _mirror_for_help(alpha,     g_train)

    return p

def build_model(model_type, alpha):
    if model_type == "ridge":
        return Pipeline ([
            ("imputer", SimpleImputer(missing_values = -1, strategy = "mean")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha = alpha))
        ])
    if model_type == "lasso":
        return Pipeline ([
            ("imputer", SimpleImputer(missing_values = -1, strategy = "mean")),
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha = alpha))
        ])
    if model_type == "elastic":
        return Pipeline ([
            ("imputer", SimpleImputer(missing_values = -1, strategy = "mean")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha = alpha, l1_ratio = 0.5))
        ])

def load_xy(x_path, y_path):
    x = np.loadtxt(x_path)
    y = np.loadtxt(y_path)
    return x, y

def training_mode(x_path, y_path, model_path, alpha):
    x_train, y_train = load_xy(x_path, y_path)
    model = build_model(alpha)
    model.fit(x_train, y_train)
    joblib.dump(model, model_path)

def prediction_mode(x_path, model_path, pred_path = None):
    x = np.loadtxt(x_path)
    model = joblib.load(model_path)
    y_predictions = model.predict(x)

    if pred_path:
        joblib.dump(y_predictions, pred_path)
    return y_predictions


def evaluation_mode(x_path, y_path, model_path):
    y = np.loadtxt(y_path)
    y_pred = prediction_mode(x_path, model_path)
    mse = mean_squared_error(y, y_pred)
    print(mse)


    

if __name__ == "__main__":
    main()

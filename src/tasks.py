from invoke import task
import os
from datetime import datetime
import glob

# Default configurations
DEFAULT_DATA_DIR = "data"
DEFAULT_MODELS_DIR = "models"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_PRED_FILE = "predictions.csv"

# Model configurations with their specific parameters and defaults
MODEL_CONFIGS = {
    'lightgbm': {
        'script': 'train_lgb.py',
        'params': {
            'num_leaves': ('--num-leaves', 31),
            'learning_rate': ('--learning-rate', 0.1),
            'n_estimators': ('--n-estimators', 100)
        }
    },
    'xgboost': {
        'script': 'train_xgb.py',
        'params': {
            'max_depth': ('--max-depth', 6),
            'learning_rate': ('--learning-rate', 0.1),
            'n_estimators': ('--n-estimators', 100)
        }
    },
    'decision_tree': {
        'script': 'train_dt.py',
        'params': {
            'max_depth': ('--max-depth', 10),
            'min_samples_split': ('--min-samples-split', 2),
            'min_samples_leaf': ('--min-samples-leaf', 1)
        }
    },
    'logistic': {
        'script': 'train_logistic.py',
        'params': {
            'max_iter': ('--max-iter', 100),
            'C': ('--C', 1.0)
        }
    },
    'adaboost': {
        'script': 'train_adaboost.py',
        'params': {
            'n_estimators': ('--n-estimators', 100),
            'learning_rate': ('--learning-rate', 0.1)
        }
    }
}


def get_available_models():
    """Get list of available models based on existing training scripts"""
    available_models = []
    for model_type, config in MODEL_CONFIGS.items():
        if os.path.exists(config['script']):
            available_models.append(model_type)
    return available_models


@task
def list_models(ctx):
    """List all available models and their parameters"""
    available_models = get_available_models()
    print("\nAvailable models and their parameters:")
    for model in available_models:
        print(f"\n{model.upper()}:")
        for param, (flag, default) in MODEL_CONFIGS[model]['params'].items():
            print(f"  {param}: {flag} (default: {default})")


@task
def train(ctx, model_type="all", model_dir=DEFAULT_MODELS_DIR, **kwargs):
    """
    Train the specified model(s) with custom parameters

    Usage examples:
    invoke train --model-type decision_tree --max-depth 15 --min-samples-split 5
    invoke train --model-type lightgbm --num-leaves 63 --learning-rate 0.05
    """
    available_models = get_available_models()

    if not available_models:
        raise ValueError("No training scripts found!")

    if model_type.lower() not in ["all"] + available_models:
        raise ValueError(f"Invalid model type. Available options: {['all'] + available_models}")

    print(f"Running training step for {model_type}...")
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    models_to_train = available_models if model_type.lower() == "all" else [model_type.lower()]

    for model in models_to_train:
        print(f"\nTraining {model} model...")
        config = MODEL_CONFIGS[model]

        # Build command with default parameters and any provided overrides
        cmd_params = []
        for param_name, (flag, default) in config['params'].items():
            # Use provided value if available, otherwise use default
            value = kwargs.get(param_name, default)
            cmd_params.append(f"{flag} {value}")

        cmd = (
            f"python {config['script']} "
            f"--model-name {model}_{timestamp} "
            f"{' '.join(cmd_params)} "
            f"--output-dir {model_dir}"
        )

        print(f"Running command: {cmd}")
        ctx.run(cmd)


@task
def run_pipeline(ctx, model_type="all", per_dataset=False, **kwargs):
    """
    Run the complete pipeline with model-specific parameters

    Usage example:
    invoke run-pipeline --model-type decision_tree --max-depth 15 --min-samples-split 5
    """
    print("Running complete pipeline...")
    preprocess(ctx)
    train(ctx, model_type=model_type, **kwargs)
    predict(ctx)
    analyze(ctx, per_dataset=per_dataset)

# ... [rest of the tasks file remains the same: preprocess, predict, analyze, clean tasks] ...
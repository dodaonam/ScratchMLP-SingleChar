import argparse
import numpy as np
import optuna
import os
import json
from optuna.exceptions import TrialPruned
from src.model import MLP
from src.load_data import load_processed_data
from joblib import Parallel, delayed

optuna.logging.set_verbosity(optuna.logging.WARNING)

def train(mlp, X_train, Y_train, X_val, Y_val, epochs, batch_size, patience, lr, decay_rate, trial):
    m = X_train.shape[1]
    best_val_loss = float('inf')
    val_acc_at_best_loss = 0.0

    wait = 0
    for epoch in range(epochs):
        mlp.lr = lr * (decay_rate ** epoch)
        perm = np.random.permutation(m)
        X_train_shuffled = X_train[:, perm]
        Y_train_shuffled = Y_train[:, perm]
        
        for i in range(0, m, batch_size):
            X_batch = X_train_shuffled[:, i:i+batch_size]
            Y_batch = Y_train_shuffled[:, i:i+batch_size]
            
            AL, caches = mlp.forward(X_batch, is_training=True)
            grads = mlp.backward(AL, Y_batch, caches)
            mlp.update_parameters(grads)

        AL_val, _ = mlp.forward(X_val, is_training=False)
        val_loss = mlp.cost(AL_val, Y_val)
        preds_val = np.argmax(AL_val, axis=0)
        true_val = np.argmax(Y_val, axis=0) 
        val_acc = np.mean(preds_val == true_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            val_acc_at_best_loss = val_acc
            best_weights = {k: v.copy() for k, v in mlp.parameters.items()}
            if mlp.use_batchnorm:
                best_bn = {k: v.copy() for k, v in mlp.bn_params.items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

        trial.report(val_acc, epoch)

        if trial.should_prune():
            raise TrialPruned()
        
    return val_acc_at_best_loss, best_weights, best_bn

def objective(trial):
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    lambd = trial.suggest_float('lambda', 1e-4, 1e-1, log=True)
    keep_prob = trial.suggest_float('keep_prob', 0.3, 0.7, step=0.1)
    n_hidden_layers = trial.suggest_int('n_hidden_layers', 1, 2)

    layer_dims = [784]
    for i in range(n_hidden_layers):
        n_neurons = trial.suggest_int(f'n_units_layer_{i}', 128, 1024, step=128)
        layer_dims.append(n_neurons)
    layer_dims.append(47)
    
    EPOCHS = 50
    BATCH_SIZE = 128
    PATIENCE = 5
    DECAY_RATE = 0.98
    OPTIMIZER = 'adam'

    mlp = MLP(layer_dims=layer_dims, init='he', lr=lr, lambd=lambd, 
              keep_prob=keep_prob, use_batchnorm=True, optimizer=OPTIMIZER)

    try:
        best_val_acc, best_weights, best_bn = train(mlp, X_train, Y_train, X_val, Y_val, 
                             epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE, 
                             lr=lr, decay_rate=DECAY_RATE, trial=trial)
        if best_weights:
            best_weights = {k: v.tolist() for k, v in best_weights.items()}
            trial.set_user_attr("best_weights", best_weights)
        if best_bn:
            best_bn = {k: v.tolist() for k, v in best_bn.items()}
            trial.set_user_attr("best_bn_params", best_bn)
    except TrialPruned:
        raise
    except Exception as e:
        print(f"Trial #{trial.number} failed with an error: {e}")
        return 0.0

    return best_val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for MLP")
    parser.add_argument("--trials", type=int, default=50, help="Total number of Optuna trials to run (default: 50)")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel workers to use for tuning (default: 1)")
    args = parser.parse_args()

    STORAGE = "postgresql+psycopg2://optuna:mypass@localhost:5432/optuna_db?connect_timeout=30"
    STUDY_NAME = "mlp_hyperparam_tuning"

    def _worker_run(n_chunk, storage_url, study_name):
        X_tr, Y_tr, X_v, Y_v, _, _ = load_processed_data(validation_split=0.1)
        globals()['X_train'] = X_tr
        globals()['Y_train'] = Y_tr
        globals()['X_val'] = X_v
        globals()['Y_val'] = Y_v

        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)

        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            pruner=pruner,
            storage=storage_url,
            load_if_exists=True
        )

        study.optimize(objective, n_trials=n_chunk, n_jobs=1)

        return True

    if args.jobs <= 1:
        X_train, Y_train, X_val, Y_val, _, _ = load_processed_data(validation_split=0.1)

        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)

        study = optuna.create_study(
            study_name=STUDY_NAME,
            direction='maximize',
            pruner=pruner,
            storage=STORAGE,
            load_if_exists=True
        )

        study.optimize(objective, n_trials=args.trials, n_jobs=1)
    else:
        n_total = args.trials
        n_workers = args.jobs
        base = n_total // n_workers
        extras = n_total % n_workers

        chunks = [base + (1 if i < extras else 0) for i in range(n_workers)]
        chunks = [c for c in chunks if c > 0]

        print(f"Launching {len(chunks)} workers via joblib with chunks: {chunks}")

        with Parallel(n_jobs=len(chunks)) as parallel:
            parallel(delayed(_worker_run)(c, STORAGE, STUDY_NAME) for c in chunks)

        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)

    print("\n\n==================================================")
    print(f"Total trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    
    print(f"\nBest Trial: #{best_trial.number}")
    print(f"Best Validation Accuracy: {best_trial.value:.4f}")

    try:
        best_weights = best_trial.user_attrs["best_weights"]
        weights_to_save = {k: np.array(v) for k, v in best_weights.items()}
        if "best_bn_params" in best_trial.user_attrs:
            best_bn = best_trial.user_attrs["best_bn_params"]
            bn_to_save = {k: np.array(v) for k, v in best_bn.items()}
            weights_to_save.update(bn_to_save)

        os.makedirs("models", exist_ok=True)
        np.savez('models/mlp_weights_tuned.npz', **weights_to_save)
        print("Best tuned weights saved to models/mlp_weights_tuned.npz")
    except KeyError:
        print("Could not find saved weights in the best trial.")
    
    print("\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    with open('models/best_params.json', 'w') as f:
        json.dump(best_trial.params, f, indent=4)
    print("Best hyperparameters saved to models/best_params.json")
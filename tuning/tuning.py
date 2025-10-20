import numpy as np
import optuna
from optuna.exceptions import TrialPruned
from src.model import MLP
from src.load_data import load_processed_data

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
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

        trial.report(val_acc, epoch)

        if trial.should_prune():
            raise TrialPruned()
        
    return val_acc_at_best_loss

def objective(trial):
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    lambd = trial.suggest_float('lambda', 1e-5, 1e-2, log=True)
    keep_prob = trial.suggest_float('keep_prob', 0.5, 0.9)
    n_hidden_layers = trial.suggest_int('n_hidden_layers', 1, 3)

    layer_dims = [784]
    for i in range(n_hidden_layers):
        n_neurons = trial.suggest_int(f'n_units_layer_{i}', 128, 512, step=128)
        layer_dims.append(n_neurons)
    layer_dims.append(47)

    mlp = MLP(layer_dims=layer_dims, init='he', lr=lr, lambd=lambd, 
              keep_prob=keep_prob, use_batchnorm=True, optimizer='adam')
    
    EPOCHS = 30
    BATCH_SIZE = 128
    PATIENCE = 5
    DECAY_RATE = 0.98

    try:
        best_val_acc = train(mlp, X_train, Y_train, X_val, Y_val, 
                             epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE, 
                             lr=lr, decay_rate=DECAY_RATE, trial=trial)
    except TrialPruned:
        print(f"Trial #{trial.number+1} pruned.")
        raise
    except Exception as e:
        print(f"Trial #{trial.number+1} failed with an error: {e}")
        return 0.0

    return best_val_acc

if __name__ == "__main__":
    X_train, Y_train, X_val, Y_val, _, _ = load_processed_data(validation_split=0.1)
    
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
    
    study = optuna.create_study(
        study_name="mlp_hyperparam_tuning",
        direction='maximize',
        pruner=pruner,
        storage="sqlite:///tuning_results.db",
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=2, n_jobs=-1)

    print("\n\n==================================================")
    print(f"Total trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    
    print(f"\nBest Trial: #{best_trial.number+1}")
    print(f"Best Validation Accuracy: {best_trial.value:.4f}")
    
    print("\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
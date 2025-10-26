import optuna
import numpy as np
import os
import matplotlib.pyplot as plt
from src.model import MLP
from src.load_data import load_processed_data

EPOCHS = 10
BATCH_SIZE = 128
PATIENCE = 3
LEARNING_RATE_DECAY = 0.98
OPTIMIZER = 'adam'

def train_best_model(mlp, X_train, Y_train, X_val, Y_val, epochs, batch_size, patience, lr, decay_rate):
    m = X_train.shape[1]
    best_val_loss = float('inf')
    best_weights = None
    best_bn = None
    history = {
        'train_acc': [], 
        'train_loss': [],
        'val_acc': [], 
        'val_loss': []
    }
    
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

        AL_train, _ = mlp.forward(X_train, is_training=False)
        train_loss = mlp.cost(AL_train, Y_train)
        preds_train = np.argmax(AL_train, axis=0)
        true_train = np.argmax(Y_train, axis=0)
        train_acc = np.mean(preds_train == true_train)
        
        AL_val, _ = mlp.forward(X_val, is_training=False)
        val_loss = mlp.cost(AL_val, Y_val)
        preds_val = np.argmax(AL_val, axis=0)
        true_val = np.argmax(Y_val, axis=0) 
        val_acc = np.mean(preds_val == true_val)

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: train_acc: {train_acc:.4f}, train_loss: {train_loss:.4f} - val_acc: {val_acc:.4f}, val_loss: {val_loss:.4f} - lr: {mlp.lr:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.copy() for k, v in mlp.parameters.items()}
            if mlp.use_batchnorm:
                 best_bn = {k: v.copy() for k, v in mlp.bn_params.items()}
            print(f"Best validation loss: {best_val_loss:.4f}")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_weights is not None:
        mlp.parameters = best_weights
        if mlp.use_batchnorm and best_bn is not None:
            mlp.bn_params = best_bn
        print(f"\nRestored best model with validation loss: {best_val_loss:.4f}")
    
    return history

def plot_training_history(history, filename="training_history_tuned.png"):
    epochs_range = range(1, len(history['train_acc']) + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_acc'], 'o-', label='Train Accuracy')
    plt.plot(epochs_range, history['val_acc'], 'o-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_loss'], 'o-', label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], 'o-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.suptitle('Tuned Model Training History', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/training_history_tuned.png")
    print(f"\nTraining history plot saved to output/training_history_tuned.png")
    plt.show()

if __name__ == "__main__":
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_processed_data(validation_split=0.1)
    
    try:
        study = optuna.load_study(
            study_name="mlp_hyperparam_tuning",
            storage="postgresql+psycopg2://optuna:mypass@localhost:5432/optuna_db"
        )
        best_params = study.best_trial.params
        print(best_params)
    except Exception as e:
        print(f"Error: Unable to load the Optuna study. ({e})")
        exit()

    n_hidden_layers = best_params['n_hidden_layers']
    layer_dims = [784]
    for i in range(n_hidden_layers):
        layer_dims.append(best_params[f'n_units_layer_{i}'])
    layer_dims.append(47)

    print("\nTrain model with best hyperparameters")
    final_mlp = MLP(
        layer_dims=layer_dims,
        init='he',
        lr=best_params['learning_rate'],
        lambd=best_params['lambda'],
        keep_prob=best_params['keep_prob'],
        use_batchnorm=True,
        optimizer=OPTIMIZER
    )

    try:
        weights = np.load('models/mlp_weights_tuned.npz')
        loaded_params = {}
        loaded_bn_params = {}

        for key in weights.keys():
            if key.startswith('gamma') or key.startswith('beta') or key.startswith('running_'):
                loaded_bn_params[key] = weights[key]
            elif key.startswith('W') or key.startswith('b'):
                loaded_params[key] = weights[key]
        
        final_mlp.parameters = loaded_params
        if final_mlp.use_batchnorm:
            final_mlp.bn_params = loaded_bn_params
        print("Weights loaded successfully. Resuming training.")
    except FileNotFoundError:
        print("Warning: 'models/mlp_weights_tuned.npz' not found. Training from scratch.")
    except Exception as e:
        print(f"Error loading weights: {e}. Training from scratch.")

    history = train_best_model(
        final_mlp, X_train, Y_train, X_val, Y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        lr=best_params['learning_rate'],
        decay_rate=LEARNING_RATE_DECAY
    )

    AL_test, _ = final_mlp.forward(X_test, is_training=False)
    test_loss = final_mlp.cost(AL_test, Y_test)
    preds_test = np.argmax(AL_test, axis=0)
    true_test = np.argmax(Y_test, axis=0)
    test_acc = np.mean(preds_test == true_test)

    print(f"\nFinal Results on Test Set:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    os.makedirs("models", exist_ok=True)
    to_save = dict(final_mlp.parameters)
    if final_mlp.use_batchnorm:
        to_save.update(final_mlp.bn_params)
    np.savez('models/mlp_final_weights.npz', **to_save)
    print(f"\nModel saved to models/mlp_final_weights.npz")

    plot_training_history(history)
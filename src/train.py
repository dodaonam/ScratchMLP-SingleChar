import numpy as np
import os
import matplotlib.pyplot as plt
from model import MLP
from load_data import load_processed_data

EPOCHS = 60
BATCH_SIZE = 64
PATIENCE = 5
VALIDATION_SPLIT = 0.1
LAYER_DIMS = [784, 512, 256, 47]
LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.95
LAMBDA = 0.005
KEEP_PROB = 0.5
USE_BATCHNORM = True
MOMENTUM = 0.9
OPTIMIZER = 'adam'

def train(mlp, X_train, Y_train, X_val, Y_val, epochs, batch_size, patience, lr, decay_rate):
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
        mlp.keep_prob = KEEP_PROB
        perm = np.random.permutation(m)
        X_train = X_train[:, perm]
        Y_train = Y_train[:, perm]
        
        for i in range(0, m, batch_size):
            X_batch = X_train[:, i:i+batch_size]
            Y_batch = Y_train[:, i:i+batch_size]
            
            AL, caches = mlp.forward(X_batch, is_training=True)
            grads = mlp.backward(AL, Y_batch, caches)
            mlp.update_parameters(grads)

        mlp.keep_prob = 1.0

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
        
        print(f"epoch {epoch+1}/{epochs}: train_acc: {train_acc:.4f}, train_loss: {train_loss:.4f} - val_acc: {val_acc:.4f}, val_loss: {val_loss:.4f} - lr: {mlp.lr:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {key: value.copy() for key, value in mlp.parameters.items()}
            best_bn = ({k: v.copy() for k, v in mlp.bn_params.items()} 
                    if mlp.use_batchnorm else None)
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
        mlp.keep_prob = 1.0
        print(f"\nRestored best model with validation loss: {best_val_loss:.4f}")
    
    return history, best_val_loss

def plot_training_history(history):
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
    
    plt.suptitle('Training History', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/training_history_regularization.png")
    print("\nTraining history plot saved to output/training_history_regularization.png")
    
    plt.show()

if __name__ == "__main__":
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_processed_data(validation_split=VALIDATION_SPLIT)
    mlp = MLP(layer_dims=LAYER_DIMS, init='he', lr=LEARNING_RATE, lambd=LAMBDA, 
              keep_prob=KEEP_PROB, use_batchnorm=USE_BATCHNORM, momentum=MOMENTUM, optimizer=OPTIMIZER)

    history, best_val_loss = train(mlp, X_train, Y_train, X_val, Y_val, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                                   patience=PATIENCE, lr=LEARNING_RATE, decay_rate=LEARNING_RATE_DECAY)

    AL_test, _ = mlp.forward(X_test, is_training=False)
    test_loss = mlp.cost(AL_test, Y_test)
    preds_test = np.argmax(AL_test, axis=0)
    true_test = np.argmax(Y_test, axis=0)
    test_acc = np.mean(preds_test == true_test)

    print(f"\nFinal Results on Test Set:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    os.makedirs("models", exist_ok=True)
    to_save = dict(mlp.parameters)
    if mlp.use_batchnorm:
        to_save.update(mlp.bn_params)
    np.savez('models/mlp_weights.npz', **to_save)
    print("Model saved to models/mlp_weights.npz")
    
    plot_training_history(history)
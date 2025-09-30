import numpy as np
import os
from model import MLP
from load_data import load_processed_data

X_train, Y_train, X_val, Y_val, X_test, Y_test = load_processed_data(validation_split=0.1)

layer_dims = [784, 512, 256, 47]
mlp = MLP(layer_dims=layer_dims, init='he', lr=0.01)

def train(mlp, X_train, Y_train, X_val, Y_val, epochs=10, batch_size=64):
    m = X_train.shape[1]
    best_val_acc = 0
    best_weights = None
    
    for epoch in range(epochs):
        perm = np.random.permutation(m)
        X_train = X_train[:, perm]
        Y_train = Y_train[:, perm]
        
        total_loss = 0
        for i in range(0, m, batch_size):
            X_batch = X_train[:, i:i+batch_size]
            Y_batch = Y_train[:, i:i+batch_size]
            
            AL, caches = mlp.forward(X_batch)
            loss = mlp.cost(AL, Y_batch)
            total_loss += loss * X_batch.shape[1]
            
            grads = mlp.backward(AL, Y_batch, caches)
            mlp.update_parameters(grads)
        
        avg_loss = total_loss / m
        
        AL_val, _ = mlp.forward(X_val)
        preds_val = np.argmax(AL_val, axis=0)
        true_val = np.argmax(Y_val, axis=0)
        val_acc = np.mean(preds_val == true_val)
        
        print(f"epoch {epoch+1}/{epochs}, val_acc: {val_acc:.4f}, loss: {avg_loss:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {key: value.copy() for key, value in mlp.parameters.items()}
            print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    if best_weights is not None:
        mlp.parameters = best_weights
        print(f"\nRestored best model with validation accuracy: {best_val_acc:.4f}")
    
    return best_val_acc

best_val_acc = train(mlp, X_train, Y_train, X_val, Y_val)

AL_test, _ = mlp.forward(X_test)
preds_test = np.argmax(AL_test, axis=0)
true_test = np.argmax(Y_test, axis=0)
test_acc = np.mean(preds_test == true_test)

print(f"\nFinal Results:")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

os.makedirs("models", exist_ok=True)
np.savez('models/mlp_weights.npz', **mlp.parameters)
print("Model saved to models/mlp_weights.npz")
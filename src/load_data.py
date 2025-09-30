import numpy as np
import os
from torchvision import datasets, transforms

def load_emnist_balanced(validation_split):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.13,), (0.30,))])
    train_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
    test_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)
    
    X_train = np.array([img.flatten().numpy() for img, _ in train_dataset])
    y_train = np.array([label for _, label in train_dataset])
    X_test = np.array([img.flatten().numpy() for img, _ in test_dataset])
    y_test = np.array([label for _, label in test_dataset])
    
    num_samples = len(X_train)
    num_val = int(num_samples * validation_split)
    
    np.random.seed(123)
    indices = np.random.permutation(num_samples)
    
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]
    
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]
    
    X_train = X_train.T
    X_val = X_val.T
    X_test = X_test.T
    
    num_classes = len(np.unique(y_train))
    Y_train = np.eye(num_classes)[y_train].T
    Y_val = np.eye(num_classes)[y_val].T
    Y_test = np.eye(num_classes)[y_test].T

    os.makedirs("data/processed", exist_ok=True)
    np.save('data/processed/train_X.npy', X_train)
    np.save('data/processed/train_y.npy', Y_train)
    np.save('data/processed/val_X.npy', X_val)
    np.save('data/processed/val_y.npy', Y_val)
    np.save('data/processed/test_X.npy', X_test)
    np.save('data/processed/test_y.npy', Y_test)
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def load_processed_data(validation_split=0.1):
    try:
        X_train = np.load('data/processed/train_X.npy')
        Y_train = np.load('data/processed/train_y.npy')
        X_val = np.load('data/processed/val_X.npy')
        Y_val = np.load('data/processed/val_y.npy')
        X_test = np.load('data/processed/test_X.npy')
        Y_test = np.load('data/processed/test_y.npy')
        
        return X_train, Y_train, X_val, Y_val, X_test, Y_test
    
    except (FileNotFoundError, OSError):
        return load_emnist_balanced(validation_split)
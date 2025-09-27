import numpy as np

class MLP:
    def __init__(self, layer_dims, init='he', lr=1e-2):
        self.layer_dims = layer_dims
        self.init = init
        self.lr = lr
        self.parameters = self.initialize_parameters(layer_dims, init)

    def initialize_parameters(self, layer_dims, init='he'):
        params = {}
        for l in range(1, len(layer_dims)):
            n_in = layer_dims[l-1]
            n_out = layer_dims[l]
            if init == 'he':    
                params[f'W{l}'] = np.random.randn(n_out, n_in) * np.sqrt(2.0/n_in)
            else:
                params[f'W{l}'] = np.random.randn(n_out, n_in) * 0.01
            params[f'b{l}'] = np.zeros((n_out, 1))
        return params
    
    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)
    
    @staticmethod
    def softmax(Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
    
    def forward(self, X):
        caches = []
        A = X
        L = len(self.parameters) // 2

        for l in range(1, L):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = np.dot(W, A) + b
            A_next = self.relu(Z)
            caches.append({'A_prev': A, 'W': W, 'b': b, 'Z': Z, 'act': 'relu'})
            A = A_next
        
        W = self.parameters[f'W{L}']
        b = self.parameters[f'b{L}']
        ZL = np.dot(W, A) + b
        AL = self.softmax(ZL)
        caches.append({'A_prev': A, 'W': W, 'b': b, 'Z': ZL, 'act': 'softmax'})

        return AL, caches
    
    def cost(self, AL, Y):
        m = Y.shape[1]
        return float(np.squeeze(-np.sum(Y * np.log(AL + 1e-8)) / m))
    
    @staticmethod
    def relu_backward(dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ
    
    def backward(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        current_cache = caches[-1]
        dZ = AL - Y
        dW = np.dot(dZ, current_cache['A_prev'].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA = np.dot(current_cache['W'].T, dZ)

        grads[f'dW{L}'] = dW      
        grads[f'db{L}'] = db

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dZ = self.relu_backward(dA, current_cache['Z'])
            dW = np.dot(dZ, current_cache['A_prev'].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dA = np.dot(current_cache['W'].T, dZ)

            grads[f'dW{l+1}'] = dW
            grads[f'db{l+1}'] = db

        return grads
    
    def update_parameters(self, grads):
        L = len(self.parameters) // 2
        
        for l in range(1, L+1):
            self.parameters[f'W{l}'] -= self.lr * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= self.lr * grads[f'db{l}']
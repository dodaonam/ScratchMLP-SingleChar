import numpy as np

class MLP:
    def __init__(self, layer_dims, init='he', lr=1e-2, lambd=0.0, keep_prob=1.0, use_batchnorm=False, momentum=0.9):
        self.layer_dims = layer_dims
        self.init = init
        self.lr = lr
        self.lambd = lambd
        self.keep_prob = keep_prob
        self.use_batchnorm = use_batchnorm
        self.momentum = momentum
        self.parameters = self.initialize_parameters(layer_dims, init)

        if self.use_batchnorm:
            self.bn_params = self.initialize_bn_parameters(layer_dims)

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
    
    def initialize_bn_parameters(self, layer_dims):
        bn_params = {}
        for l in range(1, len(layer_dims)-1):
            bn_params[f'gamma{l}'] = np.ones((layer_dims[l], 1))
            bn_params[f'beta{l}'] = np.zeros((layer_dims[l], 1))
            bn_params[f'running_mean{l}'] = np.zeros((layer_dims[l], 1))
            bn_params[f'running_var{l}'] = np.ones((layer_dims[l], 1))
        return bn_params
    
    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)
    
    @staticmethod
    def softmax(Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
    
    def forward(self, X, is_training):
        caches = []
        A = X
        L = len(self.parameters) // 2

        for l in range(1, L):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = np.dot(W, A) + b
            if self.use_batchnorm:
                gamma = self.bn_params[f'gamma{l}']
                beta = self.bn_params[f'beta{l}']
                if is_training:
                    mu = np.mean(Z, axis=1, keepdims=True)
                    var = np.var(Z, axis=1,keepdims=True)
                    inv_std = 1/ np.sqrt(var + 1e-8)
                    Z_norm = (Z - mu) * inv_std
                    self.bn_params[f'running_mean{l}'] = self.momentum*self.bn_params[f'running_mean{l}'] + (1-self.momentum)*mu
                    self.bn_params[f'running_var{l}'] = self.momentum*self.bn_params[f'running_var{l}'] + (1-self.momentum)*var

                    bn_cache = (Z_norm, Z-mu, inv_std, gamma)
                else:
                    mu = self.bn_params[f'running_mean{l}']
                    var = self.bn_params[f'running_var{l}']
                    Z_norm = (Z - mu) / np.sqrt(var + 1e-8)
                    bn_cache = None
                Z = gamma * Z_norm + beta
            else:
                bn_cache = None

            A_next = self.relu(Z)

            # dropout
            if self.keep_prob < 1.0:
                M = (np.random.rand(*A_next.shape) < self.keep_prob) / self.keep_prob
                A_next *= M
            else:
                M = None
            caches.append({'A_prev': A, 'W': W, 'b': b, 'Z': Z, 'M': M, 'bn_cache': bn_cache})
            A = A_next
        
        W = self.parameters[f'W{L}']
        b = self.parameters[f'b{L}']
        ZL = np.dot(W, A) + b
        AL = self.softmax(ZL)
        caches.append({'A_prev': A, 'W': W, 'b': b, 'Z': ZL, 'M': None})

        return AL, caches
    
    def cost(self, AL, Y):
        m = Y.shape[1]
        cross_entropy_loss = float(np.squeeze(-np.sum(Y * np.log(AL + 1e-8)) / m))
        if self.lambd == 0:
            return cross_entropy_loss
        
        # l2 regularization
        L2 = 0
        L = len(self.parameters) // 2
        for l in range(1, L+1):
            L2 += np.sum(np.square(self.parameters[f'W{l}']))
        L2 = (self.lambd / (2*m)) * L2
        return cross_entropy_loss + L2
    
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
        dW = np.dot(dZ, current_cache['A_prev'].T) / m + (self.lambd / m) * current_cache['W']         # l2
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(current_cache['W'].T, dZ)

        grads[f'dW{L}'] = dW      
        grads[f'db{L}'] = db

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA  = dA_prev
            # dropout
            M = current_cache['M']
            if M is not None:
                dA = dA * M

            dZ = self.relu_backward(dA, current_cache['Z'])

            if self.use_batchnorm:
                Z_norm, Z_minus_mu, inv_std, gamma = current_cache['bn_cache']
                grads[f'dgamma{l+1}'] = np.sum(dZ*Z_norm, axis=1, keepdims=True)
                grads[f'dbeta{l+1}'] = np.sum(dZ, axis=1, keepdims=True)
                dZ_norm = dZ * gamma
                dZ = (1/m) * inv_std * (m * dZ_norm - np.sum(dZ_norm, axis=1, keepdims=True)
                                        - Z_norm * np.sum(dZ_norm * Z_norm, axis=1, keepdims=True))
                
            dW = np.dot(dZ, current_cache['A_prev'].T) / m + (self.lambd / m) * current_cache['W']     # l2
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dA_prev = np.dot(current_cache['W'].T, dZ)

            grads[f'dW{l+1}'] = dW
            grads[f'db{l+1}'] = db

        return grads
    
    def update_parameters(self, grads):
        L = len(self.parameters) // 2
        
        for l in range(1, L+1):
            self.parameters[f'W{l}'] -= self.lr * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= self.lr * grads[f'db{l}']
            
        if self.use_batchnorm:
            for l in range(1, L):
                if f'dgamma{l}' in grads:
                    self.bn_params[f'gamma{l}'] -= self.lr * grads[f'dgamma{l}']
                if f'dbeta{l}' in grads:
                    self.bn_params[f'beta{l}']  -= self.lr * grads[f'dbeta{l}']
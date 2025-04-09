import numpy as np

X = np.load('data/sequences/X.npy')
y = np.load('data/sequences/y_both.npy')

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Total values in X: {np.prod(X.shape)}")
print(f"Total NaNs in X: {np.isnan(X).sum()}")
print(f"Total values in y: {np.prod(y.shape)}")
print(f"Total NaNs in y: {np.isnan(y).sum()}")

# inspect_sample.py
import numpy as np

p = 'dataset-Python/transportDiff.npy'  # the working sample
with open(p, 'rb') as f:
    a1 = np.load(f, allow_pickle=True)
    a2 = np.load(f, allow_pickle=True)
    a3 = np.load(f, allow_pickle=True)

print('A1 type/shape:', type(a1), getattr(a1, 'shape', None))
print('A2 type/shape:', type(a2), getattr(a2, 'shape', None))
print('A3 type/shape:', type(a3), getattr(a3, 'shape', None))
print('A2 example:', a2 if not hasattr(a2, 'shape') else 'array')

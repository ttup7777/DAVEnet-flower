import scipy.io
import json
import numpy as np
import torch

# Audiolabel = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,11,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
relevant_size = np.array(40)
# A2I_average_precision = 1.0 * np.zeros_like(Audiolabel)
Imlabel = np.array([1,  1, 19,  1,  3, 14, 14, 17, 17, 13,  4,  3,  9, 14,  4,  2, 14, 14,
        17, 17, 17, 13,  4, 17, 14, 13, 20,  9,  4,  2, 14,  9,  9, 14, 17, 14,
        14,  2,  2, 14, 14,  7, 14, 17, 17, 20, 14,  1,  8, 14])
Audiolabel = np.array([1])

hit_index = np.where(Imlabel == Audiolabel)
# print(hit_index[0])
precision = 1.0 * np.zeros_like(hit_index[0])


for j in range(hit_index[0].shape[0]):
    hitid =  hit_index[0][j]
    print(hitid)
    precision[j] = sum(Imlabel[:hitid+1] == Audiolabel) * 1.0 / (hit_index[0][j] + 1)
print(precision)
# precision = np.array([0.33333334,0.33333334,0.375,0.25,0.16666667,0.19354838, 0.15909091, 0.17021276])
# s = 9.162598848342896
# A2I_average_precision[0] = s * 1.0 / relevant_size




import numpy as np
from keras.utils import to_categorical

data = np.array([0,1,2])
one_hots = to_categorical(data)
print(one_hots)
print(one_hots[1])

from rnn import RNNLayer, Sequential, Dense
import numpy as np

# X = np.array([[1,2], [2,3], [3,4], [4,5]])
X = np.array([[1,2], [2,3]])

model = Sequential()
model.add(RNNLayer(3, 1, 1, True, input_shape=2))
model.add(Dense(4))

model.train(X)
print(model.final_output)
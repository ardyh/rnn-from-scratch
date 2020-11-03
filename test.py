from rnn.rnnlayer import RNNLayer

rnn = RNNLayer(3, input_shape=2)
rnn.input = [1,2]

print("OUTPUT")
print(rnn.run())

print("INPUT WEIGHT")
print(rnn.input_weight)

print("SELF WEIGHT")
print(rnn.self_weight)

print("BIAS WEIGHT")
print(rnn.bias_weight)

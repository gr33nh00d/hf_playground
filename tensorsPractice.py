import tensorflow as tf

# Slice a tensor
# ((16, 256, 256, 3))
# (2, 3)
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
sliced_tensor = tensor[0]

print("Original Tensor:")
print(tensor.shape)

print("\nSliced Tensor:")
print(sliced_tensor)
import tensorflow as tf

# Create a tensor with random values of shape (2, 4, 4, 3)
random_tensor = tf.random.uniform(shape=(2, 4, 4, 3))

print("Random Tensor:")

print(random_tensor.shape)
print("---------------------------------------------------")
print(random_tensor[0].shape)
print(random_tensor[0][0].shape)

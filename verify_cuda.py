import tensorflow as tf

assert tf.test.is_gpu_available()
print(tf.config.list_physical_devices('GPU'))
assert tf.test.is_built_with_cuda()
#%%
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from nn_utilities.nn_loading import loadLatestModels


MOD_PREFIX = "JAV_MULTIPLIERS"

class ModelWrapper(tf.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32, name="x")])
    def forward(self, x):
        """Forward pass subgraph"""
        return self.model(x)
    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32, name="x")])
    def jacobian(self, x):
        """Jacobian subgraph"""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            y = self.model(x)
        return tape.jacobian(y, x)

# Load model
loaded_model = loadLatestModels((MOD_PREFIX, ))[MOD_PREFIX]

# Create a wrapper instance
wrapper = ModelWrapper(loaded_model)

# Save the forward function
forward_func = wrapper.forward.get_concrete_function()
forward_func2 = convert_variables_to_constants_v2(forward_func)
graph_def = forward_func2.graph.as_graph_def()

# Export frozen graph
# See:
# - https://github.com/opencv/opencv/issues/16879#issuecomment-603815872
# - https://github.com/opencv/opencv/issues/16582#issuecomment-603819498
with tf.io.gfile.GFile('frozen_graph.pb', 'wb') as f:
   f.write(graph_def.SerializeToString())

# The below, if I were to try using it again, may require wrapt version <1.15.
# See: https://github.com/tensorflow/tensorflow/issues/59869#issuecomment-1452785730
# tf.saved_model.save(wrapper, "forward_savedmodel", signatures={"serving_default": forward_func})

# # Save the Jacobian function as TFLite
# jacobian_func = wrapper.jacobian.get_concrete_function()
# tf.saved_model.save(wrapper, "jacobian_savedmodel", signatures={"serving_default": jacobian_func})

#%%
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


class ModelExportWrapper(tf.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """Forward pass subgraph"""
        return self.model(x)
    
    def jacobian(self, x):
        """Jacobian subgraph"""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            y = self.model(x)
        unflat_jacobian = tape.jacobian(y, x)
        return tf.reshape(unflat_jacobian, [-1, tf.shape(unflat_jacobian)[-1]])


    def save_func(self, func, fname):
        tfunc = tf.function(func, input_signature=[tf.TensorSpec([None, None], tf.float32, name="x")])
        forward_func = tfunc.get_concrete_function()
        forward_func2 = convert_variables_to_constants_v2(forward_func)
        graph_def = forward_func2.graph.as_graph_def()

        # Export frozen graph
        # See:
        # - https://github.com/opencv/opencv/issues/16879#issuecomment-603815872
        # - https://github.com/opencv/opencv/issues/16582#issuecomment-603819498
        with tf.io.gfile.GFile(fname, 'wb') as f:
            f.write(graph_def.SerializeToString())
    
    def save_forward(self, fname):
        # Save the forward function
        self.save_func(self.forward, fname)
    
    def save_jacobian(self, fname):
        self.save_func(self.jacobian, fname)

    def save_as_savedmodel(self, fname):
        # Might fail unless WRAPT_DISABLE_EXTENSIONS=1 environment variable is
        # set. See the following for more info:
        # https://github.com/tensorflow/tensorflow/issues/63548#issuecomment-2008941537
        # https://github.com/GrahamDumpleton/wrapt/issues/231#issuecomment-1455800902
        tf.saved_model.save(self.model, fname)
        
    def save_tflite(self, func, fname, sh):
        tfunc = tf.function(func, input_signature=[tf.TensorSpec(sh, tf.float32, name="x")])

        concrete_func = tfunc.get_concrete_function() #(tf.TensorSpec(shape=[1, 36], dtype=tf.float32, name="x"))
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = []#tf.lite.Optimize.]
        converter.allow_custom_ops = False
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS 
        ]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()
        
        with open(fname, 'wb') as f:
            f.write(tflite_model)
# The below, if I were to try using it again, may require wrapt version <1.15.
# See: https://github.com/tensorflow/tensorflow/issues/59869#issuecomment-1452785730
# tf.saved_model.save(wrapper, "forward_savedmodel", signatures={"serving_default": forward_func})

# # Save the Jacobian function as TFLite
# jacobian_func = wrapper.jacobian.get_concrete_function()
# tf.saved_model.save(wrapper, "jacobian_savedmodel", signatures={"serving_default": jacobian_func})

#%%
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# https://answers.opencv.org/question/204121/keras-densenet121-breaks-on-opencv-dnn/
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

class ModelExportWrapper(tf.Module):
    def __init__(self, model, input_shape):
        super().__init__()
        self.model = model
        self.input_shape = input_shape

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
        tfunc = tf.function(func, input_signature=[tf.TensorSpec(self.input_shape, tf.float32, name="x")])
        concrete_func = tfunc.get_concrete_function()
        frozen_func = convert_variables_to_constants_v2(concrete_func)
        graph_def = frozen_func.graph.as_graph_def()

        # https://github.com/opencv/opencv/wiki/TensorFlow-text-graphs
        for i in reversed(range(len(graph_def.node))):
            if graph_def.node[i].op == 'Const':
                del graph_def.node[i]
            # https://medium.com/@sebastingarcaacosta/how-to-export-a-tensorflow-2-x-keras-model-to-a-frozen-and-optimized-graph-39740846d9eb
            for attr in ['T', 'data_format', 'Tshape', 'N', 'Tidx', 'Tdim',
                'use_cudnn_on_gpu', 'Index', 'Tperm', 'is_training',
                'Tpaddings']:
                if attr in graph_def.node[i].attr:
                    del graph_def.node[i].attr[attr]

        graph_def.library.Clear()




        # Export frozen graph
        # See:
        # - https://github.com/opencv/opencv/issues/16879#issuecomment-603815872
        # - https://github.com/opencv/opencv/issues/16582#issuecomment-603819498
        # with tf.io.gfile.GFile(fname, 'wb') as f:
        #     f.write(graph_def.SerializeToString())


        # tf.compat.v1.train.write_graph(graph_def, "", fname, as_text=False)
        # tf.compat.v1.train.write_graph(graph_def, "", fname + "txt", as_text=True)

        tf.io.write_graph(frozen_func.graph, "", fname, as_text=False)
        tf.io.write_graph(frozen_func.graph, "", fname + "txt", as_text=True)
        tf.compat.v1.train.Saver().save(frozen_func, fname + ".chkp")

        freeze_graph.freeze_graph(
            fname + "txt",
            None, False,
            fname + ".chkp",
            "Identity",    # output node name
            "save/restore_all",  # restore_op_name
            "save/Const:0",      # filename_tensor_name
            fname,
            True, ""
        )


        # Export frozen graph definition (.pbtxt)
        # with open(fname + '.pbtxt', 'w') as f:
        #     f.write(str(graph_def))

        # Slight variations to still try:
        # - https://github.com/TanFluent/facenet_opencv_dnn/
        # - https://medium.com/@sebastingarcaacosta/how-to-export-a-tensorflow-2-x-keras-model-to-a-frozen-and-optimized-graph-39740846d9eb
        # - https://answers.opencv.org/question/204121/keras-densenet121-breaks-on-opencv-dnn/
        #   - https://www.reddit.com/r/Lobe/comments/jl9q8m/tensorflow_and_unity/
        #   - https://stackoverflow.com/questions/54757293/tensorflow-freeze-graph-unable-to-initialize-local-variables
    
    def save_forward(self, fname):
        # Save the forward function
        self.save_func(self.forward, fname)
    
    def save_jacobian(self, fname):
        # Save the jacobian function
        self.save_func(self.jacobian, fname)

    def save_as_savedmodel(self, fname):
        # Might fail unless WRAPT_DISABLE_EXTENSIONS=1 environment variable is
        # set. See the following for more info:
        # https://github.com/tensorflow/tensorflow/issues/63548#issuecomment-2008941537
        # https://github.com/GrahamDumpleton/wrapt/issues/231#issuecomment-1455800902
        tf.saved_model.save(self.model, fname)
        
    def save_tflite(self, func, fname):
        tfunc = tf.function(func, input_signature=[tf.TensorSpec(self.input_shape, tf.float32, name="x")])

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

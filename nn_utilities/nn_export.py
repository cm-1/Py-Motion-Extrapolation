#%%
import typing
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# https://answers.opencv.org/question/204121/keras-densenet121-breaks-on-opencv-dnn/
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

class ModelExportWrapper(tf.Module):
    def __init__(self, model, input_shape):
        super().__init__()
        self.model = model
        self.input_shape: typing.Tuple[int, ...] = input_shape

    def forward(self, x):
        """Forward pass subgraph"""
        return self.model(x, training=False)
    
    def jacobian_orig(self, x):
        """Jacobian subgraph"""
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            y = self.model(x, training=False)
        unflat_jacobian = tape.jacobian(y, x)
        # tf.print("unflat jacobian shape:", tf.shape(unflat_jacobian))
        return tf.reshape(unflat_jacobian, [-1, tf.shape(unflat_jacobian)[-1]])

    def jacobian_split(self, x):
        """Jacobian subgraph"""
        x.set_shape([1, 36])
        with tf.GradientTape(persistent=True) as tape: #, watch_accessed_variables=False) as tape:
            tape.watch(x)
            y = self.model(x, training=False)
            y_unstacked = tf.unstack(y, axis=1)
        
        def temp_grad(y_ind):
            y_slice = y_unstacked[y_ind]
            return tape.gradient(y_slice, x)

        jacobian = tf.stack([
            temp_grad(0), temp_grad(1), temp_grad(2), temp_grad(3),
            temp_grad(4), temp_grad(5)
        ], axis=1)
        return jacobian
    
    # ff2 = wrapper.get_frozen_func(wrapper.jacobian_split)
    # ops_all2 = [o.name for o in ff2.graph.operations]
    # len([o for o in ops_all2 if "zeros_like" in o.lower()])
    '''
    ERROR: C:/users/username/_bazel_username/sueumax6/external/snappy/BUILD.bazel:89:8: Executing genrule @snappy//:snappy_stubs_public_h failed: (Exit 1): bash.exe failed: error executing command (from target @snappy//:snappy_stubs_public_h)
    cd /d C:/users/username/_bazel_username/sueumax6/execroot/org_tensorflow
    SET CLANG_COMPILER_PATH=C:Program FilesLLVMbinclang.exe
        SET PATH=<bunch of stuff>
        SET PYTHON_BIN_PATH=C:/Users/username/Documents/python_venvs/building_tf/Scripts/python.exe
        SET PYTHON_LIB_PATH=C:/Users/username/Documents/python_venvs/building_tf/lib/site-packages
        SET TF2_BEHAVIOR=1
    bash.exe -c source external/bazel_tools/tools/genrule/genrule-setup.sh; sed -e 's/${\(.*\)_01}/\1/g' -e 's/${SNAPPY_MAJOR}/1/g' -e 's/${SNAPPY_MINOR}/1/g' -e 's/${SNAPPY_PATCHLEVEL}/4/g' external/snappy/snappy-stubs-public.h.in >bazel-out/x64_windows-opt/bin/external/snappy/snappy-stubs-public.h
    # Configuration: 89aeafe5f56cecccb8c7d42150516b77f11f26ebfd703d19f64b0d3a70eb37b9
    # Execution platform: @local_execution_config_platform//:platform
    /bin/bash: source external/bazel_tools/tools/genrule/genrule-setup.sh; sed -e 's/${\(.*\)_01}/\1/g' -e 's/${SNAPPY_MAJOR}/1/g' -e 's/${SNAPPY_MINOR}/1/g' -e 's/${SNAPPY_PATCHLEVEL}/4/g' external/snappy/snappy-stubs-public.h.in >bazel-out/x64_windows-opt/bin/external/snappy/snappy-stubs-public.h: bad substitution
    Target //tensorflow/lite/delegates/flex:tensorflowlite_flex failed to build
    INFO: Elapsed time: 270.593s, Critical Path: 5.38s
    INFO: 115 processes: 101 internal, 14 local.
    FAILED: Build did NOT complete successfully
    '''

    def jacobian_forward(self, x):
        x.set_shape([1, 36])
        jacobian_cols = []
        for i in range(36):
            tangent = tf.one_hot(indices=i, depth=36, on_value=1, off_value=0, dtype=tf.float32)
            tangent = tf.reshape(tangent, (1, 36))
            
            with tf.autodiff.ForwardAccumulator(x, tangent) as acc:
                y = self.model(x, training=False)
                
            col = acc.jvp(y)
            jacobian_cols.append(col)
            
        jacobian = tf.stack(jacobian_cols, axis=2)
        
        return jacobian

    def jacobian(self, x):
        x.set_shape([1, 36])
        
        projected_scalars = []
        grads = []
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            # Run inference
            y = self.model(x, training=False)
            
            # y needs to be static for the loop
            # Assuming y is (1, OutputDim)
            output_dim = 6 # or y.shape[1] if known

            for i in range(output_dim):
                # --- THE FIX ---
                # Create the "Selector" as a constant vector, not a slice operation.
                # Shape: (36, 1)
                # We create this outside the tape so it's just a constant in the graph.
                projection_vec = tf.reshape(
                    tf.one_hot(i, depth=output_dim, dtype=tf.float32), [output_dim, 1]
                )
                
                # Project y to a scalar. 
                # (1, 36) @ (36, 1) -> (1, 1)
                # The backward gradient of this is just 'projection_vec'. 
                # No 'ZerosLike' needed!
                projected_scalars.append(tf.matmul(y, projection_vec))
        for projected_scalar in projected_scalars:
            # Calculate gradient of this scalar
            grad = tape.gradient(projected_scalar, x)
            grads.append(grad)

        # 3. Stack results
        # result shape: (1, OutputDim, 36)
        jacobian = tf.stack(grads, axis=1)
        
        return jacobian


    def get_frozen_func(self, func):
        tfunc = tf.function(func, input_signature=[tf.TensorSpec(self.input_shape, tf.float32, name="x")])
        concrete_func = tfunc.get_concrete_function()
        frozen_func = convert_variables_to_constants_v2(concrete_func)
        return frozen_func

    def save_func(self, func, fname):
        graph_def = self.get_frozen_func(func).graph.as_graph_def()
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

        tf.io.write_graph(graph_def, "", fname, as_text=False)
        tf.io.write_graph(graph_def, "", fname + "txt", as_text=True)
        # tf.compat.v1.train.Saver().save(frozen_func, fname + ".chkp")

        # freeze_graph.freeze_graph(
        #     fname + "txt",
        #     None, False,
        #     fname + ".chkp",
        #     "Identity",    # output node name
        #     "save/restore_all",  # restore_op_name
        #     "save/Const:0",      # filename_tensor_name
        #     fname,
        #     True, ""
        # )


        # Export frozen graph definition (.pbtxt)
        # with open(fname + '.pbtxt', 'w') as f:
        #     f.write(str(graph_def))

        # Slight variations to still try:
        # - https://github.com/TanFluent/facenet_opencv_dnn/
        # - https://medium.com/@sebastingarcaacosta/how-to-export-a-tensorflow-2-x-keras-model-to-a-frozen-and-optimized-graph-39740846d9eb
        # - https://answers.opencv.org/question/204121/keras-densenet121-breaks-on-opencv-dnn/
        #   - https://www.reddit.com/r/Lobe/comments/jl9q8m/tensorflow_and_unity/
        #   - https://stackoverflow.com/questions/54757293/tensorflow-freeze-graph-unable-to-initialize-local-variables
        # - https://jeanvitor.com/tensorflow-object-detecion-opencv/
        #  - https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API

        # transform_graph:
        # - https://github.com/tensorflow/tensorflow/tree/r2.0/tensorflow/tools/graph_transforms
        # - https://pypi.org/project/tensorflow/1.15.5/#files
        # - https://github.com/opencv/opencv/issues/11577

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
        
    def save_tflite(self, func, fname, allow_select_tf_ops: bool):
        tfunc = tf.function(func, input_signature=[tf.TensorSpec(self.input_shape, tf.float32, name="x")])

        concrete_func = tfunc.get_concrete_function() #(tf.TensorSpec(shape=[1, 36], dtype=tf.float32, name="x"))
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.experimental_new_converter = True
        converter.optimizations = []#tf.lite.Optimize.]
        converter.allow_custom_ops = False
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        if allow_select_tf_ops:
            converter.target_spec.supported_ops.append(
                tf.lite.OpsSet.SELECT_TF_OPS
            )
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

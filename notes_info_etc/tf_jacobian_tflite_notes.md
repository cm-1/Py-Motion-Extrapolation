Some notes/references for myself:

https://www.tensorflow.org/guide/advanced_autodiff
https://www.tensorflow.org/guide/autodiff
https://www.tensorflow.org/guide/intro_to_graphs
https://stackoverflow.com/questions/42848322/what-does-my-choice-of-glfw-samples-actually-do

`Don't rely on Python side effects like object mutation or LIST APPENDS.`
tf.config.run_functions_eagerly(bool)

When tf.function does decide to trace, the tracing stage is immediately followed by the second stage, so calling the tf.function both creates and runs the tf.Graph.
    Later you will see how you can run only the tracing stage with get_concrete_function.

You can use pretty_printed_concrete_signatures() to see all of the available traces:
> print(double.pretty_printed_concrete_signatures())

https://www.tensorflow.org/guide/function#obtaining_concrete_functions
https://www.tensorflow.org/guide/saved_model
https://www.tensorflow.org/api_docs/python/tf/print

https://stackoverflow.com/questions/73267868/how-to-do-slice-and-update-operation-on-tensors-in-tensorflow-2-0
https://stackoverflow.com/questions/62092147/how-to-efficiently-assign-to-a-slice-of-a-tensor-in-tensorflow
https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update


https://stackoverflow.com/questions/56616485/equality-comparison-does-not-work-inside-tensorflow-2-0-tf-function

Some functions I had for slicing in a way where the gradient was exportable:
```py

# @tf.custom_gradient
def safe_slice_ax0(x: tf.Tensor, start_index: int, end_index: int):
    """
    Slices x[:, start:end] in a way that is TFLite-Jacobian friendly.
    
    Forward Pass: Native Slicing (Zero overhead).
    Backward Pass: Uses tf.pad instead of StridedSliceGrad/ZerosLike.
    """
    # 1. Forward Pass: Standard efficient slicing
    # We cast to int just to be safe if passed as Tensors, though python ints are best.
    y = x[start_index:end_index]
        
    # 2. Define the Gradient
    def grad(dy):
        # dy is the gradient coming back from the next layer.
        # It has shape (Batch, end - start).
        # We need to turn it into shape (Batch, Full_Width) (dx).
        
        # Get input dimensions (Static is best for TFLite)
        # Using .shape handles static shapes; tf.shape handles dynamic.
        input_shape = x.shape
        full_width = input_shape[0] 

        # Calculate padding amounts
        # We want: [0, 0] padding on axis 0 (Batch)
        # We want: [start, full_width - end] padding on axis 1 (Features)

        pad_right = full_width - end_index

        paddings = tf.convert_to_tensor([[start_index, pad_right]])
        
        # Use tf.pad to reconstruct the gradient
        # TFLite supports Pad/PadV2 natively.
        dx = tf.pad(dy, paddings)
        
        # Return gradient for 'x', and None for 'start'/'end' (they are integers)
        return dx, None, None

    return y#, grad

# @tf.custom_gradient
def safe_slice_ax1(x: tf.Tensor, start_index: int, end_index: int):
    """
    Slices x[:, start:end] in a way that is TFLite-Jacobian friendly.
    
    Forward Pass: Native Slicing (Zero overhead).
    Backward Pass: Uses tf.pad instead of StridedSliceGrad/ZerosLike.
    """
    # 1. Forward Pass: Standard efficient slicing
    # We cast to int just to be safe if passed as Tensors, though python ints are best.
    y = x[:, start_index:end_index]
        
    # 2. Define the Gradient
    def grad(dy):
        # dy is the gradient coming back from the next layer.
        # It has shape (Batch, end - start).
        # We need to turn it into shape (Batch, Full_Width) (dx).
        
        # Get input dimensions (Static is best for TFLite)
        # Using .shape handles static shapes; tf.shape handles dynamic.
        input_shape = x.shape
        full_width = input_shape[1] 

        # Calculate padding amounts
        # We want: [0, 0] padding on axis 0 (Batch)
        # We want: [start, full_width - end] padding on axis 1 (Features)

        pad_right = full_width - end_index

        paddings = tf.convert_to_tensor([[0, 0], [start_index, pad_right]])
        
        # Use tf.pad to reconstruct the gradient
        # TFLite supports Pad/PadV2 natively.
        dx = tf.pad(dy, paddings)
        
        # Return gradient for 'x', and None for 'start'/'end' (they are integers)
        return dx, None, None

    return y#, grad
```
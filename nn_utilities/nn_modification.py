import tensorflow as tf


def create_hardcoded_model(refModel, const_out):
    shp = refModel.get_config()['layers'][0]['config']['batch_shape'][1:]
    oshp = refModel.compute_output_shape(shp)
    inputs = tf.keras.Input(shape=shp)
    const_out_tf = tf.constant(const_out)

    const_shp = const_out_tf.shape
    # Ensure we can take a slice of input that matches const shape.
    # That is, that its last dimensions are at least as big as the const's.
    for i, d in enumerate(const_shp):
        if shp[-(i+1)] < d:
            raise Exception("Const/In shape mismatch!")

    # Slice trailing axes to the constant's shape; leave batch/leading axes alone
    slices = (Ellipsis,) + tuple(slice(0, d) for d in const_out_tf.shape)
    # I multiply by 0 because I'm not sure if disregarding x entirely would
    # affect a fair tf export comparison compared to the non-hardcoded models.
    const_func = lambda x: (x * 0.0)[slices] + tf.constant(const_out_tf)

    # Build rest of model.
    outputs = tf.keras.layers.Lambda(const_func, output_shape=oshp)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    assert model.output_shape[1:] == refModel.output_shape[1:], "Const/Out shape mismatch!"
    
    return model
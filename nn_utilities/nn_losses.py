import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
def poseLossVec3(y_true, y_pred):
    return tf.norm(y_true - y_pred, axis=-1)

@keras.saving.register_keras_serializable()
def poseLossLagrange(y_true, y_pred):
    # Divide coeffs by sum to ensure we get an affine combination of the points.
    y_pred_n = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)

    # Positions passed in as T6, T5, ..., T0 semi-flattened into vec21s.
    prev_positions = tf.reshape(y_true[:, 3:], (-1, 6, 3))
    # Apply the coefficients:
    prod = tf.reshape(y_pred_n, (-1, 6, 1)) * prev_positions
    predictions_vec3 = tf.reduce_sum(prod, axis = -2)

    true_vec3 = y_true[:, :3]
    err_vec3 = true_vec3 - predictions_vec3
    
    return tf.norm(err_vec3, axis=-1)

# Get the pose loss for a set of Jerk, Acceleration, & Velocity multipliers.
@keras.saving.register_keras_serializable()
def poseLossJAV(y_true, y_pred):
    '''
    When we create the "JAV" data, we specify the permutation of
    (velocity, acceleration, jerk) to orthonormalize into frames. For simplicity
    below, assume the order is in fact velocity, then acceleration, then jerk.

    In that case, y_true contains the following columns, in order:
     - speed
     - accel parallel to velocity
     - accel ortho to velocity
     - jerk parallel to speed, ortho to speed but in acc plane, ortho to plane
     - correct pose displacement in the same "coordinate frame" as the jerk.

    In other words, we are working in an orthonormal coordinate frame where the 
    x-axis is aligned with velocity, the y with acceleration, and then z is
    orthogonal to both.
    
    Then, y_pred contains the multipliers for velocity, acceleration, and jerk, 
    respectively. The predicted "local" displacement is thus:
    [[speed, acc_x, jerk_x],       [vel_multiplier,
     [0,     acc_y, jerk_y],     x  acc_multiplier,
     [0,     0,     jerk_z]]        jerk_multiplier]

    Then after this matrix multiplication, we find the distance between it and
    the correct pose displacement, both vec3s.
    '''

    # y_pred2 = y_pred + y_true[:, 9:15]

    pred_disp_0 = y_true[:, 0] * y_pred[:, 0] + y_true[:, 1] * y_pred[:, 1] \
        + y_true[:, 3] * y_pred[:, 3] + y_true[:, 6] * y_pred[:, 6] \
        + y_true[:, 9] * y_pred[:, 9]
    pred_disp_1 = y_true[:, 2] * y_pred[:, 2] + y_true[:, 4] * y_pred[:, 4] \
        + y_true[:, 7] * y_pred[:, 7] + y_true[:, 10] * y_pred[:, 10]
    pred_disp_2 = y_true[:, 5] * y_pred[:, 5] + y_true[:, 8] * y_pred[:, 8] \
        + y_true[:, 11] * y_pred[:, 11]

    pred_disp = tf.stack([pred_disp_0, pred_disp_1, pred_disp_2], axis=-1)
    # pred_disp = tf.gather(y_true, (0,2,5), axis=-1) * y_pred

    true_disp = y_true[:, 12:15] #6:9]

    err_vec3 = true_disp - pred_disp
    return tf.norm(err_vec3, axis=-1)

@keras.saving.register_keras_serializable()
def poseLossResidualJAV(y_true, y_pred):

    y_pred2 = y_pred + y_true[:, 15:27] #9:15]

    pred_disp_0 = y_true[:, 0] * y_pred2[:, 0] + y_true[:, 1] * y_pred2[:, 1] \
        + y_true[:, 3] * y_pred2[:, 3] + y_true[:, 6] * y_pred2[:, 6] \
        + y_true[:, 9] * y_pred2[:, 9]
    pred_disp_1 = y_true[:, 2] * y_pred2[:, 2] + y_true[:, 4] * y_pred2[:, 4] \
        + y_true[:, 7] * y_pred2[:, 7] + y_true[:, 10] * y_pred2[:, 10]
    pred_disp_2 = y_true[:, 5] * y_pred2[:, 5] + y_true[:, 8] * y_pred2[:, 8] \
        + y_true[:, 11] * y_pred2[:, 11]

    pred_disp = tf.stack([pred_disp_0, pred_disp_1, pred_disp_2], axis=-1)
    # pred_disp = tf.gather(y_true, (0,2,5), axis=-1) * y_pred

    true_disp = y_true[:, 12:15] #6:9]

    err_vec3 = true_disp - pred_disp
    return tf.norm(err_vec3, axis=-1)

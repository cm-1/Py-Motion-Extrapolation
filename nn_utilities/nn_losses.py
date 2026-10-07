import tensorflow as tf
import keras
import math

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

# Below are some temporary Claude functions.
# Might still experiment with other ways of handling cases where the gradient's
# undefined
def _axis_angle_to_quat(v, eps=1e-8):
    """Convert axis-angle (rotation vector) to unit quaternion (w, xyz)."""
    theta_sq = tf.reduce_sum(tf.square(v), axis=-1, keepdims=True)
    theta = tf.sqrt(theta_sq + eps ** 2)  # safe sqrt (avoids NaN gradient at 0)
    half = 0.5 * theta

    # sin(theta/2) / theta, with a Taylor expansion for small angles
    small = theta < 1e-3
    scale = tf.where(small, 0.5 - theta_sq / 48.0, tf.sin(half) / theta)

    w = tf.cos(half)
    xyz = v * scale
    return w, xyz


@keras.saving.register_keras_serializable()
def poseLossRotVec3CL(y_true, y_pred):
    """Geodesic angle (radians) between two axis-angle rotations."""
    y_true = tf.cast(y_true, y_pred.dtype)

    w1, v1 = _axis_angle_to_quat(y_true)
    w2, v2 = _axis_angle_to_quat(y_pred)

    # Relative rotation q_rel = conj(q1) * q2
    w_rel = tf.squeeze(w1 * w2, -1) + tf.reduce_sum(v1 * v2, axis=-1)
    v_rel = w1 * v2 - w2 * v1 - tf.linalg.cross(v1, v2)
    v_rel_norm = tf.sqrt(tf.reduce_sum(tf.square(v_rel), axis=-1) + 1e-12)

    # abs() handles the q / -q double cover; atan2 is more stable than acos
    return 2.0 * tf.atan2(v_rel_norm, tf.abs(w_rel))


@keras.saving.register_keras_serializable()
def poseLossRotVec3CL2(y_true, y_pred):
    a = tf.cast(y_true, y_pred.dtype)
    b = y_pred
    two_pi = tf.constant(2.0 * math.pi, dtype=b.dtype)

    na_sq = tf.reduce_sum(tf.square(a), axis=-1, keepdims=True)
    nb_sq = tf.reduce_sum(tf.square(b), axis=-1, keepdims=True)
    small = (na_sq < 1e-8) | (nb_sq < 1e-8)          # shape [..., 1]

    # Inner where: the main branch never sees a zero-norm vector
    ones = tf.ones_like(a)
    safe_a = tf.where(small, ones, a)
    safe_b = tf.where(small, ones, b)

    # Main branch: your simple_angle_acos formula
    na = tf.norm(safe_a, axis=-1)
    nb = tf.norm(safe_b, axis=-1)
    cos_phi = tf.reduce_sum(safe_a * safe_b, axis=-1) / (na * nb)
    w_rel = (tf.cos(na / 2) * tf.cos(nb / 2)
             + cos_phi * tf.sin(na / 2) * tf.sin(nb / 2))
    c = tf.clip_by_value(tf.abs(w_rel), 0.0, 1.0 - 1e-7)
    main = 2.0 * tf.acos(c)

    # Fallback: ||a - b||, wrapped to [0, pi]; smoothed norm keeps the gradient finite
    n = tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=-1) + 1e-12)
    fallback = tf.abs(n - two_pi * tf.stop_gradient(tf.round(n / two_pi)))

    # Outer where
    return tf.where(tf.squeeze(small, -1), fallback, main)
"""
The following script was written by Claude code as part of a debugging session.
This exact script won't get used again, but there are ideas in here that I might
want to revisit, so I'm storing it away.

Instrumented comparison script that captures intermediate values from both
implementations and automatically identifies where they first diverge.

This script modifies the implementations to capture intermediate values,
then compares them systematically.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple
import traceback
import importlib
import sys


class ValueRecorder:
    """Records intermediate computation values for comparison."""
    
    def __init__(self, name: str):
        self.name = name
        self.values: Dict[str, Any] = {}
        self.order = []  # Track order of recordings
        
    def record(self, key: str, value: Any, convert_tf: bool = True):
        """Record a value. Converts TF tensors to numpy for comparison."""
        if convert_tf and isinstance(value, tf.Tensor):
            value = value.numpy()
        self.values[key] = value
        if key not in self.order:
            self.order.append(key)
        return value
    
    def get(self, key: str):
        """Retrieve a recorded value."""
        return self.values.get(key)


def compare_values(numpy_val: Any, tf_val: Any, tolerance: float = 1e-5) -> Tuple[bool, float, float]:
    """
    Compare two values and return (matches, max_diff, mean_diff).
    Handles various types including arrays, tuples, etc.
    """
    try:
        # Handle None
        if numpy_val is None and tf_val is None:
            return True, 0.0, 0.0
        if numpy_val is None or tf_val is None:
            return False, float('inf'), float('inf')
        
        # Handle scalars
        if np.isscalar(numpy_val) and np.isscalar(tf_val):
            diff = abs(float(numpy_val) - float(tf_val))
            return diff < tolerance, diff, diff
        
        # Convert to numpy arrays
        np_arr = np.asarray(numpy_val)
        tf_arr = np.asarray(tf_val)
        
        # Check shapes match
        if np_arr.shape != tf_arr.shape:
            return False, float('inf'), float('inf')
        
        # Calculate differences
        abs_diff = np.abs(np_arr - tf_arr)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        
        matches = max_diff < tolerance
        return matches, max_diff, mean_diff
        
    except Exception as e:
        print(f"Error comparing values: {e}")
        return False, float('inf'), float('inf')


def instrumented_numpy_calculation(x0_through_4, rmats0_through_5, x5, step, ref_keys, recorder: ValueRecorder):
    """
    Instrumented version of HypotheticalInputsForNN.getHypotheticalCalcs
    that records intermediate values.
    """
    from motiontools.hypothetical_inputs_calc import HypotheticalInputsForNN
    import posemath as pm
    
    # Create the calculator
    calc = HypotheticalInputsForNN(x0_through_4, rmats0_through_5, step, ref_keys)
    
    # Now manually execute getHypotheticalCalcs with recording
    vels = calc._stepDiv(x5 - calc.x0_through_4[-1])
    recorder.record("vels", vels)
    
    vel_mags = np.linalg.norm(vels, axis=-1, keepdims=True)
    recorder.record("vel_mags", vel_mags)
    
    vel_mag_is_0 = np.where((vel_mags == 0.0).flatten())[0]
    recorder.record("vel_mag_is_0", vel_mag_is_0)
    
    safediv_vel_mags = np.copy(vel_mags)
    safediv_vel_mags[vel_mag_is_0] = 1.0
    recorder.record("safediv_vel_mags", safediv_vel_mags)
    
    unit_vels = vels / safediv_vel_mags
    last_nz_uvels = calc.last_nonzero_unit_vels
    if last_nz_uvels.ndim > 1:
        last_nz_uvels = last_nz_uvels[vel_mag_is_0]
    unit_vels[vel_mag_is_0] = last_nz_uvels
    recorder.record("unit_vels", unit_vels)
    
    accs = calc._stepDiv(vels - calc.prev_vel)
    recorder.record("accs", accs)
    
    jerks = calc._stepDiv(accs - calc.prev_acc)
    recorder.record("jerks", jerks)
    
    snaps = calc._stepDiv(jerks - calc.prev_jerk)
    recorder.record("snaps", snaps)
    
    crackles = calc._stepDiv(snaps - calc.prev_snap)
    recorder.record("crackles", crackles)
    
    full_shape = vels.shape
    all_curr_vecs = np.stack(
        (
            vels, accs, jerks, snaps, crackles,
            np.broadcast_to(calc.new_ang_vel, full_shape),
            np.broadcast_to(calc.new_ang_acc, full_shape),
            np.broadcast_to(calc.new_ang_jerk, full_shape),
        ), axis=0
    )
    recorder.record("all_curr_vecs", all_curr_vecs)
    recorder.record("all_curr_vecs_shape", all_curr_vecs.shape)
    
    # Record prev_relative_vecs for comparison
    recorder.record("prev_relative_vecs", calc.prev_relative_vecs)
    recorder.record("prev_relative_scales", calc.prev_relative_scales)
    
    pr_str = 'bk' if calc.prev_relative_vecs.ndim == 2 else 'bik'
    all_dots_with_prev = np.einsum(
        'aik,' + pr_str + '->abi', all_curr_vecs, calc.prev_relative_vecs
    )
    recorder.record("all_dots_with_prev", all_dots_with_prev)
    
    all_proj_with_prev = pm.safeDivideElseZero(
        all_dots_with_prev[:, :-2], calc.prev_relative_scales,
        calc.prev_relative_scales_nonzero
    )
    recorder.record("all_proj_with_prev", all_proj_with_prev)
    
    (_, a_proj_v, a_ortho_v), curr_ortho_mats = pm.getOrthonormalFrames(
        True, unit_vels, accs, vecs0_are_unit_len=True,
        set_zeros_to_zero=True
    )
    recorder.record("a_proj_v", a_proj_v)
    recorder.record("a_ortho_v", a_ortho_v)
    recorder.record("curr_ortho_mats", curr_ortho_mats)
    recorder.record("curr_ortho_mats_shape", curr_ortho_mats.shape)
    
    # Early return for initial comparison
    return {
        'vels': vels,
        'vel_mags': vel_mags,
        'accs': accs,
        'jerks': jerks,
        'a_proj_v': a_proj_v,
        'a_ortho_v': a_ortho_v,
        'curr_ortho_mats': curr_ortho_mats,
        'all_curr_vecs': all_curr_vecs,
    }


def instrumented_tf_calculation(x0_through_5, aa0_through_5, step, ref_keys, recorder: ValueRecorder):
    """
    Instrumented version of PointsToInputsConstStep.calculateOutputs
    that records intermediate values.
    """
    from nn_utilities.nn_inference import (
        PointsToInputsConstStep, 
        DerivativeCollectionConstTimeTF,
        compute_relative_rotations,
        tfOrthonormalFramesFromUnitVec0s
    )
    import keras
    
    # Create the calculator
    calc = PointsToInputsConstStep(step, ref_keys)
    
    # Manually execute calculateOutputs with recording
    all_pds = DerivativeCollectionConstTimeTF(
        keras.ops.diff(x0_through_5, 1, axis=0), 5, step
    )
    
    angvel_aas = compute_relative_rotations(
        aa0_through_5[:-1], aa0_through_5[1:]
    )
    recorder.record("angvel_aas", angvel_aas)
    
    all_rds = DerivativeCollectionConstTimeTF(angvel_aas, 3, step)
    
    all_prev_vels = all_pds.velocities[:-1]
    recorder.record("all_prev_vels", all_prev_vels)
    
    prev_vel = all_prev_vels[-1]
    recorder.record("prev_vel", prev_vel)
    
    prev_acc = all_pds.accelerations[-2]
    recorder.record("prev_acc", prev_acc)
    
    prev_jerk = all_pds.jerks[-2]
    recorder.record("prev_jerk", prev_jerk)
    
    prev_snap = all_pds.snaps[-2]
    recorder.record("prev_snap", prev_snap)
    
    vels = all_pds.velocities[-1]
    recorder.record("vels", vels)
    
    vel_mags = tf.norm(vels, axis=-1, keepdims=True)
    recorder.record("vel_mags", vel_mags)
    
    accs = all_pds.accelerations[-1]
    recorder.record("accs", accs)
    
    jerks = all_pds.jerks[-1]
    recorder.record("jerks", jerks)
    
    snaps = all_pds.snaps[-1]
    recorder.record("snaps", snaps)
    
    crackles = all_pds.crackles[-1]
    recorder.record("crackles", crackles)
    
    prev_speed = tf.norm(prev_vel, axis=-1, keepdims=True)
    recorder.record("prev_speed", prev_speed)
    
    prev_unit_vel = tf.math.divide_no_nan(prev_vel, prev_speed)
    recorder.record("prev_unit_vel", prev_unit_vel)
    
    new_ang_vel = all_rds.velocities[-1]
    new_ang_acc = all_rds.accelerations[-1]
    new_ang_jerk = all_rds.jerks[-1]
    
    recorder.record("new_ang_vel", new_ang_vel)
    recorder.record("new_ang_acc", new_ang_acc)
    recorder.record("new_ang_jerk", new_ang_jerk)
    
    unit_vels = tf.math.divide_no_nan(vels, vel_mags)
    recorder.record("unit_vels", unit_vels)
    
    all_curr_vecs = tf.stack(
        (
            vels, accs, jerks, snaps, crackles,
            new_ang_vel, new_ang_acc, new_ang_jerk
        ), axis=0
    )
    recorder.record("all_curr_vecs", all_curr_vecs)
    recorder.record("all_curr_vecs_shape", all_curr_vecs.shape)
    
    (a_proj_v, a_ortho_v), curr_ortho_mats = tfOrthonormalFramesFromUnitVec0s(
        True, unit_vels, accs, calc.zero_angle_thresh 
    )
    recorder.record("a_proj_v", a_proj_v)
    recorder.record("a_ortho_v", a_ortho_v)
    recorder.record("curr_ortho_mats", curr_ortho_mats)
    recorder.record("curr_ortho_mats_shape", curr_ortho_mats.shape)
    
    # Early return for initial comparison
    return {
        'vels': vels,
        'vel_mags': vel_mags,
        'accs': accs,
        'jerks': jerks,
        'a_proj_v': a_proj_v,
        'a_ortho_v': a_ortho_v,
        'curr_ortho_mats': curr_ortho_mats,
        'all_curr_vecs': all_curr_vecs,
    }


def run_instrumented_comparison():
    """Run the instrumented comparison."""
    
    # Force reload of modules to pick up changes
    if 'nn_utilities.nn_inference' in sys.modules:
        importlib.reload(sys.modules['nn_utilities.nn_inference'])
    if 'motiontools.hypothetical_inputs_calc' in sys.modules:
        importlib.reload(sys.modules['motiontools.hypothetical_inputs_calc'])
    
    print("=" * 80)
    print("INSTRUMENTED COMPARISON")
    print("=" * 80)
    
    # Generate test data
    np.random.seed(123)
    step = 1
    n_points = 5
    
    x0_through_4 = np.random.randn(5, n_points, 3) * 10
    random_axis_angles = np.random.randn(6, n_points, 3) * 0.5
    
    rmats0_through_5 = np.zeros((6, n_points, 3, 3))
    for i in range(6):
        for j in range(n_points):
            rmats0_through_5[i, j] = np.eye(3)
            angle = np.linalg.norm(random_axis_angles[i, j])
            if angle > 1e-6:
                axis = random_axis_angles[i, j] / angle
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                rmats0_through_5[i, j] = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    x5 = np.random.randn(n_points, 3) * 10
    x0_through_5 = np.concatenate([x0_through_4, x5[np.newaxis, :, :]], axis=0)
    
    from motiontools.hypothetical_inputs_calc import HypotheticalInputsForNN
    ref_keys = HypotheticalInputsForNN._generated_key_order()
    
    # Create recorders
    numpy_recorder = ValueRecorder("NumPy")
    tf_recorder = ValueRecorder("TensorFlow")
    
    # Run instrumented versions
    print("\nRunning instrumented NumPy calculation...")
    try:
        numpy_results = instrumented_numpy_calculation(
            x0_through_4, rmats0_through_5, x5, step, ref_keys, numpy_recorder
        )
        print(f"✓ NumPy calculation completed. Recorded {len(numpy_recorder.values)} values.")
    except Exception as e:
        print(f"✗ NumPy calculation failed: {e}")
        traceback.print_exc()
        return
    
    print("\nRunning instrumented TensorFlow calculation...")
    try:
        x0_through_5_tf = tf.constant(x0_through_5, dtype=tf.float32)
        axis_angles_tf = tf.constant(random_axis_angles, dtype=tf.float32)
        
        tf_results = instrumented_tf_calculation(
            x0_through_5_tf, axis_angles_tf, step, ref_keys, tf_recorder
        )
        print(f"✓ TensorFlow calculation completed. Recorded {len(tf_recorder.values)} values.")
    except Exception as e:
        print(f"✗ TensorFlow calculation failed: {e}")
        traceback.print_exc()
        return
    
    # Compare recorded values in order
    print("\n" + "=" * 80)
    print("COMPARING INTERMEDIATE VALUES (in computation order)")
    print("=" * 80)
    
    all_keys = numpy_recorder.order
    first_divergence = None
    
    for key in all_keys:
        numpy_val = numpy_recorder.get(key)
        tf_val = tf_recorder.get(key)
        
        if tf_val is None:
            print(f"\n⚠ '{key}': Not recorded in TensorFlow version")
            continue
        
        matches, max_diff, mean_diff = compare_values(numpy_val, tf_val, tolerance=1e-3)
        
        # Determine shape info
        np_shape = np.asarray(numpy_val).shape if not np.isscalar(numpy_val) else "scalar"
        tf_shape = np.asarray(tf_val).shape if not np.isscalar(tf_val) else "scalar"
        
        status = "✓ MATCH" if matches else "✗ DIFFER"
        color = "" if matches else " <-- FIRST DIVERGENCE" if first_divergence is None else ""
        
        if not matches and first_divergence is None:
            first_divergence = key
        
        print(f"\n{status}: '{key}'{color}")
        print(f"  Shape: NumPy={np_shape}, TF={tf_shape}")
        print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        
        # Show sample values for arrays
        if not np.isscalar(numpy_val) and not matches:
            np_arr = np.asarray(numpy_val)
            tf_arr = np.asarray(tf_val)
            print(f"  Sample values (first element):")
            print(f"    NumPy: {np_arr.flat[0]:.6e}")
            print(f"    TF:    {tf_arr.flat[0]:.6e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if first_divergence:
        print(f"✗ First divergence detected at: '{first_divergence}'")
        print(f"\n  This is where you should focus your debugging efforts.")
        print(f"  Check the computation of '{first_divergence}' in both implementations.")
    else:
        print("✓ All compared values match within tolerance!")


if __name__ == "__main__":
    run_instrumented_comparison()

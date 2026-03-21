# Some refactoring in this file was performed using Olmo 3.1 32B Instruct
import tensorflow as tf
import numpy as np

class LiteTester:
    def __init__(self, model_path, test_input_shape):
        # Load the TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        print("TFLite model loaded successfully!")
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Input details: {self.input_details}")
        print(f"Output details: {self.output_details}")
        
        # Resize input tensor if needed
        self.interpreter.resize_tensor_input(
            self.input_details[0]['index'], test_input_shape, strict=False
        )
        self.interpreter.allocate_tensors()  # Sometimes needed after resize

    def use_interpreter(self, test_input):
        """
        Sets input tensor and runs inference, returning the output.
        """
        self.interpreter.set_tensor(self.input_details[0]['index'], test_input)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output

def test_tflite_export(model, tflite_path, input_shape):
    """
    Tests if the TFLite export is correct by comparing with the original model.
    """
    print("=== Testing TFLite Export ===")

    def test_tflite():
        try:
            # Initialize LiteTester with the TFLite file and the expected input shape
            lite_tester = LiteTester(tflite_path, input_shape)
            return lite_tester
        except Exception as e:
            print(f"Error loading TFLite model: {e}")
            return None

    def compare_models(lite_tester, test_input):
        try:
            # Original model prediction
            original_output = model(tf.constant(test_input))
            print(f"Original model output shape: {original_output.shape}")
            print(f"Original model output: {original_output.numpy()[:3]}")  # First 3 outputs

            # TFLite model prediction
            tflite_output = lite_tester.use_interpreter(test_input)
            print(f"TFLite model output shape: {tflite_output.shape}")
            print(f"TFLite model output: {tflite_output[:3]}")  # First 3 outputs

            # Compare outputs
            diff = np.abs(original_output.numpy() - tflite_output).max()
            print(f"Maximum difference between models: {diff}")
            if diff < 1e-4:
                print("✓ Models are consistent!")
                return True
            else:
                print("⚠ Models differ significantly")
                return False
        except Exception as e:
            print(f"Error during model comparison: {e}")
            return False

    # Generate random test input
    test_input = np.random.randn(*input_shape).astype(np.float32)

    # Test TFLite model loading
    lite_tester = test_tflite()
    if lite_tester is None:
        print("⚠ TFLite model could not be loaded. Aborting comparison.")
        return False

    # Compare models
    if compare_models(lite_tester, test_input):
        print("\n🎉 All tests passed! Your TFLite export is working correctly.")
        return True
    else:
        print("⚠ Model consistency test failed")
        return False

def debug_tflite_model(model_path):
    """
    Debug helper for TFLite model.
    """
    try:
        with open(model_path, 'rb') as f:
            model_content = f.read()
            print(f"Model size: {len(model_content)} bytes")

        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("Input details:")
        for detail in input_details:
            print(f"  {detail}")
        print("Output details:")
        for detail in output_details:
            print(f"  {detail}")
    except Exception as e:
        print(f"Debug error: {e}")

# Example usage:
# Suppose `my_keras_model` is your Keras/TensorFlow model and `my_tflite_path` is your exported TFLite file
# input_shape = (1, 36)  # adjust to your model's input shape
# test_tflite_export(my_keras_model, my_tflite_path, input_shape)
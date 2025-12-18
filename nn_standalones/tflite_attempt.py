import tensorflow as tf
import numpy as np

def load_and_test_tflite(model_path: str, test_input=None):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    print("TFLite model loaded successfully!")
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")
    
    # Test with sample input
    if test_input is None:
        test_input = np.random.randn(1, 36).astype(np.float32)
    interpreter.resize_tensor_input(
        input_details[0]['index'], test_input.shape, strict=False
    )
    interpreter.allocate_tensors()
    
    # Set input
    interpreter.set_tensor(input_details[0]['index'], test_input)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    return output
    
# Test the full workflow
def test_tflite_export(model, tflite_path, s):

    def test_loaded_model(model_path):
        try:
            output = load_and_test_tflite(model_path)
            
            print(f"Test output shape: {output.shape}")
            print(f"Sample output values: {output[0][:3]}")  # First 3 outputs
            
            return True
            
        except Exception as e:
            print(f"Error testing loaded model: {e}")
            return False
    
    # Compare results between original and TFLite models
    def compare_models():
        try:
            # Create test data
            test_input = np.random.randn(*s).astype(np.float32)
            
            # Original model prediction
            original_output = model(tf.constant(test_input))#, training=False)
            print(f"Original model output shape: {original_output.shape}")
            print(f"Original model output: {original_output}")
            
            # TFLite model prediction (if we can load it)
            try:
                tflite_output = load_and_test_tflite(tflite_path, test_input)
                
                print(f"TFLite model output shape: {tflite_output.shape}")
                print(f"TFLite model output: {tflite_output}")
                
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
                print(f"Could not test model consistency: {e}")
                return False
                
        except Exception as e:
            print(f"Error in comparison: {e}")
            return False
    
    # Run all tests
    print("=== Testing TFLite Export ===")
    
    # Test conversion
    
    if compare_models():
        print("✓ Model outputs are consistent!")
        print("\n🎉 All tests passed! Your TFLite export is working correctly.")
        return True
    else:
        print("⚠ Model consistency test failed")
    
    return False

# 6. Additional debugging function
def debug_tflite_model(model_path):
    """Debug the TFLite model in detail"""
    try:
        # Load model
        with open(model_path, 'rb') as f:
            model_content = f.read()
        
        print(f"Model size: {len(model_content)} bytes")
        
        # Try to parse with TensorFlow Lite
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

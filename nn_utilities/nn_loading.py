import glob
import keras

# Needed when loading model, even though IDE syntax highlighting marks this as
# "unused"!
from nn_utilities.nn_losses import poseLossJAV

def loadLatestModels(model_prefixes):
    model_loads = dict()
    for prefix in model_prefixes:
        matching_models = glob.glob("./results/models/" + prefix + "*.keras")
        # Get most recent model (filenames are known to be timestamped).
        chosen_model_fname = sorted(matching_models)[-1] 
        print("Loading model:", chosen_model_fname)
        curr_loaded_model = keras.models.load_model(
            chosen_model_fname, custom_objects={"poseLossJAV": poseLossJAV}
        )
        model_loads[prefix] = curr_loaded_model
    return model_loads
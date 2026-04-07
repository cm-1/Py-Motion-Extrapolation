import glob
import keras

# Needed when loading model, even though IDE syntax highlighting marks this as
# "unused"!
from nn_utilities.nn_losses import poseLossJAV
from motiontools.dataorg import SUBSET_PRESET

def loadLatestModels(model_prefixes, preset: SUBSET_PRESET):
    model_loads = dict()
    for prefix in model_prefixes:
        matching_models = glob.glob("./results/models/" + prefix + "--" + preset.name + "*.keras")
        # Get most recent model (filenames are known to be timestamped).
        if len(matching_models) > 0:
            chosen_model_fname = sorted(matching_models)[-1] 
            print("Loading model:", chosen_model_fname)
            curr_loaded_model = keras.models.load_model(
                chosen_model_fname, custom_objects={"poseLossJAV": poseLossJAV}
            )
            model_loads[prefix] = curr_loaded_model
        else:
            model_loads[prefix] = None
    return model_loads
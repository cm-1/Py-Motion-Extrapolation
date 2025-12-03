#%%
from nn_utilities.nn_loading import loadLatestModels
from nn_utilities.nn_export import ModelExportWrapper

MOD_PREFIX = "JAV_MULTIPLIERS"


# Load model
loaded_model = loadLatestModels((MOD_PREFIX, ))[MOD_PREFIX]

# Create a wrapper instance
wrapper = ModelExportWrapper(loaded_model)

wrapper.save_forward("D:\\forward_model.pb")
wrapper.save_jacobian("D:\\jacobian_model.pb")
print("Done exports!")
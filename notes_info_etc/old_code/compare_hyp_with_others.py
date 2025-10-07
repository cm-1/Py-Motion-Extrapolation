
# %%
# Essentially, you would run this code in the hypothetical_point_vis.py code.
# You'd compare the "comp" variable against ts[2:66] where ts is the rows of
# data for the selected STEP value.
from motiontools.posefeatures import CalcsForVideo
from motiontools.dataorg import DataOrganizer
cfc = CalcsForVideo()
single_loader = PoseLoaderBCOT(*(bcot_id_split[-1][-1]))
cfc.getAll([single_loader], 1)

#%%
tdog = DataOrganizer(
    cfc.all_motion_data, cfc.min_norm_labels, cfc.err_norm_lists, *bcot_id_split
)
col_inds = np.asarray([i for i, k in enumerate(tdog.motion_data_keys) if k in scaler.column_keys])
tdog.setPickAndTransform(col_inds, scaler)#, True)

print("Freeing up memory!")
cfc.freeUpMemory()

#%%
comp = ps.scaled_input_storage[DataSubsetKind.TEST]


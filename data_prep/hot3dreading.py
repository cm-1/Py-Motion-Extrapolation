

# File for getting minimal-sized translation and rotation arrays from
# HOT 3D Clips dataset's many .zip files.
# File was initiated with the zipping and file logic using ChatGPT.
# This was then manually verified/tested by me, and then I added in the
# connections to my preexisting code for converting rotation matrices into
# axis-angle form for denser storage.
# %%   
import zipfile
import json
import shutil
from pathlib import Path
import psutil
import typing
from enum import Enum

import numpy as np
from numpy.typing import NDArray

import posemath as pm

class CAM_TYPE(Enum):
    ARIA = 1
    QUEST = 2

SELECTED_CAM = CAM_TYPE.QUEST

FRAMES_PER_VID = 150
ZEROS_WIDTH = 6

ALL_CAM_KEYS = {
    CAM_TYPE.ARIA: ("1201-1", "1201-2", "214-1"),
    CAM_TYPE.QUEST: ("1201-1", "1201-2")
}

ALL_REF_CAM_IDXS = {CAM_TYPE.ARIA: 2, CAM_TYPE.QUEST: 0}

def get_zip_uncompressed_size(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        return sum([zi.file_size for zi in z.infolist()])

def get_free_space_bytes(path):
    return psutil.disk_usage(str(path)).free

def delete_oldest_extracted(dirs_to_del: typing.List[Path], space_req):
    if len(dirs_to_del) == 0:
        return
    base_path = dirs_to_del[0].parent
    deleted = []
    for i, dir in enumerate(dirs_to_del):
        if get_free_space_bytes(base_path) > space_req:
            break
        print(f"Deleting {dir.name} to free up space...")
        shutil.rmtree(dir)
        deleted.append(i)
    # Delete in reverse order so shifts don't affect later deletions.
    for del_ind in deleted[::-1]:
        dirs_to_del.pop(del_ind)

def process_zips(base_dir, zip_prefix: str, inner_path: str,
                 cam_keys: typing.Tuple[str, ...], margin_bytes=2_000_000_000):
    base_dir = Path(base_dir)
    zip_files = sorted(base_dir.glob(zip_prefix + "*.zip"))
    results = {}

    processed_dirs = []
    
    f_range = tuple(range(FRAMES_PER_VID))
    frame_strs = [str(x).zfill(ZEROS_WIDTH) for x in f_range]
    cam_fnames = [fs + ".cameras.json" for fs in frame_strs]
    obj_fnames = [fs + ".objects.json" for fs in frame_strs]
    fnums_camfs_objfs = tuple(zip(f_range, cam_fnames, obj_fnames))

    for zip_path in zip_files:            
        zip_name = zip_path.stem
        extract_dir = base_dir / zip_name

        if not extract_dir.exists():
            # Check available space
            zip_uncompressed_size = get_zip_uncompressed_size(zip_path)
            free_space = get_free_space_bytes(base_dir)

            if free_space < zip_uncompressed_size + margin_bytes:
                delete_oldest_extracted(processed_dirs, margin_bytes)
                free_space = get_free_space_bytes(base_dir)

            print(f"Extracting {zip_path.name}...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_dir)
            print("Done extracting!")            

        # Walk through extracted files
        full_inner_path = extract_dir / inner_path
        if not full_inner_path.exists():
            print(f"Warning: {full_inner_path} does not exist.")
            continue

        cdata = None
        odata = None
        for clip_dir in sorted(full_inner_path.glob("clip-*")):
            clip_frame_count = len(tuple(clip_dir.glob("*.objects.json")))
            if clip_frame_count > FRAMES_PER_VID:
                raise Exception(
                    "Frame count different than expected for {}!".format(
                        str(clip_dir)
                    )
                )
            cam_storage = np.empty((FRAMES_PER_VID, len(cam_keys), 7))
            o_poses: typing.Dict[int, NDArray] = dict()
            for frame_num, cam_fn, obj_fn in fnums_camfs_objfs:
                # fn = int(ffile.name[:ffile.name.find(json_fname_pattern)])
                with open(clip_dir / cam_fn, "r") as f:
                    try:
                        cdata = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Invalid JSON in {cam_fn}")
                        continue
                with open(clip_dir / obj_fn, "r") as f:
                    try:
                        odata = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Invalid JSON in {cam_fn}")
                        continue

                for cam_i, cam_k in enumerate(cam_keys):
                    pose = cdata[cam_k]["T_world_from_camera"]
                    cam_storage[frame_num, cam_i, :4] = pose["quaternion_wxyz"]
                    cam_storage[frame_num, cam_i, 4:] = pose["translation_xyz"]

                for obj_key, pose_data in odata.items():
                    obj_key_i = int(obj_key)
                    if frame_num == 0:
                        o_poses[obj_key_i] = np.empty((FRAMES_PER_VID, 7))
                    pose = pose_data[0]["T_world_from_object"]
                    o_poses[obj_key_i][frame_num, :4] = pose["quaternion_wxyz"]
                    o_poses[obj_key_i][frame_num, 4:] = pose["translation_xyz"]
                # Done current frame.
            # Done all frames in this clip.                

            clip_key = int(clip_dir.name[-ZEROS_WIDTH:])
            results[clip_key] = (cam_storage, o_poses)
        print("Done processing", extract_dir)
        processed_dirs.append(extract_dir)
    return results
#%%
hot_dir_name = "train_aria" if SELECTED_CAM == CAM_TYPE.ARIA else "train_quest3"

# Example usage:
# base_dir should be the path containing your .zip files
results_cam_obj_q = process_zips(
    r"D:\Datasets\HOT3D", hot_dir_name + "_pt", hot_dir_name,
    ALL_CAM_KEYS[SELECTED_CAM]
)

#%%
import posemath as pm

# : typing.Dict[int, typing.Tuple[NDArray, typing.Dict[int, NDArray]]]
def results_to_np(res, ref_cam_idx: int, static_trans_thresh, static_rot_thresh):
    num_clips: int
    n_frames = 150
    is_np_load: bool
    if isinstance(res, dict):
        num_clips = len(res.keys())
        is_np_load = False
    elif isinstance(res, np.lib.npyio.NpzFile):
        is_np_load = True
        num_clips = res['obj_ids'].shape[0]
        for f in res.files:
            if res[f].shape[0] != num_clips:
                raise Exception("Unexpected npz structure!")
        assert res['cam_poses'].shape[1:] == (n_frames, 3, 7), "Bad arr shape!"
        assert res['obj_poses'].shape[1:] == (6, n_frames, 7), "Bad arr shape!"
        # Make a local copy that will be MUCH faster to read/process.
        res = {
            'obj_ids': res['obj_ids'].copy(),
            'cam_poses': res['cam_poses'].copy(),
            'obj_poses': res['obj_poses'].copy()
        }
    else:
        raise ValueError(
            "Unexpected type for the data! Expected a dict or npz load."
        )

    print("Number of clips:", num_clips)
    # Storing which object IDs are present (and static/moving) in each clip.
    obj_ids = np.zeros((num_clips, 6), dtype=np.int8)
    moving_obj_counts = np.zeros(num_clips, dtype=np.int8)

    # Storing world-to-camera poses.
    w2c_t_array = np.empty((num_clips, n_frames, 3))
    w2c_rmat_array = np.empty((num_clips, n_frames, 3, 3))
    w2c_aa_array = np.empty((num_clips, n_frames, 3))

    # Storing object-to-world and object-to-camera poses.
    moving_o2w_t_list: typing.List[NDArray] = []
    moving_o2w_rmat_list: typing.List[NDArray] = []
    moving_o2w_aa_list: typing.List[NDArray] = []

    moving_o2c_t_list: typing.List[NDArray] = []
    moving_o2c_rmat_list: typing.List[NDArray] = []
    moving_o2c_aa_list: typing.List[NDArray] = []

    all_max_t0_diffs = np.full((num_clips, 6), -1.0, dtype=np.float64)
    all_max_q0_diffs = np.full((num_clips, 6), -1.0, dtype=np.float64)
    # We need a 0-starting clip number for storage in the numpy array.
    for clip_np_ind in range(num_clips):
        clip_cam_arr: NDArray
        obj_iter: typing.List[typing.Tuple[int, NDArray]]
        if is_np_load:
            clip_cam_arr = res['cam_poses'][clip_np_ind]
            obj_iter = list(zip(
                res['obj_ids'][clip_np_ind], res['obj_poses'][clip_np_ind]
            ))
        else:
            start_clip = np.min(res.keys())
            clip_key = start_clip + clip_np_ind
            curr_entry = res[clip_key][0]
            clip_cam_arr = curr_entry[0]
            obj_iter = list(curr_entry[1].items())

        cam_quats = pm.normalizeAll(clip_cam_arr[:, ref_cam_idx, :4])
        cam_translations = clip_cam_arr[:, ref_cam_idx, 4:]

        # Above are cam-to-world. We instead want:
        cRw = pm.conjugateQuats(cam_quats)
        cTw = pm.rotateVecsByQuats(cRw, -cam_translations)

        w2c_rmat_array[clip_np_ind] = pm.matsFromQuaternions(cRw)
        w2c_t_array[clip_np_ind] = cTw
        w2c_aa_array[clip_np_ind] = pm.axisAngleVec3sFromQuats(cRw, True)

        static_obj_found = False
        static_objs: typing.List[int] = []
        moving_objs: typing.List[int] = []

        
        all_t0_diffs = np.full((6, 148), -1.0, dtype=np.float64)
        all_q0_diffs = np.full((6, 148), -1.0, dtype=np.float64)

        obj_np_ind = 0
        for obj_num, obj_poses in obj_iter:
            if obj_num <= 0:
                continue
            obj_quats = pm.normalizeAll(obj_poses[:, :4])
            obj_translations = obj_poses[:, 4:]

            # Testing if object is static.            
            # q_diffs = pm.anglesBetweenQuats(obj_quats[1:], obj_quats[:-1])
            q0_diffs = pm.anglesBetweenQuats(
                obj_quats[2:],
                np.broadcast_to(obj_quats[0], obj_quats[2:].shape)
            )
            # t_disps = np.diff(obj_translations, 1, axis=0)
            # t_diffs = np.linalg.norm(t_disps, axis=-1)
            t0_disps = obj_translations[2:] - obj_translations[0]
            t0_diffs = np.linalg.norm(t0_disps, axis=-1)

            # q_under = np.all(q_diffs < static_rot_thresh)
            # t_under = np.all(t_diffs < static_trans_thresh)
            q0_under = np.all(q0_diffs < static_rot_thresh)
            t0_under = np.all(t0_diffs < static_trans_thresh)

            if q0_under and t0_under: # and q_under and t_under:
                static_objs.append(obj_num)

                if not static_obj_found:
                    static_obj_found = True
            else:
                moving_objs.append(obj_num)

                moving_o2w_rmat_list.append(pm.matsFromQuaternions(obj_quats))
                moving_o2w_t_list.append(obj_translations)
                moving_o2w_aa_list.append(
                    pm.axisAngleVec3sFromQuats(obj_quats, True)
                )

                # key = "clip{:06d}-obj{:02d}-aa-translation".format(clip, obj_num)

                # [cRw, cTw] * [wRo, wTo] = [cRwwRo, cRw * wTo + cTw]
                cRo = pm.normalizeAll(pm.multiplyQuatLists(cRw, obj_quats))
                cTo = pm.rotateVecsByQuats(cRw, obj_translations) + cTw

                moving_o2c_t_list.append(cTo)
                moving_o2c_rmat_list.append(pm.matsFromQuaternions(cRo))
                moving_o2c_aa_list.append(pm.axisAngleVec3sFromQuats(cRo, True))

            all_t0_diffs[obj_np_ind] = t0_diffs
            all_q0_diffs[obj_np_ind] = q0_diffs

            all_max_t0_diffs[clip_np_ind, obj_np_ind] = np.max(t0_diffs)
            all_max_q0_diffs[clip_np_ind, obj_np_ind] = np.max(q0_diffs)

            obj_np_ind += 1

        # if not static_obj_found:
        #     n_objs = len(obj_data.keys())
        #     least_trans = np.argmin(np.mean(all_t0_diffs[:n_objs], axis=-1))
        #     least_q = np.argmin(np.mean(all_q0_diffs[:n_objs], axis=-1))
        #     print("Max translation:", all_t0_diffs[least_trans].max())
        #     print("Frame, obj:", np.argmax(all_t0_diffs[least_trans]) + 2, least_trans)
        #     print("Max rotation:", all_q0_diffs[least_q].max())
        #     print("Frame, obj:", np.argmax(all_q0_diffs[least_q]), least_q)
        #     raise Exception("No static obj found for clip {}!".format(clip))
        
        obj_ind_counter = 0
        for obj_num in moving_objs:
            obj_ids[clip_np_ind, obj_ind_counter] = obj_num
            obj_ind_counter += 1
        moving_obj_counts[clip_np_ind] = obj_ind_counter
        for obj_num in static_objs:
            obj_ids[clip_np_ind, obj_ind_counter] = -obj_num
            obj_ind_counter += 1
    moving_obj_base_inds = np.cumsum(moving_obj_counts, dtype=np.int32)

    if moving_obj_base_inds[-1] != len(moving_o2c_t_list):
        raise Exception("Unexpected cardinality mismatch!")
    
    moving_obj_base_inds = np.pad(moving_obj_base_inds, (1, 0))

    # Compile our objects into something we return.
    ret_dict = {
        'obj_ids': obj_ids,
        'moving_data_base_ind_per_clip': moving_obj_base_inds,
        'w2c_translations': w2c_t_array,
        'w2c_axisangles': w2c_aa_array, 'w2c_rmats': w2c_rmat_array, 
        'o2w_translations': moving_o2w_t_list,
        'o2w_axisangles': moving_o2w_aa_list, 'o2w_rmats': moving_o2w_rmat_list,
        'o2c_translations': moving_o2c_t_list,  
        'o2c_axisangles': moving_o2c_aa_list, 'o2c_rmats': moving_o2c_rmat_list
    }
    for k, v in ret_dict.items():
        if not isinstance(v, np.ndarray):
            ret_dict[k] = np.stack(v, axis=0)

        
    return ret_dict, all_max_t0_diffs, all_max_q0_diffs
#%%
rconp_a, t_diffs_for_hist, q_diffs_for_hist = results_to_np(
    results_cam_obj_q, ALL_REF_CAM_IDXS[SELECTED_CAM], 0.05, np.deg2rad(5)
)

#%%
# The `diffs_for_hist` variables are of shape `(num_clips, 6)`, as there are
# up to 6 objects per scene, and represent the maximum displacement (translation
# for "t", rotation radians for "q") from the respective object's frame 0 pose
# in the respective clip. I filter for nonnegative values because if there are
# less than 6 objects in a scene, -1 is used as a placeholder value. 
t_flats = t_diffs_for_hist.flatten()
t_nz_flats = t_flats[t_flats >= 0.0]
q_flats = q_diffs_for_hist.flatten()
q_nz_flats = q_flats[q_flats >= 0.0]

stack_nz_flats = np.vstack((t_nz_flats, q_nz_flats))
stack_flats = np.vstack((t_flats, q_flats))

def thing():
    # keys01 = tuple(k for k in rconp.keys() if "2201" in k)
    aa_ts = rconp[keys01[w]]
    aa_mats = np.repeat([np.eye(4)], 150, axis=0)
    aa_mats[:, :3, 3] = aa_ts[:, 3:]
    aas = aa_ts[:, :3]
    aa_mats[:, :3, :3] = pm.matsFromScaledAxisAngleArray(aas)
 

#%%
# The below was an attempt to use GMMs to determine which objects in a scene
# were static and which were moving. But it made a lot of misclassifications.
from sklearn import mixture
n_comps = 2
clf_t = mixture.GaussianMixture(n_comps)
clf_t.fit(t_nz_flats.reshape(-1, 1))
clf_q = mixture.GaussianMixture(n_comps)
clf_q.fit(q_nz_flats.reshape(-1, 1))
#%%

gmm_pred_t = clf_t.predict(t_flats.reshape(-1,1))
gmm_pred_q = clf_q.predict(q_flats.reshape(-1,1))

largest_t_comp = np.argmax(clf_t.means_.flatten())
largest_q_comp = np.argmax(clf_q.means_.flatten())

gmm_pred = (gmm_pred_t == largest_t_comp) & (gmm_pred_q == largest_q_comp)
gmm_pred = gmm_pred.reshape(-1, 6)


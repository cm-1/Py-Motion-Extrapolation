

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

ALL_REF_CAM_IDXS = {CAM_TYPE.ARIA: 2}

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

hot_dir_name = "train_aria" if SELECTED_CAM == CAM_TYPE.ARIA else "train_quest3"

# Example usage:
# base_dir should be the path containing your .zip files
results_cam_obj_q = process_zips(
    r"D:\Datasets\HOT3D", hot_dir_name + "_pt", hot_dir_name,
    ALL_CAM_KEYS[SELECTED_CAM]
)

#%%
import posemath as pm

def results_to_np(res: typing.Dict[int, typing.Tuple[NDArray, typing.Dict[int, NDArray]]],
                  ref_cam_idx: int):
    ret_dict: typing.Dict[str, NDArray] = dict()
    for clip, (cam_data, obj_data) in res.items():
        cam_quats = cam_data[:, ref_cam_idx, :4]
        cam_translations = cam_data[:, ref_cam_idx, 4:]

        # Above are cam-to-world. We instead want:
        cRw = pm.conjugateQuats(cam_quats)
        cTw = pm.rotateVecsByQuats(cRw, -cam_translations)


        for obj_num, obj_poses in obj_data.items():
            key = "clip{:06d}-obj{:02d}-aa-translation".format(clip, obj_num)
            obj_quats = obj_poses[:, :4]
            obj_translations = obj_poses[:, 4:]

            # [cRw, cTw] * [wRo, wTo] = [cRwwRo, cRw * wTo + cTw]
            cRo = pm.multiplyQuatLists(cRw, obj_quats)
            cTo = pm.rotateVecsByQuats(cRw, obj_translations) + cTw

            cRo_aa = pm.axisAngleVec3sFromQuats(cRo)

            ret_dict[key] = np.concatenate((cRo_aa, cTo), axis=-1)
    return ret_dict

rconp_q = results_to_np(results_cam_obj_q, 0) #ALL_REF_CAM_IDXS[SELECTED_CAM])

def thing():
    # keys01 = tuple(k for k in rconp.keys() if "2201" in k)
    aa_ts = rconp[keys01[w]]
    aa_mats = np.repeat([np.eye(4)], 150, axis=0)
    aa_mats[:, :3, 3] = aa_ts[:, 3:]
    aas = aa_ts[:, :3]
    aa_mats[:, :3, :3] = pm.matsFromScaledAxisAngleArray(aas)
 


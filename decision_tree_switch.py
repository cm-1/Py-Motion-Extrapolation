#%% Imports
import copy
import time
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

# Decision tree imports ========================================================
from sklearn import tree as sk_tree
from sklearn.tree._tree import TREE_LEAF

from custom_tree.weighted_impurity import WeightedErrorCriterion

# Local code imports ===========================================================
# For reading the dataset into numpy arrays:
from gtCommon import PoseLoaderBCOT
import gtCommon as gtc


# MOTION_MODEL is an enum that represents some physical non-ML motion prediction
# schemes like constant-velocity, constant-acceleration, etc.
from motiontools.key_and_vec_specs import MOTION_MODEL, SpecifiedMotionData

from motiontools.dataorg import (
    DataOrganizer, SkipSubsetKind, UnitAwareScaler, SUBSET_PRESET
)

from datatools.data_splitting import DataSubsetKind


# End of imports
# ==============================================================================

USE_WEIGHTED_CRIT = True

dog = DataOrganizer.load(PoseLoaderBCOT) # Load our data.

bcot_test_ids = dog.subset_ids[DataSubsetKind.TEST] # Find test data subset.

#%% BEST-CASE SCENARIO CLASSIFICATION CALCULATIONS

# First, calculate what happens when we use only one classification per video,
# but said classification is the best possible one.
# TODO: Fix the below again so that calculation across ALL videos supported!
# Currently just does the test set!
test_bodies = np.unique([c[0] for c in bcot_test_ids])
# best_seq_means = []
test_best_seq_means = []
for skip in range(3):
    # best_seq_scores = []
    test_best_seq_scores = []
    for seq in range(len(gtc.BCOT_SEQ_NAMES)):
        seq_data = []
        test_seq_data = []
        for combo in dog.subset_ids[DataSubsetKind.TEST]:
            if combo[1] == seq:
                seq_combo_scores_stacked = dog.getSelectionData(
                    dog.concat_test_class_errs, DataSubsetKind.TEST,
                    SkipSubsetKind(skip), combo[:2]
                )
                seq_data.append(seq_combo_scores_stacked)
                if combo[0] in test_bodies:
                    test_seq_data.append(seq_combo_scores_stacked)
        if len(seq_data) > 0:
            concat_seq_data = np.concatenate(seq_data, axis=0)
            concat_test_seq_data = np.concatenate(test_seq_data, axis=0)
            seq_sums = np.sum(concat_seq_data, axis=0)
            assert seq_sums.shape == (len(MOTION_MODEL), )
            assert concat_seq_data.shape[1] == len(MOTION_MODEL)
            
            seq_best_mode = np.argmin(seq_sums)
            # best_seq_scores.append(concat_seq_data[:, seq_best_mode])
            test_best_seq_scores.append(concat_test_seq_data[:, seq_best_mode])
    # seq_best_mean = np.mean(np.concatenate(best_seq_scores))
    seq_best_test_mean = np.mean(np.concatenate(test_best_seq_scores))
    
    # best_seq_means.append(seq_best_mean)
    test_best_seq_means.append(seq_best_test_mean)
print("Best case one-model-per sequence results for skips 0, 1, 2:")
# print("All data:", best_seq_means)
print("Test data:", test_best_seq_means)


# Next, we consider per-frame perfect classification rather than per-video.
# I.e., find the "lower bound" for error when we train a classifier to use the
# physics models for prediction, by finding pose MAE for perfect classification.
error_lim_all = dog.getClassErrsTest(dog.concat_test_labels)
error_lim = dog.getClassScoresTest(dog.concat_test_labels)
print("Classification MAE limit:", error_lim)

#%% Custom tree impurity criterion setup.
# A tree depth of 8 is already way beyond "human-readable", and I think the
# graphs don't show miraculous improvements past 8, so 8 seems like a good max.
max_depth = 8

# Determine the number of unique motion models (classes).
motion_mod_len = len(MOTION_MODEL)

# Ensure that the shape of the concatenated training class errors is correct.
assert motion_mod_len == dog.concat_train_class_errs.shape[1], "Wrong shape!"

# Initialize a custom criterion with the appropriate parameters.
mc = WeightedErrorCriterion(1, np.array([motion_mod_len], dtype=np.intp))

# Reshape because custom criterion supports general case where we may have
# multiple output classes, on the 2nd axis. Here we just have 1.
class_errs_shape = dog.concat_train_class_errs.shape
y_errs_reshape = dog.concat_train_class_errs.reshape((
    class_errs_shape[0], 1, class_errs_shape[1]
))
mc.set_y_errs(y_errs_reshape)

nonco_cols, nonco_col_ks = dog.maskAndKeysForSubsetPreset(SUBSET_PRESET.FAST_TO_EXPLAIN)

non_smd_key_names = [
    k.name for k in nonco_col_ks if not isinstance(k, SpecifiedMotionData)
]
print("\n".join(non_smd_key_names))

bcs_scaler = UnitAwareScaler(nonco_col_ks)
bcs_scaler.fakeFit()
dog.setPickAndTransform(nonco_cols, bcs_scaler)

#%% Training decision tree at max depth.
# ---
# We'll train a tree at max depth and then trim it to smaller depths to evaluate
# the performance at lower depths. This yields the exact same trees as if we
# were to train individual ones with lower max depths (confirmed via tests), but
# eliminates duplicated training time. I might leave the old code for training
# individual trees below as a comment, for reference.

# Initialize the decision tree classifier with the custom criterion.
tree_crit = mc if USE_WEIGHTED_CRIT else "gini"
big_tree = sk_tree.DecisionTreeClassifier(max_depth=max_depth, criterion=tree_crit)

print("Starting decision tree training!")
start_time = time.time()

# Create an array to hold all possible labels. This is required by sklearn
# even though our custom criterion does not use these labels for impurity
# calculations.
all_possible_labels = dog.concat_train_labels.copy()

# Check if there are enough data rows to represent all classes.
if USE_WEIGHTED_CRIT and len(all_possible_labels) >= motion_mod_len:
    # Fill the first `motion_mod_len` elements with all possible class labels.
    all_possible_labels[:motion_mod_len] = np.arange(motion_mod_len)
elif USE_WEIGHTED_CRIT:
    # Raise an exception if there are fewer data rows than classes because
    # sklearn needs an input array of labels containing every possible class or
    # else my custom criterion crashes. This is a limitation due to how sklearn
    # initializes internal arrays.
    raise Exception((
        "Fewer data rows than classes, which is problematic because sklearn "
        "needs an input array of labels containing every possible class or "
        "else my custom criterion crashes. As a fix, could duplicate all rows "
        "until there are enough but then my custom criterion also does not "
        "take ownership yet of the values passed in so if it's an ephemeral "
        "copy that's an issue, but if I always take ownership I'd want to be "
        "careful that I am not duplicating any data by mistake for when I work "
        "with huge datasets. So take care if implementing a fix here!"
    ))


big_tree = big_tree.fit(dog.col_subset_train, all_possible_labels)
print("Done!")
print("Time spent:", time.time() - start_time)
#%%

# A recursive function to help trim_to_depth(). The params should be a tree's
# children_left and children_right, then a specified node's index and its
# depth, and a target tree depth. 
def _trim_to_depth_helper(left_children_inds: NDArray[np.signedinteger],
                          right_children_inds: NDArray[np.signedinteger],
                          current_node_index: int, current_node_depth: int,
                          target_depth: int):
    if current_node_depth >= target_depth:
        left_children_inds[current_node_index] = TREE_LEAF
        right_children_inds[current_node_index] = TREE_LEAF
    else:
        # Recurse left and right subtrees to set required nodes to leaves.
        _trim_to_depth_helper(
            left_children_inds, right_children_inds, 
            left_children_inds[current_node_index], current_node_depth + 1,
            target_depth
        )
        _trim_to_depth_helper(
            left_children_inds, right_children_inds, 
            right_children_inds[current_node_index], current_node_depth + 1,
            target_depth
        )
    return
        
def trim_to_depth(tree: sk_tree._classes.DecisionTreeClassifier, depth):
    assert depth >= 1, "Depth to trim tree to must be >= 1!"
    depth = min(depth, tree.get_depth())
    tree_copy = copy.deepcopy(tree)  # Make a copy to avoid modifying original.
    lefts = tree_copy.tree_.children_left
    rights = tree_copy.tree_.children_right

    _trim_to_depth_helper(lefts, rights, 0, 0, depth)
    return tree_copy

#%%
scores = defaultdict(lambda : np.empty(max_depth))
depths = np.arange(1, max_depth + 1)
for d in depths:
    # Preiously, we trained new trees from scratch using the below code, but as
    # described above, we'll instead trim the "main" tree to get new ones.
    #         clf = sk_tree.DecisionTreeClassifier(max_depth=d, criterion=mc)
    #         clf = clf.fit(concat_train_data, concat_train_labels)
    trimmed_tree = trim_to_depth(big_tree, d)
    graph_pred = trimmed_tree.predict(dog.col_subset_test)
    scores_for_depth = dog.getClassScoresTest(graph_pred)
    for k, score_for_depth in scores_for_depth.items():
        scores[k][d-1] = score_for_depth

print("Scores for trimmed trees:")
print(scores)
#%%
import matplotlib.pyplot as plt

# Plot decision tree test score errors for the different depths.
for k, score_sub in scores.items():
    #score_normed = (score_sub - score_sub.min()) / np.ptp(score_sub)
    curr_plt_ln, = plt.plot(depths, score_sub, label=k)
    err_lim_k = error_lim[k]
    curr_plt_col = curr_plt_ln.get_color()
    plt.plot(
        [1, max_depth], [err_lim_k, err_lim_k], color=curr_plt_col, dashes=[1,1]
    )
plt.legend()
plt.ylabel("Test Set Error")# (normed to [0,1])")
plt.xlabel("Max decision tree depth")
plt.show()

#%%
# Again, we'll replace old code with a trimming of our main tree.
#         mclf = sk_tree.DecisionTreeClassifier(max_depth=4, criterion=mc)
#         mclf = mclf.fit(concat_train_data, concat_train_labels)
mclf = trim_to_depth(big_tree, 3)

mclfps = mclf.predict(dog.col_subset_test).copy()

#%%

mc_errs = dog.getClassScoresTest(mclfps)
print("Error for my decision tree=", mc_errs)
print("Done!", mclfps)

def getTreeDataTrainScore(tree, data_organizer: DataOrganizer):
    p = tree.predict(data_organizer.concat_train_data)
    return data_organizer.getClassScoresTrain(p)
def getTreeDataTestScore(tree, data_organizer: DataOrganizer):
    p = tree.predict(data_organizer.concat_test_data)
    return data_organizer.getClassScoresTest(p)

#%%
from sklearn.tree import export_graphviz
import pathlib

tree_path = pathlib.Path(__file__).parent.resolve() / "results" / "tree.dot"
feature_names = [e.name for e in nonco_col_ks]
class_names = [str(i) for i in range(1, len(MOTION_MODEL) + 1)]

export_graphviz(
    mclf, out_file=str(tree_path), 
    feature_names=feature_names, 
    class_names=class_names,
    filled=True, rounded=True, special_characters=True,
    
)
# Convert to .pdf with:
# Graphviz\bin\dot.exe -Tpdf tree.dot -o tree.pdf

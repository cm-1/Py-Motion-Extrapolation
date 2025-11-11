# DataForJAV Caching Integration Plan
The below was generated during an experiment on using Cline with Claude Console
(specifically, Claude Sonnet 4.5 via an API key).

Together, they were able to execute part of the implementation but failed at
doing the rest. To get a better sense of their capabilities, I wanted to
continue trying a little bit more but I'm facing technical difficulties in
buying more API credits.

Probably for the best, though. I think I was starting to fall into the
METR-observed trap where LLM coding tools can actually slow a developer down.
So I'll try finishing the implementation myself. I think the design and current
work is decent, though, so I won't discard it and will keep it for reference.


## Overview
This document provides a detailed plan for integrating the caching system into `DataForJAV.__init__()` to save and load expensive computed arrays.

## Files to Modify
- `multi_frame_nn.py` - Contains `DataForJAV` class that needs caching integration
- `nn_utilities/data_cache.py` - Caching utilities (cache key generation, save/load functions)

## What Gets Cached
Right now, we only create the columns we need for the current OutVecMode param
passed in. But a lot of very similar data gets created for different OutVecMode
values, so now that we will be caching, we should compute and cache *all* of the
data we could possibly need regardless of OutVecMode and then, on loading, we
just select what we need. Otherwise, we would be storing too much redundant
data!

The expensive operations that should be cached are:
1. **World coordinate columns** computed in `_world_coord_cols()`:
   - If `w2l_rotations` is not provided, this consists of `other_coords`,
     `other_vecs`, and `w2l_thing`. We can call these `w2l_extra_cols`.
   - If `w2l_rotations` is provided, this consists of transformed versions of
     the above plus the columns in `(coords, rmatv9s)`. We can call all of these
     `no_w2l_extra_cols`.
   - We should save both cases but *not* save `curr_cols` in *either* case,
     because this is data that is *already* stored/saved in other code and would
     be redundant.

2. **Ground truth arrays** (`_gt_arrs`) for each OutVecMode and DataSubsetKind.

3. **ROT_FIXED_AX mode specific data**:
   - `_prev_vel_ax_concat`
   - `_prev_ang_concat`
   - `_gt_rot_vels`

## What Does NOT Get Cached
These are fast to compute or need to be done each run:
- Scaling/shifting with `bcs_scaler` (operation (a) from requirements)
- JAV calculations from `dataForCombosJAV`.
  - These are only ever used to create the data we are currently caching and
    are never intended to be accessed after `__init__`. So if we cache the above
    mentioned arrays, we have no need to create/cache these JAV calcuations.
- Reference predictions
- Final concatenation of columns
- Skip filtering (done after loading cache)

## Integration Strategy

### Step 1: Restructure `_world_coord_cols()`
Split this method into two parts:

```python
def _compute_world_coord_components(self, subset_kind: DataSubsetKind):
    """Compute expensive world coordinate components (cacheable)"""
    # Returns a dictionary with all the expensive computed arrays
    # This is what gets cached
    pass

def _world_coord_cols(self, curr_cols: NDArray, subset_kind: DataSubsetKind,
                      cached_components=None, w2l_rotations=None):
    """Assemble columns from components (fast, done each run)"""
    # If cached_components provided, use them
    # Otherwise call _compute_world_coord_components()
    # Then concatenate with curr_cols
    pass
```

### Step 2: Modify `__init__()` Cache Logic

Replace the current cache initialization with:

```python
# Generate cache key for expensive operations
cache_key = None
cached_data = None
if use_cache:
    cache_key = generate_cache_key(
        data_organizer.subset_ids,
        save_data_for_conf
    )
    cached_data = load_cached_data(
        cache_key, outVecMode, skip, data_organizer.subset_skip_inds
    )
    if cached_data is not None:
        print(f"Using cached data (key: {cache_key})")
```

### Step 3: Conditional Computation Path

First, figure out whether JAV data needs to be computed and put a conditional on
its computation, since it's only required if we're creating a new cache.

Then, after (possibly) computing JAV data and before the main
`if self.outVecMode in WORLD_VEC_MODES or _rot_align or _rot_output:` block:

```python
# Expensive computations that may be cached
world_coord_components = None
gt_arrays_unfiltered = None
rot_fixed_ax_data = None
data_to_cache = dict()

gt_arrays_key = "gt_arrays_" + self.outVecMode.name

if cached_data is not None:
    gt_arrays_unfiltered = cached_data[gt_arrays_key]
else:
    # Compute GT arrays
    gt_arrays_unfiltered = self._compute_gt_arrays_unfiltered(# existing parameters)
    if use_cache:
        data_to_cache = {
            world_coords_key: world_coord_components,
            gt_arrays_key: gt_arrays_unfiltered
        }

if self.outVecMode in WORLD_VEC_MODES or _rot_align or _rot_output:
    world_coords_key = 'w2l_extra_vecs' if _rot_align else 'no_w2l_extra_vecs'
    if cached_data is not None:
        # Use cached data
        world_coord_components = cached_data[world_coords_key]
        if self.outVecMode == OutVecMode.ROT_FIXED_AX:
            rot_fixed_ax_data = cached_data.get('rot_fixed_ax_data')
    else:
        # Compute fresh (expensive path)
        world_coord_components = {}
        for k in DataSubsetKind.nonWholeValues():
            world_coord_components[k] = self._compute_world_coord_components(k)
        
                
        # Save to cache if enabled
        if use_cache:
            if self.outVecMode == OutVecMode.ROT_FIXED_AX:
                rot_fixed_ax_data = self._compute_rot_fixed_ax_data(# existing parameters)
                data_to_cache['rot_fixed_ax_data'] = rot_fixed_ax_data
    # Now use the components (whether from cache or freshly computed)
    for k, _w2l_mats_prev_sub in _w2l_mats_prev.items():
        self._in_arrs[k] = self._world_coord_cols(
            self._in_arrs[k], k, 
            cached_components=world_coord_components[k],
            w2l_rotations=_w2l_mats_prev_sub
        )

if use_cache:
    save_cached_data(cache_key, data_to_cache)
    print(f"Cached data saved (key: {cache_key})")
    

self._gt_arrs = gt_arrays_unfiltered

if self.outVecMode == OutVecMode.ROT_FIXED_AX:
    self._prev_vel_ax_concat = rot_fixed_ax_data['prev_vel_ax_concat']
    self._prev_ang_concat = rot_fixed_ax_data['prev_ang_concat']
    self._gt_rot_vels = rot_fixed_ax_data['gt_rot_vels']
```

### Step 4: Skip Filtering
The skip filtering that currently happens near the end of `__init__()` should work unchanged, since it operates on `self._in_arrs` and `self._gt_arrs` regardless of how they were created.

## Implementation Details

### Helper Methods to Add

```python
def _compute_world_coord_components(self, subset_kind: DataSubsetKind) -> dict:
    """Extract expensive computation logic from _world_coord_cols"""
    # All the _worldvec_helper calls
    # Return dict with all computed arrays
    pass

def _compute_gt_arrays_unfiltered(self, # existing parameters) -> dict:
    """Compute ground truth arrays before skip filtering"""
    # The _worldvec_concats logic for GT
    # Return dict by subset kind
    pass

def _compute_rot_fixed_ax_data(self, # existing parameters) -> dict:
    """Compute ROT_FIXED_AX specific data"""
    # The _prev_vel_ax_concat, _prev_ang_concat, _gt_rot_vels logic
    # Return dict with these arrays
    pass
```

## Step Dependencies

The implementation steps must be followed in order due to these dependencies:

- **Step 1** must be completed first, as Steps 2 and 3 depend on the restructured `_world_coord_cols()` method
- **Step 2** depends on Step 1, as it uses the new method signatures
- **Step 3** depends on Steps 1 and 2, as it orchestrates the caching logic using the refactored methods
- **Step 4** is verification only and requires Steps 1-3 to be complete

Within Step 3, the helper methods (`_compute_world_coord_components()`, `_compute_gt_arrays_unfiltered()`, `_compute_rot_fixed_ax_data()`) can be created in any order, but all must exist before the main conditional computation logic that calls them.

## Implementation Checklist

### Step 1: Refactor `_world_coord_cols()` 
- [x] Create new method `_compute_world_coord_components(subset_kind: DataSubsetKind) -> dict`
  - ✅ Method extracts all expensive computation logic from `_world_coord_cols()`
  - ✅ Returns dict with components: coords, coords_tm1-3, vels, accs, jerks, aas, rmatv9s, rot_vels_unscaled, rot_accs, rot_jerks, w2ls_concat
- [x] Modify `_world_coord_cols()` to accept `cached_components` parameter
  - ✅ Parameter added with default value of None
- [x] Update `_world_coord_cols()` to use cached components when provided
  - ✅ Method checks if `cached_components` is None and either uses cache or computes fresh

### Step 2: Add Cache Initialization to `__init__()`
- [x] Add cache key generation logic
  - ✅ Cache key generated with `data_organizer.subset_ids` and `save_data_for_conf`
- [x] Add cache loading logic with appropriate key
  - ✅ Calls `load_cached_data()` with cache_key, skip, and subset_skip_inds
- [x] Add conditional print statement for cache usage
  - ✅ Prints "Using cached data (key: {cache_key})" when cache is loaded

### Step 3: Implement Conditional Computation Path
- [x] Add conditional for JAV data computation (only if creating new cache)
  - ✅ Added `need_jav_computation = cached_data is None` flag
  - ✅ Wrapped JAV computation in `if need_jav_computation:` block
  - ⚠️ **ISSUE**: Current code will fail when `cached_data` is not None because `jav_res` will be None and subsequent code tries to access it
- [ ] Add world coordinate components caching logic
  - [ ] Load from cache if available
  - [ ] Otherwise compute via `_compute_world_coord_components()`
  - ⚠️ **TODO**: Need to add conditional logic in the main `if self.outVecMode in WORLD_VEC_MODES or _rot_align or _rot_output:` block
- [ ] Add GT arrays caching logic
  - [ ] Create helper method `_compute_gt_arrays_unfiltered()`
  - [ ] Load from cache if available or compute fresh
  - ⚠️ **TODO**: Helper method not yet created
- [ ] Add ROT_FIXED_AX specific data caching
  - [ ] Create helper method `_compute_rot_fixed_ax_data()`
  - [ ] Load from cache if available or compute fresh
  - ⚠️ **TODO**: Helper method not yet created
- [ ] Add cache save logic at end of conditional block
  - ⚠️ **TODO**: Need to add `save_cached_data()` call with appropriate `data_to_cache` dict

### Step 4: Verify Skip Filtering
- [ ] Confirm skip filtering logic works unchanged with cached data
  - ⚠️ **BLOCKED**: Cannot verify until Step 3 is complete

## Current Status Summary

**Completed Work:**
- ✅ Cache utility functions updated and working
- ✅ `_world_coord_cols()` successfully refactored with cacheable component extraction
- ✅ Cache initialization and loading integrated into `__init__()`
- ✅ JAV computation made conditional on cache availability

**Critical Issues to Fix:**
1. **JAV data access when cached**: Code currently sets `jav_res = None` when using cache, but later code tries to access `jav_res` elements. Need to either:
   - Load JAV data from cache when available, OR
   - Restructure code to not depend on `jav_res` when using cache
   
2. **Missing helper methods**: Need to create:
   - `_compute_gt_arrays_unfiltered()` 
   - `_compute_rot_fixed_ax_data()`

3. **Incomplete caching logic**: The main conditional computation path (Step 3) needs:
   - World coordinate components caching/loading
   - GT arrays caching/loading  
   - ROT_FIXED_AX data caching/loading
   - Final `save_cached_data()` call

**Next Steps for Developer:**
1. Decide caching strategy for JAV-dependent data (either cache JAV data itself, or cache all downstream products)
2. Implement the full conditional computation path in Step 3
3. Create the missing helper methods
4. Add the cache save logic
5. Test with and without cache to verify correctness

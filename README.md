# Intro

Some code for motion extrapolation investigation.

# Dependencies
They are: `tensorflow, keras, scipy, scikit-learn, matplotlib, numpy`.

I believe these are all dependencies of tensorflow, but I'm not 100% sure.

I also use things from Python's standard library that only exist in Python
version 3.7 and later.

# Running the Code

There are three neural networks:

- World-space LSTM: code in `world_frame_lstm.py`.
- Velocity-aligned-space LSTM: code in `velocity_frame_lstm.py`.
- "Vanilla" velocity-aligned-space NN: one first must run `data_generator.py` to generate the data required, then run `velocity_frame_nn.py` for the actual NN.
  - This is currently the best-performing network.

All other files, such as `posemath.py`, `minjerk.py`, etc. are just "helper"
files. Of these, the only one that _might_ be worth reading through is `data_by_combo_functions.py`,
as it contains the most documentation and used to be in my "one big file"
containing all networks. However, I try to use docstrings for most of these
"helper" files so you should be able to see the documentation for each
function via your IDE from the files you actually run by hovering your 
cursor over the calls to the imported function/class/etc.

# About the Dataset

The source dataset contains a bunch of videos split up by "sequence" and "body".
Each "sequence" is a different set of lighting, motion, and background-clutter
choices, while each body is a 3D-printed object in the scene whose motion is
tracked. So the dataset has a bunch of folders for sequence, subfolders for
body, and then a bunch of video frames and a poses .txt for the respective 
video. E.g.:
```
easy_static_suspension/
  Driller/
     frame000.png
     frame001.png
        ...
     pose.txt
  Teapot/
     frame000.png
        ...
     pose.txt
easy_static_handheld/
  Driller/
     ...
  Teapot/
     ...
```

So, each (sequence, body) **combo** represents a unique video in the dataset.
As such, a lot of my code uses the word **"combo"** to denote this "ID"; I 
should probably have actually used "vidID" or something for the name, but now
I'd have to rename so many things for it to work.

In the "dataset", each sequence and body are a string like "easy\_static\_handheld" or "Teapot", but when I use the combos as dictionary
keys, I instead use a tuple of ints (indices, I guess) for each combo, where 
the index-to-name relationship is defined in `gtCommon.py`. 

Sometimes I'll add a third field to the "combo" to get (sequence, body, motion\_kind)
for ease of filtering combos in the non-regression-network code, but this third
field is just a "subset" of the info encoded in the string for the sequence.
Thus, when actually doing dictionary accesses, I'll just take the first two
parts of the key with a `[:2]` slice.

## "Combo" Summary:
Unique identifier "**combo**" for a video: `(str, str)`

"Encoded" unique identifier for a video: `(int, int)`

"Supplemented"/"redundant" identifier for a video for sorting in other/later code: `(int, int, str)`






# Dependencies

Essentially all files require numpy, and many files require matplotlib.

Decision tree training requires sklearn, and to use the custom impurity criterion,
cython and setuptools are needed.

Neural net code requires tensorflow, but you could just run the cells above that.
Some optional analysis of the networks uses the shap library.

Min jerk predictions require scipy.

One of the newer 3D plots of neural net outputs needs Plotly.

Code that requires sympy or that is meant to be used with Blender is currently
separated into their own folders.

## TensorFlow to TFLite conversion
I've had best success with Python 3.10 and TensorFlow 2.19.0.

## TensorFlow to ONNX conversion
It took a few tries to get a working venv for this. I finally got it working
with python 3.12.12, tensorflow 2.16.1, and tf2onnx 1.16.0. The full pip freeze
is:
```
absl-py==2.3.1
astunparse==1.6.3
certifi==2025.11.12
charset-normalizer==3.4.4
flatbuffers==25.9.23
gast==0.7.0
google-pasta==0.2.0
grpcio==1.76.0
h5py==3.15.1
idna==3.11
keras==3.12.0
libclang==18.1.1
Markdown==3.10
markdown-it-py==4.0.0
MarkupSafe==3.0.3
mdurl==0.1.2
ml-dtypes==0.3.2
namex==0.1.0
numpy==1.26.4
onnx==1.17.0
opt_einsum==3.4.0
optree==0.18.0
packaging==25.0
protobuf==3.20.3
Pygments==2.19.2
requests==2.32.5
rich==14.2.0
setuptools==80.9.0
six==1.17.0
tensorboard==2.16.2
tensorboard-data-server==0.7.2
tensorflow==2.16.1
tensorflow-intel==2.16.1
termcolor==3.2.0
tf2onnx==1.16.0
typing_extensions==4.15.0
urllib3==2.6.0
Werkzeug==3.1.4
wheel==0.45.1
wrapt==2.0.1
```

Then for actually _running_ the generated .onnx file, I made a separate
environment with Python 3.12.12 and onnxruntime 1.23.2. This worked on the first
try, so maybe it's less finicky.

## Older Dependency Versions

There are a few places where I try to accomodate older versions of numpy or
older versions of joblib (a dependency of sklearn).

For numpy, the inline comments explain it; it has to do with numpy switching
which module one should use for string arrays in newer versions.

For joblib, I had to accomodate a venv where I have Python 3.7 for running
tensorflow-gpu on Windows. Unfortunately, newer joblib versions require
Python 3.8. So that meant downgrading joblib to an older version (1.2), but said
version did not have `parallel_config`, which I think is probably a good idea
to use when possible. So I wrote some code to use it if the joblib version is
new enough, but still allow the old joblib for that one venv.

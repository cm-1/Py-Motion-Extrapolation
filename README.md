# Intro

Some code for motion extrapolation investigation.

# Dependencies

Essentially all files require numpy, and many files require matplotlib.

Decision tree training requires sklearn, and to use the custom impurity criterion,
cython and setuptools are needed. Some of the code lower down in the file
requires tensorflow, but you could just run the cells above that.

Min jerk predictions require scipy.

Code that requires sympy or that is meant to be used with Blender is currently
separated into their own folders.

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


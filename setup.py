"""
Setup configuration for Py-Motion-Extrapolation package.

This allows the project to be installed in development mode with:
    pip install -e .

After installation, you can use imports like:
    import posemath as pm
from anywhere without worrying about paths or cwd.
"""

from setuptools import setup, find_packages

setup(
    name="motion-extrapolation",
    version="0.1.0",
    description="Motion extrapolation investigation code",
    author="Your Name",
    python_requires=">=3.7",
    packages=find_packages(),
    py_modules=[
        "posemath",
        "gtCommon",
        "poseextrapolation",
        "minjerk",
        "curvetools",
        "bspline",
        "cinpact",
        "errorstats",
        "data_generator",
        "data_by_combo_functions",
        "multi_frame_nn",
        "world_frame_fcnn",
        "world_frame_lstm",
        "velocity_frame_lstm",
        "rnn_models",
        "decision_tree_switch",
        "motionExperiments",
        "spline_approximation",
        "all_bspline_param_eval",
        "gtPlotter",
        "accPlots",
        "axisAngleTesting",
        "bayes_opt_cinpact",
        "bayes_opt_sigmoid",
        "bounce_eval",
        "boyleBlackHoleBinary",
        "bspline_bug_tests",
        "cinpactTests",
        "fitting_bsplines",
        "hypothetical_point_vis",
        "JAV_confidence_intervals",
        "rotationTests",
        "testingminjerk",
    ],
    install_requires=[
        "tensorflow",
        "keras",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "numpy",
    ],
    extras_require={
        "optional": [
            "plotly",
            "shap",
        ],
    },
)

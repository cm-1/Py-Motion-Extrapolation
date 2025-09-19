import typing

import numpy as np
from numpy.typing import NDArray

import plotly.graph_objects as go


def getColourMags(input_pts: NDArray, ref_pt: NDArray):
    mags = np.linalg.norm(input_pts - ref_pt, axis=-1).flatten()
    return mags / np.max(mags)

def getScatterMarkers(input_color_vals: typing.Optional[NDArray] = None,
                      colorscale = 'Viridis', opacity=0.8, size=1):
    marker_spec = dict(size=size, opacity=opacity)
    if input_color_vals is not None:
        marker_spec.update(
            {"color": input_color_vals.flatten(), "colorscale": colorscale}
        )
    return marker_spec

def getLines(name: str, pts: NDArray, min_label_ind: int = -1, color="black",
             size=1):
    labels = None
    mode = "lines+markers"
    if min_label_ind >= 0:
        if np.any(np.isnan(pts)):
            raise NotImplementedError(
                "Labels for disjoint lines not supported!"
            )
        labels = [
            f"x{i}" for i in range(min_label_ind, min_label_ind + len(pts))
        ]
        mode += "+text"
    
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode=mode, name=name,
        text=labels, textposition="top center", line=dict(color=color),
        marker=dict(size=size, symbol="x"),
    )

def getScatter(name: str, pts: NDArray,
               input_color_vals: typing.Optional[NDArray] = None,
               colorscale = 'Viridis', opacity=0.8, size=1):
    
    marker_spec = getScatterMarkers(input_color_vals, colorscale, opacity, size)
    return go.Scatter3d(
        name=name, x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode='markers',
        marker=marker_spec
    )

def update_trace_pts(fig, vec3s: NDArray, name: str, **kwargs):
    fig.update_traces(
        x=vec3s[:, 0], y=vec3s[:, 1], z=vec3s[:, 2], selector={"name": name},
        **kwargs
    )

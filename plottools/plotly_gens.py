import typing

import numpy as np
from numpy.typing import NDArray

import plotly.graph_objects as go


def getColourMags(input_pts: NDArray, ref_pt: NDArray):
    mags = np.linalg.norm(input_pts - ref_pt, axis=-1).flatten()
    return mags / np.max(mags)

def getScatterMarkers(input_color_vals: typing.Optional[NDArray] = None,
                      colorscale = 'Viridis', opacity=0.8, size=1, **kwargs):
    if "marker" in kwargs:
        return kwargs["marker"]
    marker_spec = dict(size=size, opacity=opacity, **kwargs)
    if input_color_vals is not None:
        marker_spec.update(
            {"color": input_color_vals.flatten(), "colorscale": colorscale}
        )
    return marker_spec

def getLines(name: str, pts: NDArray, min_label_ind: int = -1, color="black",
             size=1, **kwargs):
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
        marker=dict(size=size, symbol="x"), **kwargs
    )

def getScatterAndMarkers(name: str, pts: NDArray,
               input_color_vals: typing.Optional[NDArray] = None,
               colorscale = 'Viridis', opacity=0.8, size=1, **kwargs):
    
    marker_spec = getScatterMarkers(
        input_color_vals, colorscale, opacity, size, **kwargs
    )
    sc = go.Scatter3d(
        name=name, x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode='markers',
        marker=marker_spec
    )
    return sc, marker_spec

def update_fig_trace_pts(fig, vec3s: NDArray, name: str, **kwargs):
    fig.update_traces(
        x=vec3s[:, 0], y=vec3s[:, 1], z=vec3s[:, 2], selector={"name": name},
        **kwargs
    )

def update_trace_pts(trace, pts: NDArray, **kwargs):
    trace.update(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], **kwargs
    )

# Add this class above your main callbacks and below your imports

class TraceManager:
    def __init__(self, fig):
        """
        Manages Plotly traces in a go.FigureWidget.
        Keeps track of traces by name and updates or creates as needed.
        """
        self.fig = fig
        # maps trace name to trace index in fig.data
        self.line_traces = {}  
        self.scatter_traces = {}

    def set_line_trace(self, name: str, pts: NDArray, min_label_ind: int = -1,
                       color = "black", size = 1, **kwargs):
        """
        Add a new trace or update an existing one by name.
        If the figure's data was cleared, resets the trace mapping.
        """

        if name in self.line_traces:
            idx = self.line_traces[name]
            # Update the trace in place for performance
            update_trace_pts(self.fig.data[idx], pts, **kwargs)
        else:
            self.fig.add_trace(getLines(name, pts, min_label_ind, color, size))
            self.line_traces[name] = len(self.fig.data) - 1

    def set_scatter_trace(self, name: str, pts: NDArray,
                          input_color_vals: typing.Optional[NDArray] = None,
                          colorscale = 'Viridis', opacity=0.8, size=1,
                          **kwargs):
        """
        Add a new trace or update an existing one by name.
        If the figure's data was cleared, resets the trace mapping.
        """
        
        markers = None
        if name in self.scatter_traces:
            idx = self.scatter_traces[name]
            # Update the trace in place for performance
            markers = getScatterMarkers(
                input_color_vals, colorscale, opacity, size, **kwargs
            )
            update_trace_pts(self.fig.data[idx], pts, marker=markers)
        else:
            tr, markers = getScatterAndMarkers(
                name, pts, input_color_vals, colorscale, opacity, size, **kwargs
            )
            self.fig.add_trace(tr)
            self.scatter_traces[name] = len(self.fig.data) - 1
        return markers
    
    def clear(self):
        """Clear all traces from the figure and reset mapping."""
        self.fig.data = []
        self.line_traces = {}
        self.scatter_traces = {}

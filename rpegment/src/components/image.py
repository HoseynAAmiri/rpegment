import cv2

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from dash import dcc

from skimage.measure import label as _label


def render(id: str) -> dcc.Graph:
    return dcc.Graph(
        id=id,
        figure=make(),
        style={
            'display': 'flex',
            'height': '100%',
        }
    )


def plot(img: go.Figure, image_array: npt.NDArray[np.uint8]) -> go.Figure:
    img.add_trace(go.Image(z=image_array))
    return img


def make() -> go.Figure:
    img = go.Figure()
    img.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return img


def read(image_path: str) -> npt.NDArray[np.uint8]:
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    return image.astype(np.uint8)

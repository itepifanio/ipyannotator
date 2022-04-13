#!/usr/bin/env python
# coding: utf-8

# In[1]:


# hide
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# default_exp services.bbox_trajectory


# In[3]:


# hide
from nbdev import *
from typing import Any, Dict


# In[4]:


#exporti
from ipycanvas import Canvas
from typing import List
from ipyannotator.ipytyping.annotations import AnnotationStore
from ipyannotator.mltypes import BboxCoordinate


# # Bounding Box Trajectory
# 
# The current notebook develop the data type and algorithms to store and process trajectories.

# In[5]:


#exporti
class TrajectoryStore(AnnotationStore):
    def __getitem__(self, key: str):
        assert isinstance(key, str)
        return self._annotations[key]

    def __delitem__(self, key: str):
        assert isinstance(key, str)
        if key in self:
            del self._annotations[key]

    def __setitem__(self, key: str, value: List[BboxCoordinate]):
        assert isinstance(key, str)
        assert isinstance(value, list)
        self._annotations[key] = value


# In[6]:


#exporti
class BBoxTrajectory:
    @staticmethod
    def draw_trajectory(canvas: Canvas, coords: List[BboxCoordinate], scale: float = 1.0):
        # iterate the coords two by two
        i, k = None, None
        c = iter(coords)
        lines = []
        while True:
            if i is None:
                i = next(c, None)
            k = next(c, None)
            if k and i:
                lines.append([(i.x * scale, (i.y + i.height) * scale),
                              (k.x * scale, (k.y + k.height) * scale)])
            else:
                break
            i = k

        canvas.stroke_styled_line_segments(lines, color=[50, 205, 50])


# In[7]:


# hide
from ipyannotator.bbox_canvas import draw_bounding_box, draw_bg


# In[8]:


#hide

# it can draw point in the middle of the circle

canvas = Canvas(width=100, height=100)
bbox_trajectory = BBoxTrajectory()
bbox_coords = [
    BboxCoordinate(*[0, 0, 50, 50]),
    BboxCoordinate(*[10, 20, 50, 50])
]

draw_bg(canvas)
draw_bounding_box(canvas, bbox_coords[0])
draw_bounding_box(canvas, bbox_coords[1])
bbox_trajectory.draw_trajectory(canvas=canvas, coords=bbox_coords)
canvas


# In[9]:


#hide

from attr import asdict
from ipyannotator.mltypes import BboxVideoCoordinate
from itertools import groupby
from collections import defaultdict

canvas = Canvas(width=500, height=500)
draw_bg(canvas)

storage = {
    'path1': {
        "bboxes": [
            BboxVideoCoordinate(10, 10, 103, 241, 'pedestrian1'),
            BboxVideoCoordinate(100, 350, 100, 100, 'pedestrian2'),
            BboxVideoCoordinate(300, 100, 155, 156, 'pedestrian3')
        ],
        'labels': [[], [], []]
    },
    'path2': {
        'bboxes': [
            BboxVideoCoordinate(30, 30, 102, 241, 'pedestrian1')
        ],
        'labels': [[]]
    }
}

trajectory: Dict[Any, Any] = defaultdict(list)


def key_fun(k):
    return k.id


for k, v in storage.items():
    # error: No overload variant of "sorted" matches argument types
    # "object", "Callable[[Any], Any]"
    for kk, vv in groupby(sorted(v['bboxes'], key=key_fun), key_fun):  # type: ignore
        value = list(vv)
        path = []
        for i in value:
            bbox_coordinate = asdict(i)
            bbox_coordinate.pop('id')
            path.append(BboxCoordinate(**bbox_coordinate))

        trajectory[kk] += path


# In[10]:


#hide

for k, v in trajectory.items():
    for bbox in v:
        draw_bounding_box(canvas, bbox)  # type: ignore
    if len(v) > 1:
        bbox_trajectory.draw_trajectory(canvas, v)  # type: ignore

canvas


# In[11]:


#hide
from nbdev.export import notebook2script
notebook2script()


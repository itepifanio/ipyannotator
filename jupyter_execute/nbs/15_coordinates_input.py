#!/usr/bin/env python
# coding: utf-8

# In[1]:


# default_exp custom_input.coordinates


# In[2]:


# hide
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


# hide
from nbdev import *


# In[4]:


# hide
import pytest
import ipytest
ipytest.autoconfig(raise_on_error=True)


# In[5]:


#exporti
import warnings
from attr import asdict
from ipyannotator.mltypes import BboxCoordinate
from ipywidgets import HBox, BoundedIntText, Layout
from typing import Callable, Optional


# # Coordinates Input

# In[6]:


#exporti

class CoordinateInput(HBox):
    def __init__(
        self,
        uuid: int = None,
        bbox_coord: BboxCoordinate = None,
        input_max: BboxCoordinate = None,
        coord_changed: Optional[Callable] = None,
        disabled: bool = False
    ):
        super().__init__()
        self.disabled = disabled
        self.uuid = uuid
        self._input_max = input_max
        self.coord_changed = coord_changed
        self.coord_labels = ['x', 'y', 'width', 'height']
        self.children = self.inputs
        self.layout = Layout(width="auto", overflow="initial")

        if bbox_coord:
            self.bbox_coord = bbox_coord  # type: ignore

    def __getitem__(self, key: str) -> int:
        return self.children[self.coord_labels.index(key)].value

    def __setitem__(self, key: str, value: int):
        self.children[self.coord_labels.index(key)].value = value

    @property
    def inputs(self) -> list:
        widget_inputs = []
        for in_p in self.coord_labels:
            widget_input = BoundedIntText(
                min=0,
                max=None if self._input_max is None else getattr(self._input_max, in_p),
                layout=Layout(width="55px"),
                continuous_update=False,
                disabled=self.disabled
            )
            widget_inputs.append(widget_input)
            widget_input.observe(self._on_coord_change, names="value")

        return widget_inputs

    @property
    def bbox_coord(self) -> BboxCoordinate:
        values = [c.value for c in self.children]
        return BboxCoordinate(
            **dict(zip(self.coord_labels, values))
        )

    @bbox_coord.setter
    def bbox_coord(self, bbox_coord: BboxCoordinate):
        for i, v in enumerate(asdict(bbox_coord).values()):
            self.children[i].value = v

    @property
    def input_max(self) -> Optional[BboxCoordinate]:
        return self._input_max

    @input_max.setter
    def input_max(self, input_max: dict):
        for i, label in enumerate(self.coord_labels):
            self.children[i].max = input_max[label]

    def _on_coord_change(self, change: dict):
        if self.coord_changed:
            try:
                idx = list(self.children).index(change["owner"])
                self.coord_changed(self.uuid, self.coord_labels[idx], change["new"])
            except ValueError:
                warnings.warn("Invalid coordinate change")


# In[7]:


#hide
inp_coord = CoordinateInput(
    input_max=BboxCoordinate(*[2, 2, 100, 100]),
    bbox_coord=BboxCoordinate(*[1, 1, 3, 88])
)
inp_coord


# In[8]:


@pytest.fixture
def coordinate_input_fixture() -> CoordinateInput:
    return CoordinateInput(input_max=BboxCoordinate(*[2, 2, 2, 100]))


# In[9]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_can_set_value_using_list(coordinate_input_fixture):\n    coordinate_input_fixture.bbox_coord = BboxCoordinate(*[1,1,1,1])\n    all_values_are_one = all([c.value == 1 for c in coordinate_input_fixture.children])\n    assert all_values_are_one == True')


# In[10]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_calls_callback_when_ipywidget_value_changes(coordinate_input_fixture):\n    label, value = None, None\n\n    def cb(c, l, v):\n        nonlocal label, value\n        label = l\n        value = v\n\n    coordinate_input_fixture.coord_changed = cb\n    coordinate_input_fixture.children[0].value = 2\n    coordinate_input_fixture.children[1].value = 2\n\n    assert label == 'y'\n    assert value == 2")


# In[11]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_disabled_all_input_if_coordinate_input_is_disabled():\n    inp_coord = CoordinateInput(\n        input_max=BboxCoordinate(*[2, 2, 100, 100]),\n        bbox_coord=BboxCoordinate(*[1, 1, 3, 88]),\n        disabled=True\n    )\n    \n    for inp in inp_coord.inputs:\n        assert inp.disabled is True')


# In[12]:


#hide
from nbdev.export import notebook2script
notebook2script()


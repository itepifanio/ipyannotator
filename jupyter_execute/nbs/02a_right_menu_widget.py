#!/usr/bin/env python
# coding: utf-8

# In[1]:


# hide
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# default_exp right_menu_widget


# In[3]:


# hide
from nbdev import *


# In[4]:


#exporti
from ipywidgets import HBox, Dropdown, Layout, VBox, Checkbox
from functools import partial
from typing import List, Optional, Callable
from ipyannotator.custom_input.coordinates import CoordinateInput
from ipyannotator.mltypes import BboxCoordinate, BboxVideoCoordinate
from ipyannotator.custom_input.buttons import ActionButton


# In[5]:


# hide
import ipytest
import pytest
ipytest.autoconfig(raise_on_error=True)


# # Right menu Widget

# In[6]:


#exporti

class BBoxItem(VBox):
    """BBox row with select button, coordinates inputs and delete button"""

    def __init__(
        self,
        bbox_coord: BboxCoordinate,
        max_coord_input_values: Optional[BboxCoordinate],
        index: int,
        options: List[str] = None,
        readonly: bool = False
    ):
        super().__init__()

        self.readonly = readonly
        self.bbox_coord = bbox_coord
        self.index = index
        self._max_coord_input_values = max_coord_input_values
        self.layout = Layout(display='flex', overflow='hidden')
        self.dropdown_classes = self._dropdown_classes(options)
        self.btn_select = self._btn_select(index)
        self.input_coordinates = self._coordinate_inputs(bbox_coord)

        elements = [
            self.btn_select,
            self.dropdown_classes,
            self.input_coordinates,
        ]

        if not self.readonly:
            self.btn_delete = self._btn_delete(index)
            elements.append(self.btn_delete)

        self.children = [HBox(elements)]

    def _btn_delete(self, index: int) -> ActionButton:
        return ActionButton(
            layout=Layout(width='auto'),
            icon="trash",
            button_style="danger",
            value=index
        )

    def _dropdown_classes(self, options: Optional[List[str]], value: str = None) -> Dropdown:
        return Dropdown(
            layout=Layout(width='auto'),
            options=options,
            value=value,
            disabled=self.readonly
        )

    def _btn_select(self, index: int) -> ActionButton:
        return ActionButton(
            icon="lightbulb-o",
            layout=Layout(width='auto'),
            value=index
        )

    def _coordinate_inputs(self, bbox_coord: BboxCoordinate):
        return CoordinateInput(
            bbox_coord=bbox_coord,
            input_max=self._max_coord_input_values,
            disabled=self.readonly
        )


# In[7]:


#hide

bbox_item = BBoxItem(
    bbox_coord=BboxCoordinate(*[10, 10, 10, 10]),
    index=0,
    options=['Item 01', 'Item 02'],
    max_coord_input_values=BboxCoordinate(*[10, 10, 10, 10])
)
bbox_item


# In[8]:


#exporti

class BBoxVideoItem(BBoxItem):
    def __init__(
        self,
        bbox_video_coord: BboxVideoCoordinate,
        index: int,
        label: List[str],
        options: List[str],
        selected: bool = False,
        btn_delete_enabled: bool = True,
        readonly: bool = False
    ):
        super(VBox, self).__init__()  # type: ignore
        self.readonly = readonly
        self.selected = selected
        self.bbox_video_coord = bbox_video_coord
        self.object_checkbox = self._object_checkbox()
        self.btn_select = self._btn_select(index)
        self.btn_delete = self._btn_delete(index)
        self.dropdown_classes = self._dropdown_classes(
            options=options,
            value=label[0] if label else None
        )

        self._options = [
            self.object_checkbox,
            self.btn_select,
            self.dropdown_classes,
        ]

        if btn_delete_enabled:
            self._options.append(self.btn_delete)

        self.children = [
            HBox(self._options)
        ]

    def _object_checkbox(self) -> Checkbox:
        return Checkbox(
            value=self.selected,
            indent=False,
            description=str(self.bbox_video_coord.id),
            layout=Layout(width='auto')
        )


# In[9]:


#hide

bbox_item = BBoxVideoItem(
    bbox_video_coord=BboxVideoCoordinate(10, 10, 10, 10, 'Object 01'),
    index=0,
    label=['Item 01'],
    options=['Item 01', 'Item 02']
)
bbox_item


# In[10]:


#exporti

class BBoxList(VBox):
    """Render the list of bbox items and set the interactions"""

    def __init__(
        self,
        classes: list,
        max_coord_input_values: Optional[BboxCoordinate],
        on_coords_changed: Optional[Callable],
        on_label_changed: Callable,
        on_btn_delete_clicked: Callable,
        on_btn_select_clicked: Optional[Callable],
        readonly: bool = False
    ):
        super().__init__()
        self._classes = classes
        self._max_coord_input_values = max_coord_input_values
        self._on_coords_changed = on_coords_changed
        self._on_btn_delete_clicked = on_btn_delete_clicked
        self._on_label_changed = on_label_changed
        self._on_btn_select_clicked = on_btn_select_clicked
        self.readonly = readonly

    @property
    def max_coord_input_values(self) -> Optional[BboxCoordinate]:
        return self._max_coord_input_values

    @max_coord_input_values.setter
    def max_coord_input_values(self, value: BboxCoordinate):
        for children in self.children:  # type: ignore
            children.input_max = value
        self._max_coord_input_values = value

    def render_btn_list(self, bbox_coords: List[BboxCoordinate], classes: List[List[str]]):
        elements: List[BBoxItem] = []
        num_children = len(self.children)  # type: ignore

        for index, coord in enumerate(bbox_coords[num_children:], num_children):
            bbox_item = BBoxItem(
                index=index,
                options=self._classes,
                bbox_coord=coord,
                max_coord_input_values=self._max_coord_input_values,
                readonly=self.readonly
            )

            if not self.readonly:
                bbox_item.btn_delete.on_click(self.del_element)
            bbox_item.input_coordinates.uuid = index
            bbox_item.input_coordinates.coord_changed = self._on_coords_changed
            bbox_item.btn_select.on_click(self._on_btn_select_clicked)
            if classes and classes[index]:
                bbox_item.dropdown_classes.value = classes[index][0] or None
            bbox_item.dropdown_classes.observe(
                partial(self._on_label_changed, index=index),
                names="value",
            )

            elements.append(bbox_item)

        self.children = [*list(self.children), *elements]  # type: ignore

    def __getitem__(self, index: int):
        return self.children[index]

    def clear(self):
        self.children = []

    def _update_bbox_list_index(self, elements: list, index: int):
        for index, element in enumerate(elements[index:], index):
            # updates select btn
            element.btn_select.value = index
            # update label dropdown
            dropdown = element.dropdown_classes
            dropdown.unobserve_all()
            dropdown.observe(
                partial(self._on_label_changed, index=index),
                names="value"
            )
            # update inputs
            element.children[0].children[2].uuid = index
            # updates delete btn
            element.btn_delete.value = index

    def del_element(self, btn: ActionButton):
        index = btn.value
        elements = list(self.children)
        del elements[index]
        self._update_bbox_list_index(elements, index)
        self.children = elements
        self._on_btn_delete_clicked(index)  # type: ignore


# In[11]:


#hide
def f(x):
    return x


bbox_list = BBoxList(['A', 'B'], BboxCoordinate(*[5, 5, 5, 10]), f, f, f, f)

classes: List[List[str]] = [[], [], []]
bbox_dict = [
    {'x': 10, 'y': 10, 'width': 20, 'height': 30},
    {'x': 20, 'y': 30, 'width': 10, 'height': 10},
    {'x': 30, 'y': 30, 'width': 10, 'height': 10}
]

bbox = [BboxCoordinate(**b) for b in bbox_dict]

bbox_list.render_btn_list(bbox, classes)
bbox_list


# In[12]:


#exporti

class BBoxVideoList(BBoxList):
    def __init__(
        self,
        classes: list,
        on_label_changed: Callable,
        on_btn_delete_clicked: Callable,
        on_btn_select_clicked: Optional[Callable],
        on_checkbox_object_clicked: Callable,
        btn_delete_enabled: bool = True
    ):
        super().__init__(
            classes=classes,
            max_coord_input_values=None,
            on_coords_changed=None,
            on_label_changed=on_label_changed,
            on_btn_delete_clicked=on_btn_delete_clicked,
            on_btn_select_clicked=on_btn_select_clicked
        )
        self.elements: List[BBoxVideoItem] = []
        self._btn_delete_enabled = btn_delete_enabled
        self._on_checkbox_object_clicked = on_checkbox_object_clicked

    # error: Signature of "render_btn_list" incompatible with supertype "BBoxList"
    def render_btn_list(  # type: ignore
        self,
        bbox_video_coords: List[BboxVideoCoordinate],
        classes: list,
        labels: List[List[str]],
        selected: List[int] = []
    ):
        if not bbox_video_coords:
            self.elements.clear()

        for index, bbox_video_coord in enumerate(bbox_video_coords):
            try:
                if self.elements[index]:
                    if self.elements[index].bbox_video_coord.id == bbox_video_coord.id:
                        self.elements[index].bbox_video_coord = bbox_video_coord
                    else:
                        del self.elements[index]
                        for i, _ in enumerate(bbox_video_coords[index:], index):
                            self.elements[i].index = i
            except Exception:
                bbox_item = BBoxVideoItem(
                    index=index,
                    options=self._classes,
                    bbox_video_coord=bbox_video_coord,
                    label=labels[index],
                    selected=index in selected,
                    btn_delete_enabled=self._btn_delete_enabled
                )

                bbox_item.btn_delete.on_click(self.del_element)
                bbox_item.btn_select.on_click(self._on_btn_select_clicked)

                if classes and classes[index]:
                    bbox_item.dropdown_classes.value = classes[index][0] or None

                bbox_item.dropdown_classes.observe(
                    partial(self._on_label_changed, index=index),
                    names="value",
                )

                bbox_item.object_checkbox.observe(
                    partial(
                        self._on_checkbox_object_clicked,
                        index=index,
                        bbox_video_coord=bbox_video_coord
                    ),
                    names="value",
                )

                self.elements.append(bbox_item)

        self.children = self.elements

    def clear(self):
        self.elements = []
        self.children = []

    def _update_bbox_list_index(self, elements: list, index: int):
        for index, element in enumerate(elements[index:], index):
            #updates checkbox
            checkbox = element.object_checkbox
            checkbox.unobserve_all()
            checkbox.observe(
                partial(self._on_checkbox_object_clicked, index=index),
                names="value",
            )
            # updates select btn
            element.btn_select.value = index
            # update label dropdown
            dropdown = element.dropdown_classes
            dropdown.unobserve_all()
            dropdown.observe(
                partial(self._on_label_changed, index=index),
                names="value"
            )
            # updates delete btn
            element.btn_delete.value = index

    def del_element(self, btn: ActionButton):
        index = btn.value
        elements = self.elements
        del elements[index]
        self._update_bbox_list_index(elements, index)
        self.children = elements
        self.elements = elements
        self._on_btn_delete_clicked(index)


# In[13]:


@pytest.fixture
def bbox_video_list_fixture():
    return BBoxVideoList(['A', 'B'], f, f, f, f)


# In[14]:


# hide
def list_to_bbox_item(bboxes: list) -> List[BBoxVideoItem]:
    result = []
    for i, bbox in enumerate(bboxes):
        video_item = BBoxVideoItem(
            index=i,
            options=['test' for j in bboxes],
            bbox_video_coord=bbox,
            label=['test' for j in bboxes]
        )

        result.append(video_item)

    return result


# In[15]:


get_ipython().run_cell_magic('ipytest', '', "\nfrom attr import asdict\n\ndef test_it_update_elements_on_rendering(bbox_video_list_fixture):\n    bbox = [\n        BboxVideoCoordinate(*[10, 10, 20, 30, 'object 1']),\n        BboxVideoCoordinate(*[15, 15, 25, 35, 'object 2']),\n    ]\n\n    labels = [['A'], ['B']]\n\n    bbox_video_list_fixture.elements = list_to_bbox_item(bbox)\n\n    bbox[1] = BboxVideoCoordinate(*[16, 16, 26, 26, 'object 2'])\n\n    bbox_video_list_fixture.render_btn_list(bbox_video_coords=bbox, classes=classes, labels=labels)\n\n    assert bbox_video_list_fixture.elements[1].bbox_video_coord == bbox[1]")


# In[16]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_delete_elements_on_rendering(bbox_video_list_fixture):\n    bbox = [\n        BboxVideoCoordinate(*[10, 10, 20, 30, 'object 1']),\n        BboxVideoCoordinate(*[15, 15, 25, 35, 'object 2']),\n    ]\n\n    labels = [['A'], ['B']]\n\n    bbox_video_list_fixture.elements = list_to_bbox_item(bbox)\n\n    bbox = [BboxVideoCoordinate(*[15, 15, 25, 35, 'object 2'])]\n\n    bbox_video_list_fixture.render_btn_list(\n        bbox_video_coords=bbox, \n        classes=classes, \n        labels=labels\n    )\n\n    assert len(bbox_video_list_fixture.elements) == 1\n    assert bbox_video_list_fixture.elements[0].bbox_video_coord.x == bbox[0].x")


# In[17]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_update_indexes_after_delete(bbox_video_list_fixture):\n    bbox = [\n        BboxVideoCoordinate(*[10, 10, 20, 30, 'object 1']),\n        BboxVideoCoordinate(*[15, 15, 25, 35, 'object 2']),\n        BboxVideoCoordinate(*[20, 20, 30, 40, 'object 3']),\n    ]\n\n    labels = [['A'], ['B']]\n\n    bbox_video_list_fixture.elements = list_to_bbox_item(bbox)\n\n    bbox = [\n        BboxVideoCoordinate(*[10, 10, 20, 30, 'object 1']),\n        BboxVideoCoordinate(*[20, 20, 30, 40, 'object 3']),\n    ]\n\n    bbox_video_list_fixture.render_btn_list(\n        bbox_video_coords=bbox, \n        classes=classes, \n        labels=labels\n    )\n\n    assert len(bbox_video_list_fixture.elements) == 2\n    assert bbox_video_list_fixture.elements[1].index == 1")


# In[18]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_can_render_btn_list_from_scratch(bbox_video_list_fixture):\n    assert bbox_video_list_fixture.elements == []\n    \n    classes = [[], [], []]\n    \n    bbox = [\n        BboxVideoCoordinate(**{'x': 10, 'y': 10, 'width': 20, 'height': 30, 'id': 'Object1'}),\n        BboxVideoCoordinate(**{'x': 20, 'y': 30, 'width': 10, 'height': 10, 'id': 'Object2'}),\n        BboxVideoCoordinate(**{'x': 30, 'y': 30, 'width': 10, 'height': 10, 'id': 'Object3'})\n    ]\n    \n    labels = [['A'], ['B'], ['A']]\n\n    bbox_video_list_fixture.render_btn_list(\n        bbox_video_coords=bbox, \n        classes=classes, \n        labels=labels\n    )\n    \n    assert bbox_video_list_fixture.elements != []\n    assert len(bbox_video_list_fixture.elements) == 3")


# In[19]:


@pytest.fixture
def readonly_fixture() -> BBoxList:
    bbox_list = BBoxList(['A', 'B'], BboxCoordinate(*[5, 5, 5, 10]), f, f, f, f, readonly=True)

    classes: List[List[str]] = [[], [], []]
    bbox_dict = [
        {'x': 10, 'y': 10, 'width': 20, 'height': 30},
        {'x': 20, 'y': 30, 'width': 10, 'height': 10},
        {'x': 30, 'y': 30, 'width': 10, 'height': 10}
    ]

    bbox = [BboxCoordinate(**b) for b in bbox_dict]

    bbox_list.render_btn_list(bbox, classes)

    return bbox_list


# In[20]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_doesnt_render_btn_delete_if_readonly(readonly_fixture):\n    assert hasattr(readonly_fixture[0], 'btn_delete') is False")


# In[21]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_cant_change_input_if_readonly(readonly_fixture):\n    assert readonly_fixture[0].dropdown_classes.disabled is True')


# In[22]:


#hide
from nbdev.export import notebook2script
notebook2script()


# In[ ]:





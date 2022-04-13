#!/usr/bin/env python
# coding: utf-8

# In[1]:


# hide
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# default_exp bbox_annotator


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
from pubsub import pub
from attr import asdict
from pathlib import Path
from copy import deepcopy
from typing import Optional, List, Callable

from IPython.display import display
from ipywidgets import AppLayout, Button, HBox, VBox, Layout

from ipyannotator.mltypes import BboxCoordinate
from ipyannotator.base import BaseState, AppWidgetState, Annotator
from ipyannotator.mltypes import InputImage, OutputImageBbox
from ipyannotator.bbox_canvas import BBoxCanvas, BBoxCanvasState
from ipyannotator.navi_widget import Navi
from ipyannotator.right_menu_widget import BBoxList, BBoxVideoItem
from ipyannotator.storage import JsonCaptureStorage
from ipyannotator.custom_input.buttons import ActionButton


# # Bounding Box Annotator
# 
# Bounding Box Annotator allows users to freely draw on top of images.

# ## State

# In[6]:


#exporti

class BBoxState(BaseState):
    coords: Optional[List[BboxCoordinate]]
    image: Optional[Path]
    classes: List[str]
    labels: List[List[str]] = []
    drawing_enabled: bool = True


# ## View

# In[7]:


#exporti

class BBoxCoordinates(VBox):
    """Connects the BBoxList and the states"""

    def __init__(
        self,
        app_state: AppWidgetState,
        bbox_canvas_state: BBoxCanvasState,
        bbox_state: BBoxState,
        on_btn_select_clicked: Callable = None
    ):
        super().__init__()

        self._app_state = app_state
        self._bbox_state = bbox_state
        self._bbox_canvas_state = bbox_canvas_state
        self.on_btn_select_clicked = on_btn_select_clicked

        self._init_bbox_list(self._bbox_state.drawing_enabled)

        if self._bbox_canvas_state.bbox_coords:
            self._bbox_list.render_btn_list(
                self._bbox_canvas_state.bbox_coords,
                self._bbox_state.labels
            )

        app_state.subscribe(self._refresh_children, 'index')
        bbox_state.subscribe(self._init_bbox_list, 'drawing_enabled')
        bbox_canvas_state.subscribe(self._sync_labels, 'bbox_coords')
        self._bbox_canvas_state.subscribe(self._update_max_coord_input, 'image_scale')
        self._update_max_coord_input(self._bbox_canvas_state.image_scale)
        self.children = self._bbox_list.children
        self.layout = Layout(
            max_height=f'{self._app_state.size[1]}px',
            display='block'
        )

    def _init_bbox_list(self, drawing_enabled: bool):
        self._bbox_list = BBoxList(
            max_coord_input_values=None,
            on_coords_changed=self.on_coords_change,
            on_label_changed=self.on_label_change,
            on_btn_delete_clicked=self.on_btn_delete_clicked,
            on_btn_select_clicked=self.on_btn_select_clicked,
            classes=self._bbox_state.classes,
            readonly=not drawing_enabled
        )

        self._refresh_children(0)

    def __getitem__(self, index: int) -> BBoxVideoItem:
        return self.children[index]

    def _refresh_children(self, index: int):
        self._bbox_list.clear()
        self._render(
            self._bbox_canvas_state.bbox_coords,
            self._bbox_state.labels
        )

    def _sync_labels(self, bbox_coords: List[BboxCoordinate]):
        """Every time a new coord is added to the annotator
        it's added an empty label to the state"""
        num_classes = len(self._bbox_state.labels)

        for i in bbox_coords[num_classes:]:
            self._bbox_state.labels.append([])

        self._render(bbox_coords, self._bbox_state.labels)

    def on_coords_change(self, index: int, key: str, value: int):
        setattr(self._bbox_canvas_state.bbox_coords[index], key, value)

        pub.sendMessage(
            f'{self._bbox_canvas_state.root_topic}.coord_changed',
            bbox_coords=self._bbox_canvas_state.bbox_coords
        )

    def _render(self, bbox_coords: list, labels: list):
        self._bbox_list.render_btn_list(bbox_coords, labels)
        self.children = self._bbox_list.children

    def on_label_change(self, change: dict, index: int):
        self._bbox_state.labels[index] = [change['new']]

    def remove_label(self, index: int):
        tmp_labels = deepcopy(self._bbox_state.labels)
        del tmp_labels[index]
        self._bbox_state.set_quietly('labels', tmp_labels)

    def on_btn_delete_clicked(self, index: int):
        bbox_coords = self._bbox_canvas_state.bbox_coords.copy()
        del bbox_coords[index]
        self.remove_label(index)
        self._bbox_canvas_state.bbox_coords = bbox_coords

    def _update_max_coord_input(self, image_scale: float):
        """CoordinateInput maximum values that a user
        can change. 'x' and 'y' can be improved to avoid
        bbox outside of the canvas area."""
        im_width = self._bbox_canvas_state.image_width
        im_height = self._bbox_canvas_state.image_height
        if im_height is not None and im_width is not None:
            size = [
                im_width // image_scale,
                im_height // image_scale
            ]
            coords = [int(size[i & 1]) for i in range(4)]
            self._bbox_list.max_coord_input_values = BboxCoordinate(*coords)


# In[8]:


#hide

app_state = AppWidgetState()
bbox_state = BBoxState(classes=['A', 'B'])
bbox_canvas_state = BBoxCanvasState(image_width=100, image_height=100)

bbox_coordinates = BBoxCoordinates(app_state, bbox_canvas_state, bbox_state)

bbox_coordinates


# In[9]:


#hide

# on bbox_canvas_state annotation change it reflects on the element list
assert len(bbox_coordinates.children) == 0  # type: ignore
bbox_canvas_state.bbox_coords = [BboxCoordinate(**{'x': 10, 'y': 10, 'width': 20, 'height': 30})]
assert len(bbox_coordinates.children) == 1  # type: ignore

# on element click it removes from state
bbox_coordinates.children[0].children[0].children[-1].click()  # type: ignore
assert len(bbox_coordinates.children) == 0  # type: ignore


# In[10]:


#hide
# it sync coords with classes
bbox_canvas_state.bbox_coords = [BboxCoordinate(**{'x': 10, 'y': 10, 'width': 20, 'height': 30})]
assert len(bbox_state.labels) == 1


# In[11]:


#exporti
class BBoxAnnotatorGUI(AppLayout):
    def __init__(
        self,
        app_state: AppWidgetState,
        bbox_state: BBoxState,
        fit_canvas: bool,
        on_save_btn_clicked: Callable = None,
        has_border: bool = False
    ):
        self._app_state = app_state
        self._bbox_state = bbox_state
        self._on_save_btn_clicked = on_save_btn_clicked
        self._label_history: List[List[str]] = []
        self.fit_canvas = fit_canvas
        self.has_border = has_border

        self._navi = Navi()

        self._save_btn = Button(description="Save",
                                layout=Layout(width='auto'))

        self._undo_btn = Button(description="Undo",
                                icon="undo",
                                layout=Layout(width='auto'))

        self._redo_btn = Button(description="Redo",
                                icon="rotate-right",
                                layout=Layout(width='auto'))

        self._controls_box = HBox(
            [self._navi, self._save_btn, self._undo_btn, self._redo_btn],
            layout=Layout(
                display='flex',
                flex_flow='row wrap',
                align_items='center'
            )
        )

        self._init_canvas(self._bbox_state.drawing_enabled)

        self.right_menu = BBoxCoordinates(
            app_state=self._app_state,
            bbox_canvas_state=self._image_box.state,
            bbox_state=self._bbox_state,
            on_btn_select_clicked=self._highlight_bbox
        )

        self._annotator_box = HBox(
            [self._image_box, self.right_menu],
            layout=Layout(
                display='flex',
                flex_flow='row'
            )
        )

        # set the values already instantiated on state
        if self._app_state.max_im_number:
            self._set_max_im_number(self._app_state.max_im_number)

        if self._bbox_state.image:
            self._set_image_path(str(self._bbox_state.image))

        # set the GUI interactions as callables
        self._navi.on_navi_clicked = self._idx_changed
        self._save_btn.on_click(self._save_clicked)
        self._undo_btn.on_click(self._undo_clicked)
        self._redo_btn.on_click(self._redo_clicked)

        bbox_state.subscribe(self._set_image_path, 'image')
        bbox_state.subscribe(self._init_canvas, 'drawing_enabled')
        bbox_state.subscribe(self._set_coords, 'coords')
        app_state.subscribe(self._set_max_im_number, 'max_im_number')

        super().__init__(
            header=None,
            left_sidebar=None,
            center=self._annotator_box,
            right_sidebar=None,
            footer=self._controls_box,
            pane_widths=(2, 8, 0),
            pane_heights=(1, 4, 1))

    def _init_canvas(self, drawing_enabled: bool):
        self._image_box = BBoxCanvas(
            *self._app_state.size,
            drawing_enabled=drawing_enabled,
            fit_canvas=self.fit_canvas,
            has_border=self.has_border
        )

    def _highlight_bbox(self, btn: ActionButton):
        self._image_box.highlight = btn.value

    def _redo_clicked(self, event: dict):
        self._image_box.redo_bbox()
        if self._label_history:
            self._bbox_state.labels[-1] = self._label_history.pop()
        self.right_menu._refresh_children(-1)

    def _undo_clicked(self, event: dict):
        if len(self._bbox_state.labels) > 0:
            self._label_history = [self._bbox_state.labels[-1]]
        self._image_box.undo_bbox()
        self.right_menu.remove_label(-1)
        self.right_menu._refresh_children(-1)

    def _set_image_path(self, image: Optional[str]):
        self._image_box._state.image_path = image

    def _set_coords(self, coords: List[BboxCoordinate]):
        if coords:
            tmp_coords = deepcopy(self._image_box._state.bbox_coords)
            # error: Argument 1 to "append" of "list" has incompatible
            # type "List[BboxCoordinate]"; expected "BboxCoordinate"
            tmp_coords.append(coords)  # type: ignore
            self._image_box._state.bbox_coords = coords

    def _set_max_im_number(self, max_im_number: int):
        # sync the navi GUI with AppWidgetState
        self._navi.max_im_num = max_im_number

    def _idx_changed(self, index: int):
        # store the last bbox drawn before index update
        self._bbox_state.set_quietly('coords', self._image_box._state.bbox_coords)
        self._app_state.index = index

    def _save_clicked(self, *args):
        if self._on_save_btn_clicked:
            self._on_save_btn_clicked(self._image_box._state.bbox_coords)
        else:
            warnings.warn("Save button click didn't triggered any event.")

    def on_client_ready(self, callback):
        self._image_box.observe_client_ready(callback)


# In[12]:


#hide
app_state = AppWidgetState()
bbox_state = BBoxState(classes=['test'])
# TODO::check why this 'test' str it's been used on the actual annotator.
BBoxAnnotatorGUI(
    app_state=app_state,
    bbox_state=bbox_state,
    fit_canvas=False
)


# ## Controller

# In[13]:


#exporti
class BBoxAnnotatorController:
    def __init__(
        self,
        app_state: AppWidgetState,
        bbox_state: BBoxState,
        storage: JsonCaptureStorage,
        render_previous_coords: bool = True,
        **kwargs
    ):
        self._app_state = app_state
        self._bbox_state = bbox_state
        self._storage = storage
        self._last_index = 0

        app_state.subscribe(self._idx_changed, 'index')

        self._update_im(self._last_index)
        self._app_state.max_im_number = len(self._storage)
        if render_previous_coords:
            self._update_coords(self._last_index)

    def save_current_annotations(self, coords: List[BboxCoordinate]):
        self._bbox_state.set_quietly('coords', coords)
        self._save_annotations(self._app_state.index)

    def _update_im(self, index: int):
        self._bbox_state.image = self._storage.images[index]

    def _update_coords(self, index: int):  # from annotations
        image_path = str(self._storage.images[index])
        coords = self._storage.get(image_path) or {}
        self._bbox_state.labels = coords.get('labels', [])
        self._bbox_state.coords = [BboxCoordinate(**c) for c in coords.get('bbox', [])]

    def _save_annotations(self, index: int, *args, **kwargs):  # to disk
        image_path = str(self._storage.images[index])
        self._storage[image_path] = {
            # error: Item "None" of "Optional[List[BboxCoordinate]]" has
            # no attribute "__iter__"
            'bbox': [asdict(bbox) for bbox in self._bbox_state.coords],  # type: ignore
            'labels': self._bbox_state.labels
        }
        self._storage.save()

    def _idx_changed(self, index: int):
        """
        On index change save an old state and update
        current image path and bbox coordinates for
        visualisation
        """
        self._save_annotations(self._last_index)
        self._update_im(index)
        self._update_coords(index)
        self._last_index = index

    def handle_client_ready(self):
        self._idx_changed(self._last_index)


# We have annotation saved in dictionary lile: `{'path/to/imagename.jpg': {'x':0, 'y': 0, 'width': 100, 'heigth': 100}}`
# 
# Navi widget has `index` and prev/next buttons to iterate over `max_im_number` of images (todo: change name as we can iterate of any object).
# 
# BBoxAnnotator has coupled `index` (with Navi one), and onchange event to update the current image path and label.
# 
# On image_path change event BBoxCanvas rerenders new image and label

# In[14]:


#hide
# new index ->  save *old* annotations -> update image -> update coordinates from annotation
#                     |
#                     |-> _update_annotations -> get current bbox values -> save to self.annotations


# In[15]:


#export

class BBoxAnnotator(Annotator):
    """
    Represents bounding box annotator.

    Gives an ability to itarate through image dataset,
    draw 2D bounding box annotations for object detection and localization,
    export final annotations in json format

    """

    def __init__(
        self,
        project_path: Path,
        input_item: InputImage,
        output_item: OutputImageBbox,
        annotation_file_path: Path,
        has_border: bool = False,
        *args, **kwargs
    ):
        app_state = AppWidgetState(
            uuid=str(id(self)),
            **{
                'size': (input_item.width, input_item.height),
            }
        )

        super().__init__(app_state)

        self._input_item = input_item
        self._output_item = output_item

        self.bbox_state = BBoxState(
            uuid=str(id(self)),
            classes=output_item.classes,
            drawing_enabled=self._output_item.drawing_enabled
        )

        self.storage = JsonCaptureStorage(
            im_dir=project_path / input_item.dir,
            annotation_file_path=annotation_file_path
        )

        self.controller = BBoxAnnotatorController(
            app_state=self.app_state,
            bbox_state=self.bbox_state,
            storage=self.storage,
            **kwargs
        )

        self.view = BBoxAnnotatorGUI(
            app_state=self.app_state,
            bbox_state=self.bbox_state,
            fit_canvas=self._input_item.fit_canvas,
            on_save_btn_clicked=self.controller.save_current_annotations,
            has_border=has_border
        )

        self.view.on_client_ready(self.controller.handle_client_ready)

    def __repr__(self):
        display(self.view)
        return ""

    def to_dict(self, only_annotated=True):
        return self.storage.to_dict(only_annotated)


# In[16]:


#hide
in_p = InputImage(image_dir='pics', image_width=640, image_height=400, fit_canvas=True)
out_p = OutputImageBbox(classes=['Label 01', 'Label 02'])


# In[17]:


#hide

get_ipython().system(' rm -rf ../data/projects/bbox/results')


# In[18]:


#hide
from ipyannotator.storage import construct_annotation_path

project_path = Path('../data/projects/bbox')

anno_file_path = construct_annotation_path(project_path)


# In[19]:


#hide
bb = BBoxAnnotator(
    project_path=Path(project_path),
    input_item=in_p,
    output_item=out_p,
    annotation_file_path=anno_file_path
)


# In[20]:


#hide
bb


# In[21]:


bb.view._image_box.debug_output


# In[22]:


bb.view._image_box._controller.debug_output


# In[23]:


@pytest.fixture
def bbox_fixture():
    bb = BBoxAnnotator(
        project_path=Path(project_path),
        input_item=in_p,
        output_item=out_p,
        annotation_file_path=anno_file_path
    )

    bbox_sample = [
        {'x': 10, 'y': 10, 'width': 20, 'height': 30},
        {'x': 10, 'y': 20, 'width': 20, 'height': 30},
        {'x': 10, 'y': 30, 'width': 20, 'height': 30},
    ]

    bb.view._image_box._state.bbox_coords = [BboxCoordinate(**bbox) for bbox in bbox_sample]

    return bb


# In[24]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_highlight_on_cursor_btn_click(bbox_fixture):\n    bbox_fixture.view.right_menu[0].btn_select.click()\n    assert bbox_fixture.view._image_box._state.bbox_selected == 0')


# In[25]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_unhighlight_on_second_click(bbox_fixture):\n    bbox_fixture.view.right_menu[0].btn_select.click()\n    assert bbox_fixture.view._image_box._state.bbox_selected == 0\n    bbox_fixture.view.right_menu[0].btn_select.click()\n    assert bbox_fixture.view._image_box._state.bbox_selected is None')


# In[26]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_unhighlight_on_second_click(bbox_fixture):\n    bbox_fixture.view.right_menu[0].btn_select.click()\n    bbox_fixture.view.right_menu[0].btn_select.click()\n    assert bbox_fixture.view._image_box._state.bbox_selected is None')


# In[27]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_changes_coordinates_when_bbox_state_changes(bbox_fixture):\n    bbox_fixture.view.right_menu[0].input_coordinates['x'] = 50\n    assert bbox_fixture.view._image_box._state.bbox_coords[0] == BboxCoordinate(\n        **{'x': 50, 'y': 10, 'width': 20, 'height': 30}\n    )")


# In[28]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_deletes_on_trash_button_click(bbox_fixture):\n    bbox_fixture.view.right_menu[1].btn_delete.click()\n    assert len(bbox_fixture.view._image_box._state.bbox_coords) == 2\n    bbox_fixture.view.right_menu[1].btn_delete.click()\n    assert len(bbox_fixture.view._image_box._state.bbox_coords) == 1\n    bbox_coordinate = [BboxCoordinate(**{'x': 10, 'y': 10, 'width': 20, 'height': 30})]\n    assert bbox_fixture.view._image_box._state.bbox_coords == bbox_coordinate")


# In[29]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_select_dropdown_option(bbox_fixture):\n    bbox_fixture.view.right_menu[0].dropdown_classes.value = 'Label 01'\n    assert bbox_fixture.bbox_state.labels == [['Label 01'], [], []]")


# In[30]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_removes_labels_on_trash_button_click(bbox_fixture):\n    bbox_fixture.view.right_menu[0].dropdown_classes.value = 'Label 01'\n    bbox_fixture.view.right_menu[0].btn_delete.click()\n    bbox_fixture.view.right_menu[1].dropdown_classes.value = 'Label 02'\n    assert bbox_fixture.bbox_state.labels == [[], ['Label 02']]")


# In[31]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_render_the_canvas_coordinates_on_user_navigation(bbox_fixture):\n    bbox_fixture.view._navi._next_btn.click()\n    assert len(bbox_fixture.view.right_menu.children) == 0\n    assert len(bbox_fixture.view.right_menu._bbox_canvas_state.bbox_coords) == 0\n    assert len(bbox_fixture.view.right_menu._bbox_state.labels) == 0\n    bbox_fixture.view._navi._prev_btn.click()\n    assert len(bbox_fixture.view.right_menu.children) == 3\n    assert len(bbox_fixture.view.right_menu._bbox_canvas_state.bbox_coords) == 3\n    assert len(bbox_fixture.view.right_menu._bbox_state.labels) == 3\n    bbox_fixture.view._navi._next_btn.click()\n    assert len(bbox_fixture.view.right_menu.children) == 0\n    assert len(bbox_fixture.view.right_menu._bbox_canvas_state.bbox_coords) == 0\n    assert len(bbox_fixture.view.right_menu._bbox_state.labels) == 0')


# In[32]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_loads_storage_values_on_inputs(bbox_fixture):\n    value = 'Label 01'\n    bbox_fixture.view.right_menu[0].dropdown_classes.value = value\n    bbox_fixture.view._save_btn.click()\n    \n    test_bb = BBoxAnnotator(\n        project_path=Path(project_path), \n        input_item=in_p,\n        output_item=out_p,\n        annotation_file_path=anno_file_path\n    )\n    \n    test_bb.app_state.index = bb.app_state.index\n    \n    assert test_bb.view.right_menu[0].dropdown_classes.value == value")


# In[33]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_can_undo_coordinate_input(bbox_fixture):\n    bbox_fixture.view._undo_btn.click()\n    assert len(bbox_fixture.view.right_menu.children) == 2\n    assert len(bbox_fixture.bbox_state.labels) == 2')


# In[34]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_can_redo_coordinate_input(bbox_fixture):\n    value = 'Label 01'\n    bbox_fixture.view.right_menu[2].dropdown_classes.value = value\n    bbox_fixture.view._undo_btn.click()\n    bbox_fixture.view._redo_btn.click()\n    assert len(bbox_fixture.view.right_menu.children) == 3\n    assert len(bbox_fixture.bbox_state.labels) == 3\n    assert bbox_fixture.bbox_state.labels[2] == [value]")


# In[35]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_change_max_input_coord_value_on_image_scale_change(bbox_fixture):\n    size = [\n        bbox_fixture.view.right_menu._bbox_canvas_state.image_width,\n        bbox_fixture.view.right_menu._bbox_canvas_state.image_height\n    ]\n    new_scale = 0.1\n    bbox_fixture.view._image_box._state.image_scale = new_scale\n    scaled_width = size[0]//new_scale\n    scaled_height = size[1]//new_scale\n    new_max_coord = bbox_fixture.view.right_menu._bbox_list._max_coord_input_values\n\n    result = [int(size[i & 1]// new_scale) for i in range(4)]\n    assert new_max_coord == BboxCoordinate(*result)')


# In[ ]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_can_fit_canvas(bbox_fixture):\n    bbox_fixture.view._image_box._state.fit_canvas = True\n    bbox_fixture.view._navi._next_btn.click()\n    state = bbox_fixture.view._image_box._state\n    assert state.height == 400 \n    assert state.width == 640')


# In[ ]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_can_fit_canvas_on_init():\n    in_p.image_width = None\n    in_p.image_height = None\n    in_p.fit_canvas = True\n    \n    bbox_fixture = BBoxAnnotator(\n        project_path=Path(project_path),\n        input_item=in_p,\n        output_item=out_p,\n        annotation_file_path=anno_file_path\n    )\n    \n    state = bbox_fixture.view._image_box._state\n\n    assert bbox_fixture.view._image_box.state.fit_canvas == True\n    assert state.height == 400\n    assert state.width == 640')


# In[ ]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_can_disable_drawing(bbox_fixture):\n    bbox_fixture.bbox_state.drawing_enabled = False\n    assert bbox_fixture.view._image_box.drawing_enabled is False')


# In[ ]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_cant_delete_annotation_when_drawing_enable(bbox_fixture):\n    bbox_fixture.bbox_state.drawing_enabled = False\n    assert hasattr(bbox_fixture.view.right_menu[0], 'btn_delete') is False")


# In[ ]:


#hide
bb.storage.to_dict(False)


# In[ ]:


#hide
bb.to_dict()


# In[ ]:


#hide
from nbdev.export import notebook2script
notebook2script()


# In[ ]:





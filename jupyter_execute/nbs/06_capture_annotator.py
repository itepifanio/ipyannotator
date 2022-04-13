#!/usr/bin/env python
# coding: utf-8

# In[1]:


# default_exp capture_annotator


# In[2]:


# hide
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


# hide
from nbdev import *


# In[4]:


#exporti

import math
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Callable, List

from IPython.display import display
from ipywidgets import (AppLayout, HBox, Button, HTML, VBox,
                        Layout, Checkbox, Output)

from ipyannotator.custom_widgets.grid_menu import GridMenu, Grid
from ipyannotator.base import BaseState, AppWidgetState, Annotator
from ipyannotator.navi_widget import Navi
from ipyannotator.ipytyping.annotations import LabelStore, _label_store_to_image_button
from ipyannotator.storage import JsonCaptureStorage


# In[5]:


# hide
import ipytest
import pytest
ipytest.autoconfig(raise_on_error=True)


# # Capture annotator
# 
# The current notebook develop a annotator (`Capture Annotator`) that displays multiple items options, allowing users to select multiple of them and save their answers.

# ## State
# 
# The data shared across the annotator are:
# 
# - The `annotations` attribute represents all annotations that could be selected and the user's answers;
# - The `disp_number` attribute represents the number of options to be displayed;
# - The `question_value` attribute represents the label to be shown above the selectable options;
# - The `all_none` attribute represents that no option was selected;
# - The `n_rows` and `n_cols` displays the number of options to be shows per rows and cols respectively.

# In[6]:


#exporti
class CaptureState(BaseState):
    annotations: LabelStore = LabelStore()
    grid: Grid
    question_value: str = ''
    all_none: bool = False


# ## View

# The ` CaptureAnnotatorGUI ` joins the internal component (`GridMenu`) with the navi component and its interaction.

# In[7]:


#exporti

class CaptureAnnotatorGUI(AppLayout):
    """
    save_btn_clicked: callable
        activated when the user clicked on the save button
    grid_box_clicked: callable
        activated when the user clicked on the grid box
    on_navi_clicked: callable
        activated when the user navigates through the annotator
    """

    debug_output = Output(layout={'border': '1px solid black'})

    def __init__(
        self,
        app_state: AppWidgetState,
        capture_state: CaptureState,
        save_btn_clicked: Callable = None,
        grid_box_clicked: Callable = None,
        on_navi_clicked: Callable = None,
        select_none_changed: Callable = None
    ):
        self._app_state = app_state
        self._capture_state = capture_state
        self._save_btn_clicked = save_btn_clicked
        self._grid_box_clicked = grid_box_clicked
        self._select_none_changed = select_none_changed

        self._navi = Navi()

        self._save_btn = Button(description="Save",
                                layout=Layout(width='auto'))

        self._none_checkbox = Checkbox(description="Select none",
                                       indent=False,
                                       layout=Layout(width='100px'))

        self._controls_box = HBox(
            [
                self._navi,
                self._save_btn,
                self._none_checkbox,
            ],
            layout=Layout(
                display='flex',
                justify_content='center',
                flex_flow='wrap',
                align_items='center'
            )
        )

        self._grid_box = GridMenu(capture_state.grid)

        self._grid_label = HTML()
        self._labels_box = VBox(
            children=[
                self._grid_label,
                self._grid_box
            ],
            layout=Layout(
                display='flex',
                justify_content='center',
                flex_wrap='wrap',
                align_items='center'
            )
        )

        self.on_navi_clicked = on_navi_clicked
        self._navi.on_navi_clicked = self._on_navi_clicked
        self._save_btn.on_click(self._btn_clicked)
        self._grid_box.on_click(self.on_grid_clicked)
        self._none_checkbox.observe(self._none_checkbox_changed, 'value')

        if self._capture_state.question_value:
            self._set_label(self._capture_state.question_value)

        if self._app_state.max_im_number:
            self._set_navi_max_im_number(self._app_state.max_im_number)

        if self._capture_state.annotations:
            self._load_menu(self._capture_state.annotations)

        self._capture_state.subscribe(self._set_none_checkbox, 'all_none')
        self._capture_state.subscribe(self._set_label, 'question_value')
        self._app_state.subscribe(self._set_navi_max_im_number, 'max_im_number')
        self._capture_state.subscribe(self._load_menu, 'annotations')

        super().__init__(
            header=None,
            left_sidebar=None,
            center=self._labels_box,
            right_sidebar=None,
            footer=self._controls_box,
            pane_widths=(2, 8, 0),
            pane_heights=(1, 4, 1))

    def _on_navi_clicked(self, index: int):
        if self.on_navi_clicked:
            self.on_navi_clicked(index)

        self._grid_box.load(
            _label_store_to_image_button(self._capture_state.annotations)
        )

    @debug_output.capture(clear_output=True)
    def _load_menu(self, annotations: LabelStore):
        self._grid_box.load(
            _label_store_to_image_button(annotations)
        )

    def _set_none_checkbox(self, all_none: bool):
        self._none_checkbox.value = all_none

    def _set_navi_max_im_number(self, max_im_number: int):
        self._navi.max_im_num = max_im_number

    def _set_label(self, question_value: str):
        self._grid_label.value = question_value

    def _btn_clicked(self, *args):
        if self._save_btn_clicked:
            self._save_btn_clicked(self._app_state.index)
        else:
            warnings.warn("Save button click didn't triggered any event.")

    def _none_checkbox_changed(self, change: dict):
        self._capture_state.set_quietly('all_none', change['new'])
        if self._select_none_changed:
            self._select_none_changed(change)

    def on_grid_clicked(self, event, name=None):
        if self._grid_box_clicked:
            self._grid_box_clicked(event, name)
        else:
            warnings.warn("Grid box click didn't triggered any event.")


# In[8]:


#hide
from ipyannotator.custom_input.buttons import ImageButton


# In[9]:


get_ipython().run_cell_magic('ipytest', '', "def test_gui_loads_image_button_from_menu():\n    annotations = LabelStore()\n    annotations['../data/projects/capture1/pics/pink25x25.png'] = {'answer': True}\n    grid = Grid(width=200, height=200)\n    gui = CaptureAnnotatorGUI(\n        app_state=AppWidgetState(),\n        capture_state=CaptureState(annotations=annotations, grid=grid)\n    )\n    gui._load_menu(gui._capture_state.annotations)\n    assert isinstance(gui._grid_box.widgets[0], ImageButton)")


# In[10]:


# hide
# error: Argument 1 to "AppWidgetState" has incompatible type
# "**Dict[str, Tuple[int, int]]"; expected "Optional[str]"
app_state = AppWidgetState(**{'size': (50, 50)})  # type: ignore

grid = Grid(width=100, height=100, n_rows=5, n_cols=5)

# error: Argument 1 to "CaptureState" has incompatible type
# "**Dict[str, int]"; expected "Optional[str]"
capture_state = CaptureState(
    **{'grid': grid, 'annotations': LabelStore()})  # type: ignore

ca = CaptureAnnotatorGUI(
    capture_state=capture_state,
    app_state=app_state
)

ca._grid_label.value = 'Select smth'

ca


# In[11]:


# hide
data = {
    '../data/projects/capture1/pics/pink25x25.png': {'answer': True}
}
ca._capture_state.annotations.update(data)


# In[12]:


# hide
project_path = Path('../data/projects/capture1')


# In[13]:


# hide
from ipyannotator.mltypes import InputImage, OutputGridBox

imz = InputImage()
grid_bx = OutputGridBox()


# In[14]:


# hide

# it loads annotation labels when CaptureState.annotations changes
from pubsub import pub

pub.sendMessage('CaptureState.annotations', annotations={
    '../data/projects/capture1/pics/pink25x25.png': {'answer': False}
})
assert ca._grid_box.widgets is not None
assert list(filter(lambda l: l.active, ca._grid_box.widgets)) == []

# it throw warning if no btn_clicked callable is provided

with warnings.catch_warnings(record=True) as w:
    ca._save_btn.click()
    assert len(w) == 1
    assert "Save button click didn't triggered any event." in str(w[-1].message)

# it doesnt throw warning if btn_clicked callable is provided

with warnings.catch_warnings(record=True) as w:
    ca._save_btn_clicked = lambda index: index
    ca._save_btn.click()
    assert len(w) == 0


# ## Storage
# 
# The `CaptureAnnotationStorage` saves the user annotations on the disk

# In[15]:


#exporti

class CaptureAnnotationStorage:
    def __init__(
        self,
        input_item_path: Path,
        annotation_file_path: str
    ):
        self.input_item_path = input_item_path
        self.annotation_file_path = annotation_file_path

        self.annotations = JsonCaptureStorage(
            self.input_item_path,
            annotation_file_path=self.annotation_file_path
        )

    def __setitem__(self, key, value):
        self.annotations[key] = value

    def _save(self):
        self.annotations.save()

    def get_im_names(self, filter_files) -> List[str]:
        return self.annotations.get_im_names(filter_files)

    def get(self, path: str) -> Optional[Dict]:
        return self.annotations.get(path)

    def update_annotations(self, annotations: dict):
        for p, anno in annotations.items():
            self.annotations[str(p)] = anno
        self._save()

    def to_dict(self, only_annotated: bool = True) -> dict:
        return self.annotations.to_dict(only_annotated)


# ## Controller
# 
# The controller communicates with the state and the storage layer, updating the states and saving the data on disk using the storage module. 

# In[16]:


#exporti

class CaptureAnnotatorController:
    debug_output = Output(layout={'border': '1px solid black'})

    def __init__(
        self,
        app_state: AppWidgetState,
        capture_state: CaptureState,
        storage: CaptureAnnotationStorage,
        input_item=None,
        output_item=None,
        filter_files=None,
        question=None,
        *args,
        **kwargs
    ):
        self._app_state = app_state
        self._capture_state = capture_state
        self._storage = storage
        self.input_item = input_item
        self.output_item = output_item
        self._last_index = 0

        self.images = self._storage.get_im_names(filter_files)
        self.current_im_number = len(self.images)

        if question:
            self._capture_state.question_value = ('<center><p style="font-size:20px;"'
                                                  f'>{question}</p></center>')

        self.update_state()
        self._calc_screens_num()

    def update_state(self):
        state_images = self._get_state_names(self._app_state.index)
        tmp_annotations = deepcopy(self._capture_state.annotations)
        current_state = {}

        for im_path in state_images:
            current_state[str(im_path)] = self._storage.get(str(im_path)) or {}

        self._update_all_none_state(current_state)

        # error: Incompatible types in assignment (expression has type
        # "Dict[str, Dict[Any, Any]]", variable has type
        # "Dict[str, Optional[Dict[str, bool]]]")
        tmp_annotations.clear()
        tmp_annotations.update(current_state)
        self._capture_state.annotations = tmp_annotations  # type: ignore

    def _update_all_none_state(self, state_images: dict):
        self._capture_state.all_none = all(
            value == {'answer': False} for value in state_images.values()
        )

    def save_annotations(self, index: int):  # to disk
        state_images = dict(self._capture_state.annotations)

        self._storage.update_annotations(state_images)

    def _get_state_names(self, index: int) -> List[str]:
        start = index * self._capture_state.grid.disp_number
        end = start + self._capture_state.grid.disp_number
        im_names = self.images[start:end]
        return im_names

    def idx_changed(self, index: int):
        ''' On index change save old state
            and update current state for visualisation
        '''
        self._app_state.set_quietly('index', index)
        self.save_annotations(self._last_index)
        self.update_state()
        self._last_index = index

    def _calc_screens_num(self):
        self._app_state.max_im_number = math.ceil(
            self.current_im_number / self._capture_state.grid.disp_number
        )

    @debug_output.capture(clear_output=False)
    def handle_grid_click(self, event: dict, name=None):
        p = self._storage.input_item_path / name
        current_state = deepcopy(self._capture_state.annotations)
        if not p.is_dir():
            state_answer = self._capture_state.annotations[
                str(p)].get('answer', False)
            current_state[str(p)] = {'answer': not state_answer}

            for k, v in current_state.items():
                if v == {}:
                    current_state[k] = {'answer': False}

            if self._capture_state.all_none:
                self._capture_state.all_none = False
            else:
                self._update_all_none_state(dict(current_state))
        else:
            return

        self._capture_state.annotations = current_state

    def select_none(self, change: dict):
        if self._capture_state.all_none:
            tmp_annotations = deepcopy(self._capture_state.annotations)
            tmp_annotations.clear()
            tmp_annotations.update(
                {p: {
                    'answer': False} for p in self._capture_state.annotations}
            )
            self._capture_state.annotations = tmp_annotations


# In[17]:


# remove if the results folder exists this allows
# the next command to construct the annotation path
get_ipython().system(' rm -rf ../data/projects/capture1/results')


# In[18]:


from ipyannotator.storage import construct_annotation_path

anno_file_path = construct_annotation_path(project_path)

storage = CaptureAnnotationStorage(
    input_item_path=project_path / imz.dir,
    annotation_file_path=anno_file_path
)

app_state = AppWidgetState()
capture_state = CaptureState(grid=grid)

caController = CaptureAnnotatorController(
    app_state=app_state,
    capture_state=capture_state,
    storage=storage,
    input_item=imz,
    output_item=grid_bx,
    annotation_file_path=anno_file_path,
    question='hello'
)

caController._capture_state.disp_number = 9  # should be synced from gui


# We have 16 images in `capture1` project on disk, so first screen should load 9 images;
# 7 images (16-9) left for second screen.

# In[19]:


# hide
assert caController._app_state.max_im_number == 2


# In[20]:


caController.images


# This will output the path of all the 16 images
# 
# ```bash
# [Path('../data/projects/capture1/pics/pink25x25.png'),
#  Path('../data/projects/capture1/pics/pink50x125.png'),
#  Path('../data/projects/capture1/pics/pink50x50.png'),
#  Path('../data/projects/capture1/pics/pink50x50_0.png'),
#  Path('../data/projects/capture1/pics/pink50x50_1.png'),
#  Path('../data/projects/capture1/pics/pink50x50_3.png'),
#  Path('../data/projects/capture1/pics/teal125x50.png'),
#  Path('../data/projects/capture1/pics/teal50x50.png'),
#  Path('../data/projects/capture1/pics/teal50x50_0.png'),
#  Path('../data/projects/capture1/pics/teal50x50_1.png'),
#  Path('../data/projects/capture1/pics/teal50x50_2.png'),
#  Path('../data/projects/capture1/pics/teal50x50_3.png'),
#  Path('../data/projects/capture1/pics/teal50x50_4.png'),
#  Path('../data/projects/capture1/pics/teal50x50_5.png'),
#  Path('../data/projects/capture1/pics/teal50x50_6.png'),
#  Path('../data/projects/capture1/pics/teal75x75.png')]
# ```

# In[21]:


# hide
assert len(caController._get_state_names(0)) == 9
assert len(caController._get_state_names(1)) == 7
assert len(caController._get_state_names(5)) == 0


# List of image names for the 1st screen:

# In[22]:


caController._capture_state.annotations


# ```bash
# {'../data/projects/capture1/pics/pink25x25.png': {},
#  '../data/projects/capture1/pics/pink50x125.png': {},
#  '../data/projects/capture1/pics/pink50x50.png': {},
#  '../data/projects/capture1/pics/pink50x50_0.png': {},
#  '../data/projects/capture1/pics/pink50x50_1.png': {},
#  '../data/projects/capture1/pics/pink50x50_3.png': {},
#  '../data/projects/capture1/pics/teal125x50.png': {},
#  '../data/projects/capture1/pics/teal50x50.png': {},
#  '../data/projects/capture1/pics/teal50x50_0.png': {}}
# ```

# Suppose state change from gui:

# In[23]:


caController._capture_state.annotations[
    '../data/projects/capture1/pics/pink25x25.png'] = {'answer': False}


# ##### (Next-> button emulation) 
# Increment index to initiate annotation save and switch state for a new screen

# In[24]:


caController._app_state.index = 1
caController._capture_state.annotations


# ##### (<-Prev button emulation) 
# Decrement index to initiate annotation save and switch state for previous screen, loading existing annotation

# In[25]:


# error: "CaptureAnnotatorController" has no attribute "index"
caController.index = 0  # type: ignore
caController._capture_state.annotations


# In[26]:


caController.select_none({'new': True})
caController._capture_state


# In[27]:


#export

class CaptureAnnotator(Annotator):
    debug_output = Output(layout={'border': '1px solid black'})
    """
    Represents capture annotator.

    Gives an ability to itarate through image dataset,
    select images of same class,
    export final annotations in json format

    """
#     @debug_output.capture(clear_output=True)

    def __init__(
        self,
        project_path: Path,
        input_item,
        output_item,
        annotation_file_path,
        n_rows=3,
        n_cols=3,
        disp_number=9,
        question=None,
        filter_files=None
    ):

        assert input_item, "WARNING: Provide valid Input"
        assert output_item, "WARNING: Provide valid Output"

        self._project_path = project_path
        self._input_item = input_item
        self._output_item = output_item
        self._annotation_file_path = annotation_file_path
        self._n_rows = n_rows
        self._n_cols = n_cols
        self._question = question
        self._filter_files = filter_files

        app_state = AppWidgetState(
            uuid=str(id(self)),
            **{'size': (input_item.width, input_item.height)}
        )

        super().__init__(app_state)

        grid = Grid(
            width=input_item.width,
            height=input_item.height,
            n_rows=n_rows,
            n_cols=n_cols,
            display_label=False,
            disp_number=disp_number
        )

        self.capture_state = CaptureState(
            uuid=str(id(self)),
            annotations=LabelStore(),
            grid=grid
        )

        self.storage = CaptureAnnotationStorage(
            input_item_path=project_path / input_item.dir,
            annotation_file_path=annotation_file_path
        )

        self.controller = CaptureAnnotatorController(
            app_state=self.app_state,
            storage=self.storage,
            capture_state=self.capture_state,
            input_item=input_item,
            output_item=output_item,
            question=question,
            n_rows=n_rows,
            n_cols=n_cols,
            filter_files=filter_files
        )

        self.view = CaptureAnnotatorGUI(
            capture_state=self.capture_state,
            app_state=self.app_state,
            save_btn_clicked=self.controller.save_annotations,
            grid_box_clicked=self.controller.handle_grid_click,
            on_navi_clicked=self.controller.idx_changed,
            select_none_changed=self.controller.select_none
        )

    def __repr__(self):
        display(self.view)
        return ""

    def to_dict(self, only_annotated=True) -> dict:
        return self.storage.to_dict(only_annotated)


# In[28]:


# hide
proj_path = Path('../data/projects/capture1')

anno_file_path = construct_annotation_path(
    file_name='../data/projects/capture1/results/annotation-all.json')

in_p = InputImage(image_dir='pics', image_width=75, image_height=75)

out_p = OutputGridBox()

ca_annotator = CaptureAnnotator(
    proj_path,
    input_item=in_p,
    output_item=out_p,
    annotation_file_path=anno_file_path,
    question="Select pink squares"
)


# In[29]:


ca_annotator


# In[30]:


# it should not mark annotations as False
# if user navigates (or clicks on save button)
# without clicking on any cell or select all none checkbox

assert ca_annotator.capture_state.annotations['../data/projects/capture1/pics/pink25x25.png'] == {}
assert ca_annotator.storage.annotations['../data/projects/capture1/pics/pink25x25.png'] is None

ca_annotator.view._save_btn.click()
ca_annotator.view._navi._next_btn.click()

assert ca_annotator.storage.annotations['../data/projects/capture1/pics/pink25x25.png'] == {}

ca_annotator.view._navi._next_btn.click()

# it doesn't fill all annotations as False when loading
assert ca_annotator.capture_state.annotations['../data/projects/capture1/pics/pink25x25.png'] == {}
assert ca_annotator.storage.annotations['../data/projects/capture1/pics/pink25x25.png'] == {}

# it can select a grid item (it update the state, but not save at storage)

ca_annotator.app_state.index = 0
ca_annotator.controller.handle_grid_click(
    event={},
    name='pink25x25.png'
)

assert ca_annotator.capture_state.annotations['../data/projects/capture1/pics/pink25x25.png'] == {
    'answer': True}

assert ca_annotator.storage.annotations[
    '../data/projects/capture1/pics/pink25x25.png'] != {'answer': True}

# it select the remain of the grid item as False if user clicks on save button

ca_annotator.view._save_btn.click()

assert ca_annotator.storage.annotations[
    '../data/projects/capture1/pics/pink25x25.png'] == {'answer': True}
assert ca_annotator.storage.annotations[
    '../data/projects/capture1/pics/pink50x125.png'] == {'answer': False}
assert ca_annotator.capture_state.annotations[
    '../data/projects/capture1/pics/pink50x125.png'] == {'answer': False}

# it save annotations status when user navigates

ca_annotator.controller.handle_grid_click(
    event=None,
    name='pink50x125.png'
)

assert ca_annotator.capture_state.annotations['../data/projects/capture1/pics/pink50x125.png'] == {
    'answer': True}

assert ca_annotator.storage.annotations[
    '../data/projects/capture1/pics/pink50x125.png'] == {'answer': False}

ca_annotator.view._navi._next_btn.click()

assert ca_annotator.storage.annotations[
    '../data/projects/capture1/pics/pink50x125.png'] == {'answer': True}

# it can sync select none checkbox when navigating

ca_annotator.view._none_checkbox.value = True
ca_annotator.view._navi._next_btn.click()
assert ca_annotator.view._none_checkbox.value is False
ca_annotator.view._navi._next_btn.click()
assert ca_annotator.view._none_checkbox.value is True

# annotators doesn't share states with each other

assert ca_annotator.app_state.index == 1

other_annotator = CaptureAnnotator(
    proj_path,
    input_item=in_p,
    output_item=out_p,
    annotation_file_path=anno_file_path,
    n_cols=3,
    question="Select pink squares"
)

assert other_annotator.app_state.index == 0
assert ca_annotator.app_state.index == 1
assert ca_annotator.app_state.index != other_annotator.app_state.index


# In[31]:


ca_annotator.storage.annotations


# In[32]:


ca_annotator.to_dict()


# In[33]:


#hide
from nbdev.export import notebook2script
notebook2script()


# In[ ]:





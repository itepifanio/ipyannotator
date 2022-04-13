#!/usr/bin/env python
# coding: utf-8

# In[1]:


# default_exp im2im_annotator


# In[2]:


# hide
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


# hide
from nbdev import *


# In[4]:


#exporti
import io
import warnings
from pathlib import Path
from copy import deepcopy
from typing import Optional, Callable, Union, Iterable

from ipycanvas import Canvas
from ipywidgets import (AppLayout, VBox, HBox, Button, Layout, HTML, Output, Image)

from ipyannotator.base import BaseState, AppWidgetState, Annotator, AnnotatorStep
from ipyannotator.bbox_canvas import ImageRenderer
from ipyannotator.mltypes import OutputImageLabel, OutputLabel, InputImage
from ipyannotator.ipytyping.annotations import LabelStore, LabelStoreCaster
from ipyannotator.custom_widgets.grid_menu import GridMenu, Grid
from ipyannotator.navi_widget import Navi
from ipyannotator.storage import JsonLabelStorage
from IPython.display import display
from ipyannotator.doc_utils import is_building_docs
from PIL import Image as PILImage


# In[5]:


# hide
import ipytest
import pytest
ipytest.autoconfig(raise_on_error=True)


# # Image to image annotator
# 
# The image to image annotator (Im2ImAnnotator) allows the users to navigate through multiple images and select multiple options in any one of them. The current notebook develops this annotator.

# ## State
# 
# The data shared across the annotator are:
# 
# - The `annotations` attribute represents all annotations options that could be selected and the user's answers;
# - The `disp_number` attribute represents the number of options to be displayed;
# - The `question_value` attribute represents the label to be shown above the selectable options;
# - The `n_rows` and `n_cols` displays the number of options to be shows per rows and columns respectively;
# - The `image_path` attibute it's the image been currently annotated;
# - The `im_width` and `im_height` displays the image size;
# - The `label_width` and `label_height` displays the selectable options size;

# In[6]:


# exporti

class Im2ImState(BaseState):
    annotations: LabelStore = LabelStore()
    question_value: str = ''
    grid: Grid
    image_path: Optional[str]
    im_width: int = 300
    im_height: int = 300


# ## View
# 
# For the view an internal component called `ImCanvas` was developed and used to display the current image been annotated.

# In[7]:


# export
if is_building_docs():
    class ImCanvas(Image):
        def __init__(
            self,
            width: int = 150,
            height: int = 150,
            has_border: bool = False,
            fit_canvas: bool = False
        ):
            super().__init__(width=width, height=height)
            image = PILImage.new('RGB', (100, 100), (255, 255, 255))
            b = io.BytesIO()
            image.save(b, format='PNG')
            self.value = b.getvalue()

        def _draw_image(self, image_path: str):
            self.value = Image.from_file(image_path).value

        def _clear_image(self):
            pass

        def observe_client_ready(self, cb=None):
            pass
else:
    class ImCanvas(HBox):  # type: ignore
        def __init__(
            self,
            width: int = 150,
            height: int = 150,
            has_border: bool = False,
            fit_canvas: bool = False
        ):
            self.has_border = has_border
            self.fit_canvas = fit_canvas
            self._canvas = Canvas(width=width, height=height)
            super().__init__([self._canvas])

        def _draw_image(self, image_path: str):
            img_render_strategy = ImageRenderer(
                clear=True,
                has_border=self.has_border,
                fit_canvas=self.fit_canvas
            )

            self._image_scale = img_render_strategy.render(
                self._canvas,
                image_path
            )

        def _clear_image(self):
            self._canvas.clear()

        # needed to support voila
        # https://ipycanvas.readthedocs.io/en/latest/advanced.html#ipycanvas-in-voila
        def observe_client_ready(self, cb=None):
            self._canvas.on_client_ready(cb)


# In[8]:


# hide
im = ImCanvas(35, 35)
im._draw_image('../data/projects/im2im1/class_images/blocks_1.png')
im


# In[9]:


# hide
im._clear_image()


# The `Im2ImAnnotatorGUI` uses the `ImCanvas` developed before and the component `CaptureGrid` that displays the selectable options on the view.

# In[10]:


#exporti

class Im2ImAnnotatorGUI(AppLayout):
    debug_output = Output(layout={'border': '1px solid black'})

    def __init__(
        self,
        app_state: AppWidgetState,
        im2im_state: Im2ImState,
        state_to_widget: LabelStoreCaster,
        label_autosize=False,
        on_save_btn_clicked: Callable = None,
        on_grid_box_clicked: Callable = None,
        on_navi_clicked: Callable = None,
        has_border: bool = False,
        fit_canvas: bool = False
    ):
        self._app_state = app_state
        self._im2im_state = im2im_state
        self._on_save_btn_clicked = on_save_btn_clicked
        self._on_navi_clicked = on_navi_clicked
        self._on_grid_box_clicked = on_grid_box_clicked
        self.state_to_widget = state_to_widget

        if label_autosize:
            if self._im2im_state.im_width < 100 or self._im2im_state.im_height < 100:
                self._im2im_state.grid.width = 10
                self._im2im_state.grid.height = 10
            elif self._im2im_state.im_width > 1000 or self._im2im_state.im_height > 1000:
                self._im2im_state.grid.width = 50
                self._im2im_state.grid.height = 10
            else:
                label_width = min(self._im2im_state.im_width, self._im2im_state.im_height) // 10
                self._im2im_state.grid.width = label_width
                self._im2im_state.grid.height = label_width

        self._image = ImCanvas(
            width=self._im2im_state.im_width,
            height=self._im2im_state.im_height,
            has_border=has_border,
            fit_canvas=fit_canvas
        )

        self._navi = Navi()
        self._navi.on_navi_clicked = self.on_navi_clicked
        self._save_btn = Button(description="Save",
                                layout=Layout(width='auto'))

        self._controls_box = HBox(
            [self._navi, self._save_btn],
            layout=Layout(
                display='flex',
                justify_content='center',
                flex_flow='wrap',
                align_items='center'
            )
        )

        self._grid_box = GridMenu(self._im2im_state.grid)

        self._grid_label = HTML(value="<b>LABEL</b>",)
        self._labels_box = VBox(
            children=[self._grid_label, self._grid_box],
            layout=Layout(
                display='flex',
                justify_content='center',
                align_items='center')
        )

        self._save_btn.on_click(self._on_btn_clicked)
        self._grid_box.on_click(self.on_grid_clicked)

        if self._app_state.max_im_number:
            self._set_navi_max_im_number(self._app_state.max_im_number)

        if self._im2im_state.annotations:
            self._grid_box.load(
                self.state_to_widget(self._im2im_state.annotations)
            )

        if self._im2im_state.question_value:
            self._set_label(self._im2im_state.question_value)

        self._im2im_state.subscribe(self._set_label, 'question_value')
        self._im2im_state.subscribe(self._image._draw_image, 'image_path')
        self._im2im_state.subscribe(self.load_menu, 'annotations')

        layout = Layout(
            display='flex',
            justify_content='center',
            align_items='center'
        )

        im2im_display = HBox([
            VBox([self._image, self._controls_box]),
            self._labels_box
        ], layout=layout)

        super().__init__(
            header=None,
            left_sidebar=None,
            center=im2im_display,
            right_sidebar=None,
            footer=None,
            pane_widths=(6, 4, 0),
            pane_heights=(1, 1, 1))

    @debug_output.capture(clear_output=False)
    def load_menu(self, annotations: LabelStore):
        self._grid_box.load(
            self.state_to_widget(annotations)
        )

    @debug_output.capture(clear_output=False)
    def on_navi_clicked(self, index: int):
        if self._on_navi_clicked:
            self._on_navi_clicked(index)

    def _set_navi_max_im_number(self, max_im_number: int):
        self._navi.max_im_num = max_im_number

    def _set_label(self, question_value: str):
        self._grid_label.value = question_value

    def _on_btn_clicked(self, *args):
        if self._on_save_btn_clicked:
            self._on_save_btn_clicked(*args)
        else:
            warnings.warn("Save button click didn't triggered any event.")

    @debug_output.capture(clear_output=False)
    def on_grid_clicked(self, event, value=None):
        if self._on_grid_box_clicked:
            self._on_grid_box_clicked(event, value)
        else:
            warnings.warn("Grid box click didn't triggered any event.")

    def on_client_ready(self, callback):
        self._image.observe_client_ready(callback)


# In[11]:


# hide
label_state = {
    '../data/projects/im2im1/class_images/blocks_1.png': {'answer': False},
    '../data/projects/im2im1/class_images/blocks_9.png': {'answer': False},
    '../data/projects/im2im1/class_images/blocks_12.png': {'answer': True},
    '../data/projects/im2im1/class_images/blocks_32.png': {'answer': False},
    '../data/projects/im2im1/class_images/blocks_37.png': {'answer': False},
    '../data/projects/im2im1/class_images/blocks_69.png': {'answer': True}
}


# In[12]:


grid = Grid(
    width=50,
    height=50,
    n_rows=2,
    n_cols=3
)
im2im_state_dict = {
    'im_height': 200,
    'im_width': 200,
    'grid': grid
}

output = OutputImageLabel()
state_to_widget = LabelStoreCaster(output)

app_state = AppWidgetState()
im2im_state = Im2ImState(**im2im_state_dict)  # type: ignore

im2im_ = Im2ImAnnotatorGUI(
    state_to_widget=state_to_widget,
    app_state=app_state,
    im2im_state=im2im_state
)

im2im_._im2im_state.image_path = '../data/projects/im2im1/pics/Grass1.png'
im2im_


# In[13]:


# hide
im2im_._grid_box.load(state_to_widget(label_state))  # type: ignore


# In[14]:


#exporti
def _label_state_to_storage_format(label_state):
    return [Path(k).name for k, v in label_state.items() if v['answer']]


# In[15]:


# hide
label_state_storage = _label_state_to_storage_format(label_state)
label_state_storage


# In[16]:


#exporti
def _storage_format_to_label_state(
    storage_format,
    label_names,
    label_dir: str
):
    try:
        path = Path(label_dir)
        return {str(path / label): {
            'answer': label in storage_format} for label in label_names}
    except Exception:
        return {label: {'answer': label in storage_format} for label in label_names}


# In[17]:


# hide
from fastcore.test import test_eq

label_names = ['blocks_1.png', 'blocks_9.png', 'blocks_12.png',
               'blocks_32.png', 'blocks_37.png', 'blocks_69.png']

restored_label_state = _storage_format_to_label_state(
    label_state_storage, label_names, '../data/projects/im2im1/class_images/')
test_eq(label_state, restored_label_state)


# In[18]:


# hide
import tempfile
tmp_dir = tempfile.TemporaryDirectory()

print(tmp_dir.name)


# In[19]:


# hide
# dataset generator annotation format

# annotations = {
#     'img_0.jpg': {'labels': [('red', 'rectangle'), ('red', 'rectangle')],
#                   'bboxs': [(3, 21, 82, 82), (19, 98, 82, 145)]},
#     'img_1.jpg': {'labels': [('blue', 'ellipse')],
#                   'bboxs': [(22, 51, 67, 84)]},
#     'img_2.jpg': {'labels': [('yellow', 'ellipse'), ('yellow', 'ellipse'), ('blue', 'rectangle')],
#                   'bboxs': [(75, 33, 128, 120), (4, 66, 59, 95), (30, 35, 75, 62)]},
#     'img_3.jpg': {'labels': [('blue', 'ellipse'), ('red', 'ellipse'), ('yellow', 'ellipse')],
#                   'bboxs': [(47, 55, 116, 96), (99, 27, 138, 50), (0, 3, 47, 56)]}
# }


# In[20]:


# hide
#old ipyannotator annotation format

annotations = {
    str(Path(tmp_dir.name) / 'img_0.jpg'): ['yellow.jpg'],
    str(Path(tmp_dir.name) / 'img_1.jpg'): ['red'],
    str(Path(tmp_dir.name) / 'img_2.jpg'): ['red'],
    str(Path(tmp_dir.name) / 'img_3.jpg'): ['red'],
    str(Path(tmp_dir.name) / 'img_4.jpg'): ['yellow'],
    str(Path(tmp_dir.name) / 'img_5.jpg'): ['yellow'],
    str(Path(tmp_dir.name) / 'img_6.jpg'): ['yellow'],
    str(Path(tmp_dir.name) / 'img_7.jpg'): ['blue'],
    str(Path(tmp_dir.name) / 'img_8.jpg'): ['blue'],
    str(Path(tmp_dir.name) / 'img_9.jpg'): ['yellow']
}


# In[21]:


# hide
import json

annot_file = Path(tmp_dir.name) / 'annotations.json'
with open(annot_file, 'w') as f:
    json.dump(annotations, f, indent=2)


# In[22]:


# hide
from ipyannotator.storage import validate_project_path
project_path = validate_project_path('../data/projects/im2im1/')


# In[23]:


# hide
imz = InputImage('pics')

lblz = OutputImageLabel(label_dir='class_images')


# ## Controller

# In[24]:


#exporti

class Im2ImAnnotatorController:
    debug_output = Output(layout={'border': '1px solid black'})

    def __init__(
        self,
        app_state: AppWidgetState,
        im2im_state: Im2ImState,
        storage: JsonLabelStorage,
        input_item=None,
        output_item=None,
        question=None,
    ):
        self._app_state = app_state
        self._im2im_state = im2im_state
        self._storage = storage

        self.input_item = input_item
        self.output_item = output_item

        self.images = self._storage.get_im_names(None)
        self._app_state.max_im_number = len(self.images)

        self.labels = self._storage.get_labels()
        self.labels_num = len(self.labels)

        # Tracks the app_state.index history
        self._last_index = 0

        if question:
            self._im2im_state.question_value = (f'<center><p style="font-size:20px;">'
                                                f'{question}</p></center>')

    def _update_im(self):
        # print('_update_im')
        index = self._app_state.index
        self._im2im_state.image_path = str(self.images[index])

    def _update_state(self, change=None):  # from annotations
        # print('_update_state')
        image_path = self._im2im_state.image_path

        if not image_path:
            return
        tmp_annotations = LabelStore()
        if image_path in self._storage:
            current_annotation = self._storage.get(str(image_path)) or {}
            tmp_annotations.update(
                _storage_format_to_label_state(
                    storage_format=current_annotation or [],
                    label_names=self.labels,
                    label_dir=self._storage.label_dir
                )
            )
            self._im2im_state.annotations = tmp_annotations

    def _update_annotations(self, index: int):  # from screen
        # print('_update_annotations')
        image_path = self._im2im_state.image_path
        if image_path:
            self._storage[image_path] = _label_state_to_storage_format(
                self._im2im_state.annotations
            )

    def save_annotations(self, index: int):  # to disk
        # print('_save_annotations')
        self._update_annotations(index)
        self._storage.save()

    def idx_changed(self, index: int):
        """ On index change save old state
            and update current state for visualisation
        """
        # print('_idx_changed')
        self._app_state.set_quietly('index', index)
        self.save_annotations(self._last_index)
        # update new screen
        self._update_im()
        self._update_state()
        self._last_index = index

    @debug_output.capture(clear_output=False)
    def handle_grid_click(self, event, name):
        # print('_handle_grid_click')
        label_changed = name

        # check if the im2im is using the label as path
        # otherwise it uses the iterable of labels
        if isinstance(self._storage.label_dir, Path):
            label_changed = self._storage.label_dir / name

            if label_changed.is_dir():
                # button without image - invalid
                return

            label_changed = str(label_changed)
        current_label_state = deepcopy(self._im2im_state.annotations)

        # inverse state
        current_label_state[label_changed] = {
            'answer': not self._im2im_state.annotations[label_changed].get('answer', False)
        }

        # change traitlets.Dict entirely to have change events issued
        self._im2im_state.annotations = current_label_state

    def handle_client_ready(self):
        self._update_im()
        self._update_state()

    def to_dict(self, only_annotated: bool) -> dict:
        return self._storage.to_dict(only_annotated)


# In[25]:


# remove if the results folder exists this allows
# the next command to construct the annotation path
get_ipython().system(' rm -rf ../data/projects/im2im1/results')


# In[26]:


# hide
from ipyannotator.storage import construct_annotation_path

anno_file_path = construct_annotation_path(project_path)

app_state = AppWidgetState()
im2im_state = Im2ImState(grid=grid)

storage = JsonLabelStorage(
    im_dir=project_path / imz.dir,
    label_dir=project_path / lblz.dir,
    annotation_file_path=anno_file_path
)

i_ = Im2ImAnnotatorController(
    app_state=app_state,
    im2im_state=im2im_state,
    storage=storage,
    input_item=imz,
    output_item=lblz
)


# In[27]:


# hide
# (Next-> button emulation)
# Increment index to initiate annotation save and switch state for a new screen


# In[28]:


# # hide

# # "Im2ImAnnotatorController" has no attribute "index"
i_.index = 2  # type: ignore
dict(i_._im2im_state.annotations)


# In[29]:


# export

class Im2ImAnnotator(Annotator):
    """
    Represents image-to-image annotator.

    Gives an ability to itarate through image dataset,
    map images with labels for classification,
    export final annotations in json format

    """

    def __init__(
        self,
        project_path: Path,
        input_item: InputImage,
        output_item: Union[OutputImageLabel, OutputLabel],
        annotation_file_path,
        n_rows=None,
        n_cols=None,
        label_autosize=False,
        question=None,
        has_border=False
    ):
        assert input_item, "WARNING: Provide valid Input"
        assert output_item, "WARNING: Provide valid Output"

        self.project_path = project_path
        self.input_item = input_item
        self.output_item = output_item
        app_state = AppWidgetState(uuid=str(id(self)))

        super().__init__(app_state)

        grid = Grid(
            width=output_item.width,
            height=output_item.height,
            n_rows=n_rows,
            n_cols=n_cols
        )

        self.im2im_state = Im2ImState(
            uuid=str(id(self)),
            grid=grid,
            annotations=LabelStore(),
            im_height=input_item.height,
            im_width=input_item.width
        )

        self.storage = JsonLabelStorage(
            im_dir=project_path / input_item.dir,
            label_dir=self._get_label_dir(),
            annotation_file_path=annotation_file_path
        )

        self.controller = Im2ImAnnotatorController(
            app_state=self.app_state,
            im2im_state=self.im2im_state,
            storage=self.storage,
            input_item=input_item,
            output_item=output_item,
            question=question,
        )

        self.state_to_widget = LabelStoreCaster(output_item)

        self.view = Im2ImAnnotatorGUI(
            app_state=self.app_state,
            im2im_state=self.im2im_state,
            state_to_widget=self.state_to_widget,
            label_autosize=label_autosize,
            on_navi_clicked=self.controller.idx_changed,
            on_save_btn_clicked=self.controller.save_annotations,
            on_grid_box_clicked=self.controller.handle_grid_click,
            has_border=has_border,
            fit_canvas=input_item.fit_canvas
        )

        self.app_state.subscribe(self._on_annotation_step_change, 'annotation_step')

        # draw current image and bbox only when client is ready
        self.view.on_client_ready(self.controller.handle_client_ready)

    def _on_annotation_step_change(self, annotation_step: AnnotatorStep):
        if annotation_step == AnnotatorStep.EXPLORE:
            self.state_to_widget.widgets_disabled = True
            self.view._grid_box.clear()
        elif self.state_to_widget.widgets_disabled:
            self.state_to_widget.widgets_disabled = False

        # forces annotator to have img loaded
        self.controller._update_im()
        self.controller._update_state()
        self.view.load_menu(self.im2im_state.annotations)

    def _get_label_dir(self) -> Union[Iterable[str], Path]:
        if isinstance(self.output_item, OutputImageLabel):
            return self.project_path / self.output_item.dir
        elif isinstance(self.output_item, OutputLabel):
            return self.output_item.class_labels
        else:
            raise ValueError(
                "output_item should have type OutputLabel or OutputImageLabel"
            )

    def __repr__(self):
        display(self.view)
        return ""

    def to_dict(self, only_annotated=True):
        return self.controller.to_dict(only_annotated)


# In[30]:


get_ipython().system(' rm -rf ..data/projects/im2im1/results')


# In[31]:


proj_path = validate_project_path('../data/projects/im2im1')
anno_file_path = construct_annotation_path(
    file_name='../data/projects/im2im1/results/annotation.json')

in_p = InputImage(image_dir='pics', image_width=300, image_height=300)

out_p = OutputImageLabel(label_dir='class_images', label_width=150, label_height=50)

im2im = Im2ImAnnotator(
    project_path=proj_path,
    input_item=in_p,
    output_item=out_p,
    annotation_file_path=anno_file_path,
    n_cols=2,
    question="HelloWorld"
)

im2im


# In[32]:


#hide
im2im.to_dict()


# In[33]:


@pytest.fixture
def im2im_class_labels_fixture():
    get_ipython().system(' rm -rf ../data/projects/im2im1/results/annotation.json')

    proj_path = validate_project_path('../data/projects/im2im1')
    anno_file_path = construct_annotation_path(
        file_name='../data/projects/im2im1/results/annotation.json')

    in_p = InputImage(image_dir='pics', image_width=300, image_height=300)

    out_p = OutputLabel(class_labels=('horse', 'airplane', 'dog'))

    im2im = Im2ImAnnotator(
        project_path=proj_path,
        input_item=in_p,
        output_item=out_p,
        annotation_file_path=anno_file_path,
        n_cols=2,
        question="Testing classes"
    )

    # force fixture to already load its children
    im2im.controller.idx_changed(0)
    assert len(im2im.view._grid_box.children) > 0

    return im2im


# In[34]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_doesnt_share_state_with_other_annotators(im2im_class_labels_fixture):\n    other_im2im = Im2ImAnnotator(\n        project_path=proj_path,\n        input_item=in_p,\n        output_item=out_p,\n        annotation_file_path=anno_file_path,\n        n_cols=2,\n        question="Hello World"\n    )\n    assert other_im2im.app_state.index == 0\n    other_im2im.app_state.index = 1\n    assert other_im2im.app_state.index != im2im_class_labels_fixture.app_state.index')


# In[35]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_activate_button_on_user_click(im2im_class_labels_fixture):\n    im2im_class_labels_fixture.controller._update_state()\n    buttons = im2im_class_labels_fixture.view._grid_box.children\n    airplane_btn = buttons[0]\n    airplane_btn.click()\n    assert im2im_class_labels_fixture.im2im_state.annotations[airplane_btn.value] == {'answer': True}\n    assert im2im_class_labels_fixture.view._grid_box.children[0].layout.border is not None\n    for button in buttons[1:]:\n        assert button.layout.border is None")


# In[36]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_disables_grid_menu_when_app_state_step_is_explore(im2im_class_labels_fixture):\n    assert len(im2im_class_labels_fixture.view._grid_box.children) > 0\n    im2im_class_labels_fixture.app_state.annotation_step = AnnotatorStep.EXPLORE\n    assert len(im2im_class_labels_fixture.view._grid_box.children) == 3\n    im2im_class_labels_fixture.view._navi._next_btn.click()\n    assert len(im2im_class_labels_fixture.view._grid_box.children) == 3\n    for button in im2im_class_labels_fixture.view._grid_box.children:\n        assert button.disabled is True')


# In[37]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_loads_grid_menu_when_app_state_step_is_not_explore(im2im_class_labels_fixture):\n    im2im_class_labels_fixture.app_state.annotation_step = AnnotatorStep.EXPLORE\n    im2im_class_labels_fixture.app_state.annotation_step = AnnotatorStep.CREATE\n    assert len(im2im_class_labels_fixture.view._grid_box.children) == 3')


# In[38]:


#hide
from nbdev.export import notebook2script
notebook2script()


# In[ ]:





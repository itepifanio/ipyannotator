#!/usr/bin/env python
# coding: utf-8

# In[1]:


# hide
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# default_exp explore_annotator


# In[3]:


#exporti
from ipyannotator.im2im_annotator import ImCanvas
from ipyannotator.base import BaseState, AppWidgetState, Annotator
from ipyannotator.navi_widget import Navi
from ipyannotator.storage import MapeableStorage, get_image_list_from_folder
from ipyannotator.mltypes import InputImage, Output
from abc import ABC, abstractmethod
from IPython.display import display
from pathlib import Path
from ipywidgets import AppLayout, HBox, Layout
from typing import Any, List, Optional


# # Annotator Explorer

# In[4]:


#export

class ExploreAnnotatorState(BaseState):
    image_path: Optional[str]


# In[5]:


#exporti

class ExploreAnnotatorGUI(AppLayout):

    def __init__(
        self,
        app_state: AppWidgetState,
        explorer_state: ExploreAnnotatorState,
        fit_canvas: bool = False,
        has_border: bool = False
    ):
        self._app_state = app_state
        self._state = explorer_state

        self._navi = Navi()

        self._controls_box = HBox(
            [self._navi],
            layout=Layout(
                display='flex',
                flex_flow='row wrap',
                align_items='center'
            )
        )

        self._image = ImCanvas(
            width=self._app_state.size[0],
            height=self._app_state.size[1],
            fit_canvas=fit_canvas,
            has_border=has_border
        )

        # set the values already instantiated on state
        if self._state.image_path:
            self._image._draw_image(self._state.image_path)

        self._listen_max_im_number_changes()
        self._navi.on_navi_clicked = self._update_index

        self._state.subscribe(self._image._draw_image, 'image_path')

        super().__init__(header=None,
                         left_sidebar=None,
                         center=self._image,
                         right_sidebar=None,
                         footer=self._controls_box,
                         pane_widths=(2, 8, 0),
                         pane_heights=(1, 4, 1))

    def _listen_max_im_number_changes(self):
        self._update_max_navi_slider(self._app_state.max_im_number)
        self._app_state.subscribe(self._update_max_navi_slider, 'max_im_number')

    def _update_max_navi_slider(self, max_im_number: int):
        self._navi.max_im_num = max_im_number

    def _update_index(self, index: int):
        self._app_state.index = index

    def on_client_ready(self, callback):
        self._image.observe_client_ready(callback)


# In[6]:


app_state = AppWidgetState()
explorer_state = ExploreAnnotatorState()

e_ = ExploreAnnotatorGUI(
    app_state=app_state,
    explorer_state=explorer_state
)

e_._state.image_path = '../data/projects/im2im1/pics/Grass1.png'
e_


# In[7]:


#exporti
class Storage(ABC):
    @abstractmethod
    def bulk_annotation(self, index: int, annotation: List):
        pass

    @abstractmethod
    def find(self, index: int):
        pass


# In[8]:


#exporti
class InMemoryStorage(Storage, MapeableStorage):
    def __init__(
        self,
        image_dir: Path,
    ):
        super().__init__()
        self.images = sorted(get_image_list_from_folder(image_dir))
        self.update({str(image): [] for image in self.images})

    def get_image(self, index: int) -> str:
        return str(self.images[index])

    def bulk_annotation(self, index: int, annotations: list):
        image_path = self.get_image(index)
        self.mapping[image_path] = annotations

    def find(self, index: int):
        image_path = self.get_image(index)
        return self.__getitem__(image_path)


# In[9]:


#exporti

class ExploreAnnotatorController:
    def __init__(
        self,
        app_state: AppWidgetState,
        explorer_state: ExploreAnnotatorState,
        storage: Storage
    ):
        self._last_index = 0
        self._app_state = app_state
        self._state = explorer_state
        self._storage = storage

        self._app_state.subscribe(self._update_current_frame, 'index')
        self._update_max_im_number()
        self._update_current_frame()

    def _update_max_im_number(self):
        self._app_state.max_im_number = len(self._storage)

    def _update_current_frame(self, index: int = 0):
        self._save_annotation(self._last_index)
        # "Storage" has no attribute "get_image"
        self._state.image_path = self._storage.get_image(index)  # type: ignore
        self._last_index = index

    def _save_annotation(self, index: int):
        annotations: List[Any] = []
        self._storage.bulk_annotation(index, annotations)


# In[10]:


storage = InMemoryStorage(Path('../data/projects/bbox/pics'))


# In[11]:


app_state = AppWidgetState()


# In[12]:


explorer_state = ExploreAnnotatorState()


# In[13]:


controller = ExploreAnnotatorController(app_state, explorer_state, storage)


# In[14]:


ExploreAnnotatorGUI(app_state, explorer_state)


# In[15]:


#export

class ExploreAnnotator(Annotator):
    def __init__(
        self,
        project_path: Path,
        input_item: InputImage,
        output_item: Output,
        has_border: bool = False,
        *args, **kwargs
    ):
        app_state = AppWidgetState(uuid=str(id(self)), **{
            # "Input" has no attribute "width", "height"
            'size': (input_item.width, input_item.height)  # type: ignore
        })

        super().__init__(app_state)

        self._state = ExploreAnnotatorState(uuid=str(id(self)))

        # "Input" has no attribute "dir"
        self._storage = InMemoryStorage(project_path / input_item.dir)  # type: ignore

        self._controller = ExploreAnnotatorController(
            self.app_state,
            self._state,
            self._storage
        )

        self._view = ExploreAnnotatorGUI(
            self.app_state,
            self._state,
            fit_canvas=input_item.fit_canvas,
            has_border=has_border
        )

    def __repr__(self):
        display(self._view)
        return ""


# In[16]:


from ipyannotator.mltypes import NoOutput

exp = ExploreAnnotator(
    project_path=Path('../data/projects/bbox/'),
    input_item=InputImage(image_dir='pics', image_width=400, image_height=400),
    output_item=NoOutput()
)


# In[17]:


#hide
from nbdev.export import notebook2script
notebook2script()


# In[ ]:





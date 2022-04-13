#!/usr/bin/env python
# coding: utf-8

# # Build Annotator - Understanding Ipyannotator design to easily extend and customize
# 
# Ipyannotator is a framework that allows users to *hack* the inbuilt annotators, thus, extend and customize the framework according to their needs. In the other tutorials Ipyannotator API was used in simple annotation projects to display the easy usage. The current tutorial will demonstrate how to build new annotators that can be part of the Ipyannotator API.
# 
# Ipyannotator architecture uses four main layers:
# - The **View** is responsible for rendering the visualizations. Ipyannotator uses [ipycanvas](https://ipycanvas.readthedocs.io/en/latest/) and [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/)  to structure and mount the visualization layer. Additionally, internal components such as the navigation menue were developed which helps the users to navigate through the images that need to be annotated.
# 
# - The **Storage** layer is the layer that receives the data and stores it. Ipyannotator uses different types of storage formats like .txt and .SQLite.
# - The **Controller** layer acts as a mediator between state, storage and view. This layer tells when the information from the state will be stored.
# - **"Model/State (in memory)"** is the central function of the Ipyannotator layer structure. It is assigned to centralize the data and ensures the syncronization across the applications. If something changes in the Model/State layer, the information is passed on to other layers, ensuring  synchronization of information.
# 
# The image below exemplifies how the layers are structured and how the communication path is set up.
# 
# The annotator developed in the current notebook is a minimal example called CircleAnnotator. It draws a circle every time a user clicks on the canvas.

# ![New Component Diagram for annotators](https://www.plantuml.com/plantuml/png/VPA_ReCm48TtFuNbgLt82o0KIXUaP2b3vsjmAPB_87EqYYhUlHUZOEEsP0du--xym-VZYE1mqegrWF06e-JYR5kf3Wq2IlxG6wwbjkxwA3YCl3Pd_yQ_6PkPanS4qoagAeVXjMyxYLvRtyZZz0iYaTeCqntmBJt9TooTir-t9g8en5_IIzzzH5QUBzQSx2GgC9ymHgdYXR3_xn8VCAf80YT5dfPxy6aFLdlmUONp_t6Zf4b8kA1rD9lRdxVeAUauqnZNeUQPoehDCIKi33QQVMKDEgkKT2myaZ-HVo-Fz8R2G2TFSDqM-FuRkDwLp14AQhYhfJ4M0NjhwDCvcmCH3UX1oV5GQ-gtdD6ov1T89plUZGAtMH5tPB5FU4hp7QLf9wr-0000)

# ## Model/State (in memory)
# 
# To develop a model/state layer, Ipyannotator uses [Pydantic models](https://pydantic-docs.helpmanual.io/usage/models/) to determine the data type of the output model. Every change made in a state is monitored using [PyPubSub](https://pypubsub.readthedocs.io/en/v4.0.3/) and the information is passed on to other layers to ensure the synchronization between components.
# 
# For the `CircleAnnotator` we split the data into two states:
# 
# - **AppWidgetState** is a common state for all annotators. The `AppWidgetState` stores the canvas size, navigation index and maximum number of images. You can use it to communicate with the Ipyannotator navigation component (Navi) or on your own custom navigation component.
# - **CircleAnnotatorState** is the state responsible to store the `CircleAnnotator` data. Is stores the circle radius, view layers, current image, and circle drawn.
# 
# **Observation:** The model/state doesn't have to be restricted to a single class (as shown in the image above). Its data should make sense according to the structure of the annotator. 

# In[1]:


from pubsub import pub
from typing import Tuple, List, Dict, Optional
from ipyannotator.base import BaseState, AppWidgetState, Annotator
from abc import ABC, abstractmethod
from IPython.display import display


# In[2]:


class CircleAnnotatorState(BaseState):
    radius: float = 30
    current_frame: Optional[str]
    circles: Dict = {}
    layers: Dict[str, int] = {
        'bg': 0,
        'image': 1,
        'circle': 2,
    }


# ## View
# 
# The view layer should stores all ipywidgets that are used by the annotator. The next commands will start the GUI for the CircleAnnotator.

# In[3]:


from ipywidgets import AppLayout, VBox, HBox, Layout, Output, Image
from ipycanvas import MultiCanvas
from pathlib import Path
from ipyannotator.navi_widget import Navi
from ipyannotator.bbox_canvas import ImageRenderer, draw_bg
from ipyannotator.debug_utils import IpyLogger
from ipyannotator.storage import MapeableStorage, get_image_list_from_folder


# The `CircleCanvas` class will be a component of our GUI, allowing to draw circles, backgrounds, images and also clears them.

# In[4]:


class CircleCanvas(HBox):
    debug_output = Output(layout={'border': '1px solid black'})

    def __init__(self, width: float, height: float, layers: dict):
        super().__init__()

        self._multi_canvas = MultiCanvas(
            len(layers),
            width=width,
            height=height,
        )

        children = [VBox([self._multi_canvas])]
        self.children = children

    def clear(self, layer: int):
        self._multi_canvas[layer].clear()

    def draw_circle(self, layer: int, x: float, y: float, radius: float):
        self._multi_canvas[layer].stroke_circle(x, y, radius)

    def _draw_bg(self, layer: int = 0):
        draw_bg(self._multi_canvas[layer])

    def draw_image(self, layer: int, image_path: str):
        ImageRenderer(clear=True).render(self._multi_canvas[layer], image_path)


# In[5]:


circle_canvas = CircleCanvas(width=200, height=200, layers={'image': 1, 'bg': 0, 'circle': 2})
circle_canvas.draw_image(0, '../data/projects/bbox/pics/blueSquare800x600.png')


# In[6]:


circle_canvas.debug_output


# In[7]:


circle_canvas


# In[8]:


circle_canvas.draw_circle(1, 63, 62, 15)


# The ` CircleAnnotatorGUI ` corresponds to the view layer. This layer communicates with the states, for example, if the state index changes the view layer will clear the draw layer, change the image and redraw the circles that were load to the state.

# In[9]:


class CircleAnnotatorGUI(AppLayout):
    debug_output = Output(layout={'border': '1px solid black'})

    def __init__(self, app_widget: AppWidgetState, circle_state: CircleAnnotatorState):
        self._app_widget = app_widget
        self._circle_state = circle_state

        self._navi = Navi()

        self._controls_box = HBox(
            [self._navi],
            layout=Layout(
                display='flex',
                flex_flow='row wrap',
                align_items='center'
            )
        )

        self._image_box = CircleCanvas(
            width=self._app_widget.size[0],
            height=self._app_widget.size[1],
            layers=self._circle_state.layers
        )

        self._listen_index_changes()
        self._listen_click()
        self._listen_max_im_number_changes()
        self._navi.on_navi_clicked = self._update_index

        super().__init__(
            header=None,
            left_sidebar=None,
            center=self._image_box,
            right_sidebar=None,
            footer=self._controls_box,
            pane_widths=(2, 8, 0),
            pane_heights=(1, 4, 1))

    def _listen_click(self):
        layer = self._circle_state.layers['circle']
        self._image_box._multi_canvas[layer].on_mouse_down(self._draw_circle)

    def _draw_circle(self, x: float, y: float, radius: float = None, append=True):
        layer = self._circle_state.layers['circle']

        draw = {
            'x': x,
            'y': y,
            'radius': radius or self._circle_state.radius,
        }

        if append:
            self._circle_state.circles[self._circle_state.current_frame].append(draw)

        self._image_box.draw_circle(layer, draw['x'], draw['y'], draw['radius'])

    def _draw_circle_from_state(self, frame: str):
        circles = self._circle_state.circles[frame]

        for circle in circles:
            circle['append'] = False
            self._draw_circle(**circle)

    def _draw_image(self, image_path: str):
        image_layer = self._circle_state.layers['image']
        self._image_box.draw_image(image_layer, image_path)

    def _listen_max_im_number_changes(self):
        self._update_max_navi_slider(self._app_widget.max_im_number)
        self._app_widget.subscribe(self._update_max_navi_slider, 'max_im_number')

    def _update_max_navi_slider(self, max_im_number: int):
        self._navi.max_im_num = max_im_number

    def _listen_index_changes(self):
        if self._circle_state.current_frame:
            self._change_image(self._circle_state.current_frame)
        self._circle_state.subscribe(self._change_image, 'current_frame')

    def _change_image(self, current_frame: str):
        self._image_box.clear(self._circle_state.layers['circle'])
        self._draw_image(current_frame)
        self._draw_circle_from_state(current_frame)

    def _update_index(self, index: int):
        self._app_widget.index = index


# ## Storage
# 
# Ipyannotator uses JSON as a data structure to store the annotation data. The package also allows the users to change the type of storage according to the users needs. For example, you can store your data in files or databases like SQlite. In this tutorial a `Storage` module is developed that keeps our data in memory (using the `InMemoryStorage` class).

# In[10]:


class Storage(ABC):
    @abstractmethod
    def bulk_annotation(self, index: int, annotation: list):
        pass

    @abstractmethod
    def find(self, index: int):
        pass


# In[11]:


class InMemoryStorage(Storage, MapeableStorage):
    def __init__(
        self,
        image_dir: Path,
    ):
        super().__init__()
        self.images = get_image_list_from_folder(image_dir)
        self.update({str(image): [] for image in self.images})

    def get_image(self, index: int) -> str:
        return str(self.images[index])  # type: ignore

    def bulk_annotation(self, index: int, annotations: list):
        image_path = self.get_image(index)
        self.mapping[image_path] = annotations

    def find(self, index: int):
        image_path = self.get_image(index)
        return self.__getitem__(image_path)


# ## Controller
# 
# The controller serves as a mediator between the states, the GUI, and the storage. This layer listens to states changes and stores the data on the storage. It can also load the storage data into the states.
# 
# To demonstrate how the communication works, the `IpyLogger` class can be used as a [decorator](https://docs.python.org/3/glossary.html#term-decorator) to output all the pubsub communication into the logger. The `pub.ALL_TOPICS` parameter will get all the messages.

# In[12]:


logger = IpyLogger('CircleLogger')


# In[13]:


logger.show_logs()


# In[14]:


@logger.subscribe('AppWidgetState')
class CircleAnnotatorController:
    def __init__(
        self,
        app_widget: AppWidgetState,
        circle_state: CircleAnnotatorState,
        storage: Storage
    ):
        self._last_index = 0
        self._app_widget = app_widget
        self._circle_state = circle_state
        self._storage = storage
        self._storage_to_state()

        app_widget.subscribe(self._update_current_frame, 'index')

    def _storage_to_state(self):
        logger.info("-> _storage_to_state")
        self._circle_state.circles = self._storage.mapping
        self._update_current_frame()
        self._update_max_im_number()

    def _update_max_im_number(self):
        logger.info("-> _update_max_im_number")
        self._app_widget.max_im_number = len(self._storage)

    def _update_current_frame(self, index: int = 0):
        self._save_annotation(self._last_index)
        # error: "Storage" has no attribute "get_image"
        image_path = self._storage.get_image(index)  # type: ignore
        self._circle_state.current_frame = image_path
        self._last_index = index

    def _save_annotation(self, index: int):
        logger.info("-> _save_annotation")
        # error: "Storage" has no attribute "get_image"
        image_path = self._storage.get_image(index)  # type: ignore
        annotations = self._circle_state.circles[image_path]
        self._storage.bulk_annotation(index, annotations)


# In[15]:


storage = InMemoryStorage(Path('../data/projects/bbox/pics'))


# In[16]:


app_widget = AppWidgetState()


# In[17]:


circle_state = CircleAnnotatorState()


# In[18]:


controller = CircleAnnotatorController(app_widget, circle_state, storage)


# In[19]:


CircleAnnotatorGUI(app_widget, circle_state)


# In[20]:


logger.show_logs()


# ## Annotator
# 
# The Ipyannotator design can be described by three properties: input, output, actions. The goal is to develop flexible modules with a common interface.
# 
# With all `CircleAnnotator` layers developed we can now create a single instance. For the current annotator these are the used properties:
# 
# - input: Image
# - output: Circle
# - actions: explore, improve, create

# In[21]:


from ipyannotator.mltypes import InputImage, Input, Output


# In[22]:


class CircleOutput(Output):
    pass


# In[23]:


class CircleAnnotator(Annotator):
    def __init__(
        self,
        project_path: Path,
        input_item: InputImage,
        output_item: Output,
        *args, **kwargs
    ):
        app_state = AppWidgetState(uuid=str(id(self)), **{
            'size': (input_item.width, input_item.height)
        })

        super().__init__(app_state)

        self._circle_state = CircleAnnotatorState(uuid=str(id(self)))

        self._storage = InMemoryStorage(project_path / input_item.dir)

        self._controller = CircleAnnotatorController(
            self.app_state,
            self._circle_state,
            self._storage
        )

        self._view = CircleAnnotatorGUI(
            self.app_state,
            self._circle_state
        )

    def __repr__(self):
        display(self._view)
        return ""


# In[24]:


in_p = InputImage(image_dir='pics', image_width=600, image_height=400)
out_p = CircleOutput()

circle_annotator = CircleAnnotator(
    project_path=Path('../data/projects/bbox/'),
    input_item=in_p,
    output_item=out_p
)

circle_annotator


# ## Diagram

# The following sequence diagram shows how the CircleAnnotator communicates with its components when a user clicks on the next button navigation.
# 
# ![sequence diagram](https://www.plantuml.com/plantuml/png/XLFDRjim3BxpARZsqXts0aPHD3z0CM0R0WJ13aLXq3Rh2f5bJwBIP4y_EfQxE32sKoJo-o7rnOz1o4jiB8IzSHrvQZ3mhyYkvEyS0jMyiAPsw4tz9l2fyrGtXCBjRnGV6M1HIkjn5zW35EqH-IXR8M6yxOpRWqgAAKr7Jd3HTJzDLNC2K43gkk6C4-3A-DBomhbMIDNF461NeHeCBZTFkwytUFkjd-h4rhRlsXSZfskkuiv6Ud-APWJze8D97TV_tjfUgB2HSQgp8dUWaA3bPIcQnAezi_ixNTawyQqzMwpIkRTPYRSNFYFkUjv4iUml79KwCGCDA39kTiljRjdZDbk4YeGA2enRbQ6Q-_fw2T17WryUXaKpT1fG8GxwgvQ7mJ8CBBbn5H_XND3EHpWPnax5UUZZVKdM5bJk7_0vTxfbtXUeiFm2L8evAFI32-D11NNA3EzrJ_DwKgwfZYzGyGmbLHGFcwqI7oxU8SCyJLD6xzb9_kgfuIGqqY0HqYRhPOP5jFkSXcSshGjtba9Q-VCCl6PjDbJpNV8PeQEDec2zLFXaECyIlSCpCpnFg9DbbJpndFtB3wbCznmLPWbmNPn_-OdYPAmP_cmUwNFICCetSZKFJtKTGa8fu_hJoL1lv37jz0zSvUazgVyNDbG3FBAhOcF_0000)

# In[ ]:





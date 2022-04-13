#!/usr/bin/env python
# coding: utf-8

# In[1]:


# default_exp base


# # Base

# In[2]:


# hide
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


#exporti
import json
import random
from pubsub import pub
from pathlib import Path
from enum import Enum, auto
from typing import NamedTuple, Optional, Tuple, Any, Callable
from abc import ABC
from pydantic import BaseModel, BaseSettings


# In[4]:


# hide
import ipytest
import pytest
ipytest.autoconfig(raise_on_error=True)


# ## Ipyannotator base
# 
# The current notebook define the classes, enum and helper functions that will be used on the whole application.

# ## State

# In[5]:


#exporti

class StateSettings(BaseSettings):
    class Config:
        validate_assignment = True


class BaseState(StateSettings, BaseModel):
    def __init__(self, uuid: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_quietly('_uuid', uuid)
        self.set_quietly('event_map', {})

    def set_quietly(self, key: str, value: Any):
        """
        Assigns a value to a state's attribute.

        This function can be used to avoid that
        the state dispatches a PyPubSub event.
        It's very usefull to avoid event recursion,
        ex: a component is listening for an event A
        but it also changes the state that dispatch
        the event A. Using set_quietly to set the
        value at the component will avoid the recursion.
        """
        object.__setattr__(self, key, value)

    @property
    def root_topic(self) -> str:
        if hasattr(self, '_uuid') and self._uuid:  # type: ignore
            return f'{self._uuid}.{type(self).__name__}'  # type: ignore

        return type(self).__name__

    def subscribe(self, change: Callable, attribute: str):
        key = f'{self.root_topic}.{attribute}'
        self.event_map[key] = change  # type: ignore
        pub.subscribe(change, key)

    def unsubscribe(self, attribute: str):
        key = self.topic_attribute(attribute)
        pub.unsubscribe(self.event_map[key], key)  # type: ignore
        del self.event_map[key]  # type: ignore

    def topic_attribute(self, attribute: str):
        return f'{self.root_topic}.{attribute}'

    def is_subscribed(self, attribute: str) -> bool:
        return attribute in self.event_map  # type: ignore

    def __setattr__(self, key: str, value: Any):
        self.set_quietly(key, value)

        if key != '__class__':
            pub.sendMessage(f'{self.root_topic}.{key}', **{key: value})


# In[6]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_can_unsubscribe():\n    count = 0\n    class Increment(BaseState):\n        inc: int = 1\n\n    def incrementing(inc):\n        nonlocal count\n        count += inc\n\n    state = Increment()\n    state.subscribe(incrementing, 'inc')\n    state.inc = 1\n    assert count == 1\n    state.unsubscribe('inc')\n    state.inc = 1\n    assert count == 1")


# ## Annotator
# 
# All annotator share some states and types, the next cells will design this shared features.

# Ipyannotator's uses a `create`, `explore`, `improve` steps when handling data in it's annotators. This enum will be used across the application to check and change the annotators on every step change

# In[7]:


#exporti
class AnnotatorStep(Enum):
    EXPLORE = auto()
    CREATE = auto()
    IMPROVE = auto()


# In[8]:


#exporti

class AppWidgetState(BaseState):
    annotation_step: AnnotatorStep = AnnotatorStep.CREATE
    size: Tuple[int, int] = (640, 400)
    max_im_number: int = 1
    index: int = 0


# The following cells will define a common interface for all annotators. Every annotator has a `app_state` that should be implemented.

# In[9]:


#exporti
class Annotator(ABC):
    def __init__(self, app_state: AppWidgetState):
        self.app_state = app_state


# In[10]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_raises_not_implemented_app_state():\n    with pytest.raises(TypeError):\n        class Anno(Annotator):\n            pass\n\n        anno = Anno()\n        anno.app_state')


# In[11]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_not_raises_if_implemented_app_state():\n    try:\n        class Anno(Annotator):\n            def __init__(self):\n                self._app_state = AppWidgetState()\n                \n            @property\n            def app_state(self):\n                return self._app_state\n\n        anno = Anno()\n        anno.app_state\n    except:\n        pytest.fail("Anno couldn\'t call app_state")')


# ## Helpers

# In[12]:


#exporti

class Settings(NamedTuple):
    project_path: Path = Path('user_project')
    project_file: Optional[Path] = None
    image_dir: str = 'images'
    label_dir: Optional[str] = None
    result_dir: Optional[str] = None

    im_width: int = 50
    im_height: int = 50
    label_width: int = 50
    label_height: int = 50

    n_cols: int = 3
    n_rows: Optional[int] = None


# In[13]:


#exporti


def generate_subset_anno_json(project_path: Path, project_file,
                              number_of_labels,
                              out_filename='subset_anno.json'):
    """
    generates random subset from full dataset based on <number_of_labels>
    """
    if number_of_labels == -1:
        return project_file

    with project_file.open() as f:
        data = json.load(f)

    all_labels = data.values()
    unique_labels = set(label for item_labels in all_labels for label in item_labels)

    #  get <number_of_labels> random labels and generate annotation file with them:
    assert (number_of_labels <= len(unique_labels))
    subset_labels = random.sample([[a] for a in unique_labels], k=number_of_labels)
    subset_annotations = {k: v for k, v in data.items() if v in subset_labels}

    subset_file = Path(project_path) / out_filename
    with subset_file.open('w', encoding='utf-8') as fi:
        json.dump(subset_annotations, fi, ensure_ascii=False, indent=4)

    return subset_file


# In[14]:


#exporti
def validate_project_path(project_path):
    project_path = Path(project_path)
    assert project_path.exists(), "WARNING: Project path should point to "                                   "existing directory"
    assert project_path.is_dir(), "WARNING: Project path should point to "                                   "existing directory"
    return project_path


# In[15]:


# hide
im2im_proj_path = validate_project_path('../data/projects/im2im1/')


# In[16]:


#hide
from nbdev.export import notebook2script
notebook2script()


# In[ ]:





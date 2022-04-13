#!/usr/bin/env python
# coding: utf-8

# In[1]:


# default_exp ipytyping.annotations


# In[2]:


# hide
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


#exporti
from pathlib import Path
from collections.abc import MutableMapping
from typing import Dict, Optional, Iterable, Any, Union
from ipywidgets import Layout
from ipyannotator.mltypes import OutputImageLabel, OutputLabel
from ipyannotator.custom_input.buttons import ImageButton, ImageButtonSetting, ActionButton


# In[4]:


# hide
import ipytest
import pytest
ipytest.autoconfig(raise_on_error=True)


# ## Annotation Types
# 
# The current notebook store the annotation data typing. Every annotator stores its data in a particular way, this notebook designs the store and it's casting types.

# In[5]:


#exporti
class AnnotationStore(MutableMapping):
    def __init__(self, annotations: Optional[Dict] = None):
        self._annotations = annotations or {}

    def __getitem__(self, key: str):
        return self._annotations[key]

    def __delitem__(self, key: str):
        if key in self:
            del self._annotations[key]

    def __setitem__(self, key: str, value: Any):
        self._annotations[key] = value

    def __iter__(self):
        return iter(self._annotations)

    def __len__(self):
        return len(self._annotations)

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self._annotations)


# ### LabelStore Data Type
# 
# The `LabelStore` stores a path as a key it's answer in the format: `{'<path>': {'answer': <bool>}`.

# In[6]:


#exporti
class LabelStore(AnnotationStore):
    def __getitem__(self, key: str):
        assert isinstance(key, str)
        return self._annotations[key]

    def __delitem__(self, key: str):
        assert isinstance(key, str)
        if key in self:
            del self._annotations[key]

    def __setitem__(self, key: str, value: Optional[Dict[str, bool]]):
        assert isinstance(key, str)
        if value:
            assert isinstance(value, dict)
        self._annotations[key] = value


# The following cell will define a cast from the annotation to a custom widget called `ImageButton`.

# In[7]:


#exporti
def _label_store_to_image_button(
    annotation: LabelStore,
    width: int = 150,
    height: int = 150,
    disabled: bool = False
) -> Iterable[ImageButton]:
    button_setting = ImageButtonSetting(
        display_label=False,
        image_width=f'{width}px',
        image_height=f'{height}px'
    )

    buttons = []

    for path, value in annotation.items():
        image_button = ImageButton(button_setting)
        image_button.image_path = str(path)
        image_button.label_value = Path(path).stem
        image_button.active = value.get('answer', False)
        image_button.disabled = disabled
        buttons.append(image_button)

    return buttons


# In[8]:


#exporti
def _label_store_to_button(
    annotation: LabelStore,
    disabled: bool
) -> Iterable[ActionButton]:
    layout = {
        'width': 'auto',
        'height': 'auto'
    }
    buttons = []

    for label, value in annotation.items():
        button = ActionButton(layout=Layout(**layout))
        button.description = label
        button.value = label
        button.tooltip = label
        button.disabled = disabled
        if value.get('answer', True):
            button.layout.border = 'solid 2px #1B8CF3'
        buttons.append(button)

    return buttons


# In[9]:


#exporti
class LabelStoreCaster:  # pylint: disable=too-few-public-methods
    """Factory that casts the correctly widget
    accordingly with the input"""

    def __init__(
        self,
        output: Union[OutputImageLabel, OutputLabel],
        width: int = 150,
        height: int = 150,
        widgets_disabled: bool = False
    ):
        self.width = width
        self.height = height
        self.output = output
        self.widgets_disabled = widgets_disabled

    def __call__(self, annotation: LabelStore) -> Iterable:
        if isinstance(self.output, OutputImageLabel):
            return _label_store_to_image_button(
                annotation,
                self.width,
                self.height,
                self.widgets_disabled
            )

        if isinstance(self.output, OutputLabel):
            return _label_store_to_button(
                annotation,
                disabled=self.widgets_disabled
            )

        raise ValueError(
            f"output should have type OutputImageLabel or OutputLabel. {type(self.output)} given"
        )


# In[10]:


@pytest.fixture
def str_label_fixture():
    return {
        'A': {'answer': False},
        'B': {'answer': True}
    }


# In[11]:


@pytest.fixture
def img_label_fixture():
    return {
        '../data/projects/capture1/pics/pink25x25.png': {'answer': False},
    }


# In[12]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_cast_label_store_to_image_button(img_label_fixture):\n    label_store = LabelStore()\n    label_store.update(img_label_fixture)\n    \n    output = OutputImageLabel()\n    caster = LabelStoreCaster(output)\n    image_buttons = caster(label_store)\n\n    for image_button in image_buttons:\n        assert isinstance(image_button, ImageButton)\n    assert len(image_buttons) == 1')


# In[13]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_cast_label_store_to_button(str_label_fixture):    \n    label_store = LabelStore()\n    label_store.update(str_label_fixture)\n    \n    output = OutputLabel(class_labels=list(str_label_fixture.keys()))\n    caster = LabelStoreCaster(output)\n    buttons = caster(label_store)\n\n    assert len(buttons) == 2\n    for button in buttons:\n        assert isinstance(button, ActionButton)\n    assert buttons[0].description == 'A'\n    assert buttons[1].description == 'B'\n    assert buttons[0].value == 'A'\n    assert buttons[1].value == 'B'")


# In[14]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_can_disable_widgets(str_label_fixture):\n    label_store = LabelStore()\n    label_store.update(str_label_fixture)\n    \n    output = OutputLabel(class_labels=list(str_label_fixture.keys()))\n    caster = LabelStoreCaster(output, widgets_disabled=True)\n    buttons = caster(label_store)\n    for button in buttons:\n        assert button.disabled is True')


# In[15]:


#hide
from nbdev.export import notebook2script
notebook2script()


# In[ ]:





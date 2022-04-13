#!/usr/bin/env python
# coding: utf-8

# In[1]:


# default_exp custom_input.buttons


# In[2]:


# hide
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


# hide
from nbdev import *


# In[4]:


#exporti
from pathlib import Path

import attr
from ipyevents import Event
from ipywidgets import Image, VBox, Layout, Output, HTML
from traitlets import Bool, Unicode, HasTraits, observe
from typing import Optional, Union, Any


# In[5]:


# hide

from IPython.display import display


# # Image button

# In[6]:


#exporti
@attr.define(slots=False)
class ImageButtonSetting:
    im_path: Optional[str] = None
    label: Optional[Union[HTML, str]] = None
    im_name: Optional[str] = None
    im_index: Optional[Any] = None
    display_label: bool = True
    image_width: str = '50px'
    image_height: Optional[str] = None


# In[7]:


#export

class ImageButton(VBox, HasTraits):
    """
    Represents simple image with label and toggle button functionality.

    # Class methods

    - clear(): Clear image infos

    - on_click(p_function): Handle click events

    # Class methods

    - clear(): Clear image infos

    - on_click(p_function): Handle click events

    - reset_callbacks(): Reset event callbacks
    """
    debug_output = Output(layout={'border': '1px solid black'})
    active = Bool()
    image_path = Unicode()
    label_value = Unicode()

    def __init__(self, setting: ImageButtonSetting):

        self.setting = setting
        self.image = Image(
            layout=Layout(display='flex',
                          justify_content='center',
                          align_items='center',
                          align_content='center',
                          width=setting.image_width,
                          margin='0 0 0 0',
                          height=setting.image_height),
        )

        if self.setting.display_label:  # both image and label
            self.setting.label = HTML(
                value='?',
                layout=Layout(display='flex',
                              justify_content='center',
                              align_items='center',
                              align_content='center'),
            )
        else:  # no label (capture image case)
            self.im_name = self.setting.im_name
            self.im_index = self.setting.im_index
            self.image.layout.border = 'solid 1px gray'
            self.image.layout.object_fit = 'contain'
            self.image.margin = '0 0 0 0'
            self.image.layout.overflow = 'hidden'

        super().__init__(layout=Layout(align_items='center',
                                       margin='3px',
                                       overflow='hidden',
                                       padding='2px'))
        if not setting.im_path:
            self.clear()

        self.d = Event(source=self, watched_events=['click'])

    @observe('image_path')
    def _read_image(self, change=None):
        new_path = change['new']
        if new_path:
            self.image.value = open(new_path, "rb").read()
            if not self.children:
                self.children = (self.image,)
                if self.setting.display_label:
                    self.children += (self.setting.label,)
        else:
            #do not display image widget
            self.children = tuple()

    @observe('label_value')
    def _read_label(self, change=None):
        new_label = change['new']

        if isinstance(self.setting.label, HTML):
            self.setting.label.value = new_label
        else:
            self.setting.label = new_label

    def clear(self):
        if isinstance(self.setting.label, HTML):
            self.setting.label.value = ''
        else:
            self.setting.label = ''
        self.image_path = ''
        self.active = False

    @observe('active')
    def mark(self, ev):
        # pad to compensate self size with border
        if self.active:
            if self.setting.display_label:
                self.layout.border = 'solid 2px #1B8CF3'
                self.layout.padding = '0px'
            else:
                self.image.layout.border = 'solid 3px #1B8CF3'
                self.image.layout.padding = '0px'
        else:
            if self.setting.display_label:
                self.layout.border = 'none'
                self.layout.padding = '2px'
            else:
                self.image.layout.border = 'solid 1px gray'

    def __eq__(self, other):
        equals = [
            other.image_path == self.image_path,
            other.label_value == self.label_value,
            other.active == self.active,
        ]

        return all(equals)

    def update(self, other):
        if self != other:
            self.image_path = other.image_path
            self.label_value = other.label_value
            self.active = other.active

    @property
    def value(self):
        return Path(self.image_path).name

    @debug_output.capture(clear_output=False)
    def on_click(self, cb):
        self.d.on_dom_event(cb)

    def reset_callbacks(self):
        self.d.reset_callbacks()


# In[8]:


# hide
im = Image()
im.value = open('../data/mock/pics/test200x200.png', "rb").read()
im


# In[9]:


# hide
setting = ImageButtonSetting()
imb = ImageButton(setting)
display(imb), display(imb.debug_output)


# In[10]:


# hide
assert not imb.active
imb.value


# In[11]:


# hide
h = HTML('Event info')
display(h)


# In[12]:


# hide
from functools import partial


def handle_event(event, name=None):
    event.update({'bname': name})
    lines = ['{}: {}'.format(k, v) for k, v in event.items() if k in ['bname', 'type']]
    content = '<br>'.join(lines)
    h.value = content


imb.on_click(partial(handle_event, name='imb'))
imb.d._dom_handlers.callbacks


# In[13]:


# hide
imb.reset_callbacks()
imb.d._dom_handlers.callbacks


# In[14]:


# hide
imb.image_path = '../data/mock/pics/test200x200.png'
imb.label_value = 'new_label'
imb.active = True
assert imb.value == 'test200x200.png'


# In[15]:


# hide
imb.clear()


# In[16]:


# hide
button_setting = ImageButtonSetting(
    im_path='../data/mock/pics/test200x200.png',
    label='hm',
    display_label=False
)
im_button = ImageButton(button_setting)


def handle_event_(event, name=None):
    if name == im_button.name:
        im_button.active = not im_button.active


im_button.on_click(partial(handle_event_, name='test200x200.png'))

display(im_button)


# In[17]:


#hide
from nbdev.export import notebook2script
notebook2script()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


# hide
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# default_exp custom_input.buttons


# In[3]:


# hide
from nbdev import *


# In[4]:


#exporti

from ipywidgets import Button


# # Custom Buttons

# In[5]:


#exporti

class ActionButton(Button):
    def __init__(self, value=None, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def reset_callbacks(self):
        self.on_click(None, remove=True)

    def update(self, other):
        self.value = other.value
        self.layout = other.layout


# In[6]:


action_btn = ActionButton()
action_btn


# In[7]:


# it can retrieve button value on callback

value = 2
action_btn.value = value

new_value = None


def on_btn_click(event: ActionButton):
    global new_value
    new_value = event.value


action_btn.on_click(on_btn_click)
action_btn.click()

assert new_value == value


# In[8]:


# it can accept default value

action_btn = ActionButton(value=value)
assert action_btn.value == value


# In[9]:


#hide
from nbdev.export import notebook2script
notebook2script()


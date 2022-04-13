#!/usr/bin/env python
# coding: utf-8

# In[1]:


# hide
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from nbdev import *
# default_exp navi_widget


# # Navi Widget

# In[3]:


#exporti
from ipywidgets import Button, IntSlider, HBox, Layout
import warnings
from typing import Callable


# In[4]:


#exporti

class NaviGUI(HBox):
    def __init__(self, max_im_number: int = 0):
        self._im_number_slider = IntSlider(
            min=0,
            max=max_im_number,
            value=0,
            description='Image Nr.'
        )

        self._prev_btn = Button(description='< Previous',
                                layout=Layout(width='auto'))

        self._next_btn = Button(description='Next >',
                                layout=Layout(width='auto'))

        super().__init__(children=[self._prev_btn, self._im_number_slider, self._next_btn],
                         layout=Layout(display='flex', flex_flow='row wrap', align_items='center'))


# In[5]:


#exporti

class NaviLogic:
    """
    Acts like an intermediator between GUI and its interactions
    """

    def __init__(self, gui: NaviGUI):
        self._gui = gui

    def slider_updated(self, change: dict):
        self._gui._index = change['new']
        self.set_slider_value(change['new'])

    def set_slider_value(self, index: int):
        self._gui._im_number_slider.value = index

    def set_slider_max(self, max_im_number: int):
        self._gui._im_number_slider.max = max_im_number

    def _increment_state_index(self, index: int):
        max_im_number = self._gui._max_im_num
        safe_index = (self._gui._index + index) % max_im_number
        self._gui._index = (safe_index + max_im_number) % max_im_number
        self.set_slider_value(self._gui._index)

    def check_im_num(self, max_im_number: int):
        if not hasattr(self._gui, '_im_number_slider'):
            return
        self._gui._im_number_slider.max = max_im_number - 1


# In[6]:


#export

class Navi(NaviGUI):
    """
    Represents simple navigation module with slider.

    on_navi_clicked: callable
        A callback that runs after every navigation
        change. The callback should have, as a
        parameter the navi's index.
    """

    def __init__(self, max_im_num: int = 1, on_navi_clicked: Callable = None):
        super().__init__(max_im_num)
        self._max_im_num = max_im_num
        self.on_navi_clicked = on_navi_clicked
        self._index = 0

        self.model = NaviLogic(gui=self)

        self._listen_next_click()
        self._listen_prev_click()
        self._listen_slider_changes()

    @property
    def index(self) -> int:
        return self._index

    @index.setter
    def index(self, value: int):
        self.model.set_slider_value(value)
        self._index = value
        self._external_call()

    @property
    def max_im_num(self) -> int:
        return self._max_im_num

    @max_im_num.setter
    def max_im_num(self, value: int):
        self.model.set_slider_max(value - 1)
        self._max_im_num = value

    def _next_clicked(self, *args):
        self.model._increment_state_index(1)

    def _slider_updated(self, value: dict):
        self.model.slider_updated(value)
        self._external_call()

    def _prev_clicked(self, *args):
        self.model._increment_state_index(-1)

    def _listen_slider_changes(self):
        self._im_number_slider.observe(
            self._slider_updated, names='value'
        )

    def _listen_next_click(self):
        self._next_btn.on_click(self._next_clicked)

    def _listen_prev_click(self):
        self._prev_btn.on_click(self._prev_clicked)

    def _external_call(self):
        if self.on_navi_clicked:
            self.on_navi_clicked(self._index)
        else:
            warnings.warn(
                "Navi callable was not defined."
                "The navigation will not trigger any action!"
            )


# In[7]:


# it start navi with slider index at 0

navi = Navi(6)

assert navi._im_number_slider.value == 0

# it changes state if slider.value changes

navi._im_number_slider.value = 2

assert navi._index == 2

# it changes state and slider.value if button is clicked

navi._next_btn.click()

assert navi._index == 3
assert navi._im_number_slider.value == 3

navi._prev_btn.click()

assert navi._index == 2
assert navi._im_number_slider.value == 2

# it changes slider.max if navi changes max im num
navi.max_im_num = 6
assert navi._im_number_slider.max == 5

# it changes slider.index if navi changes its index
navi.index = 3
assert navi._im_number_slider.value == 3

# testing callback
callback_index = 0


def increment_callback(index):
    global callback_index
    callback_index = index


navi.on_navi_clicked = increment_callback

navi._next_btn.click()
assert callback_index == 4

navi._prev_btn.click()
assert callback_index == 3

navi.index = 2
assert callback_index == 2


# In[8]:


navi


# In[9]:


#hide
from nbdev.export import notebook2script
notebook2script()


# In[ ]:





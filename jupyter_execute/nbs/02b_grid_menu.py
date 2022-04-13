#!/usr/bin/env python
# coding: utf-8

# In[1]:


# default_exp custom_widgets.grid_menu


# In[2]:


# hide
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


# hide
from nbdev import *


# In[4]:


#exporti
from math import ceil
from functools import partial
from typing import Callable, Iterable, Optional, Tuple
import warnings
import attr
from ipywidgets import GridBox, Output, Layout


# In[5]:


# hide
import pytest
import ipytest
ipytest.autoconfig(raise_on_error=True)


# ## Grid Menu
# 
# The current notebook develop a grid menu widget that allows clickable widgets to be displayed as grid. The next cell will design the `Grid` class that contain the settings for the `GridMenu` component.

# In[6]:


#exporti
@attr.define(slots=False)
class Grid:
    width: int
    height: int
    n_rows: Optional[int] = 3
    n_cols: Optional[int] = 3
    disp_number: int = 9
    display_label: bool = False

    @property
    def num_items(self) -> int:
        row, col = self.area_adjusted(self.disp_number)
        return row * col

    def area_adjusted(self, n_total: int) -> Tuple[int, int]:
        """Returns the row and col automatic arranged"""
        if self.n_cols is None:
            if self.n_rows is None:  # automatic arrange
                label_cols = 3
                label_rows = ceil(n_total / label_cols)
            else:  # calc cols to show all labels
                label_rows = self.n_rows
                label_cols = ceil(n_total / label_rows)
        else:
            if self.n_rows is None:  # calc rows to show all labels
                label_cols = self.n_cols
                label_rows = ceil(n_total / label_cols)
            else:  # user defined
                label_cols = self.n_cols
                label_rows = self.n_rows

        return label_rows, label_cols


# In[7]:


@pytest.fixture
def grid_fixture() -> Grid:
    return Grid(width=300, height=300)


# In[8]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_return_num_items(grid_fixture):\n    assert grid_fixture.num_items == 9')


# In[9]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_adjusts_area_missing_args(grid_fixture):\n    grid_fixture.n_rows = None\n    assert grid_fixture.area_adjusted(12) == (4, 3)')


# The `GridMenu` doesn't have a `on_click` event listener, but grid elements itself should implement `on_click(ev)`, `reset_callbacks()` and `update(other: SameWidgetType)` methods to register/reset onclick callback function and update its internal values, respectively. Also grid element shoudl have a field name in order user can destinguish between grid children.

# In[10]:


#export
class GridMenu(GridBox):
    debug_output = Output(layout={'border': '1px solid black'})

    def __init__(
        self,
        grid: Grid,
        widgets: Optional[Iterable] = None,
    ):
        self.callback = None
        self.gap = 40 if grid.display_label else 15
        self.grid = grid

        n_row, n_col = grid.area_adjusted(grid.disp_number)
        column = grid.width + self.gap
        row = grid.height + self.gap
        centered_settings = {
            'grid_template_columns': " ".join([f'{(column)}px' for _
                                               in range(n_col)]),
            'grid_template_rows': " ".join([f'{row}px' for _
                                            in range(n_row)]),
            'justify_content': 'center',
            'align_content': 'space-around'
        }

        super().__init__(
            layout=Layout(**centered_settings)
        )

        if widgets:
            self.load(widgets)
        self.widgets = widgets

    def _fill_widgets(self, widgets: Iterable):
        if self.widgets is None:
            self.widgets = widgets

            self.children = self.widgets

            if self.callback:
                self.register_on_click()
        else:
            iter_state = iter(widgets)

            for widget in self.widgets:
                i_widget = next(iter_state, None)
                if i_widget:
                    widget.update(i_widget)
                else:
                    widget.clear()

    def _filter_widgets(self, widgets: Iterable) -> Iterable:
        """Limit the number of widgets to be rendered
        according to the grid's area"""
        widgets_list = list(widgets)  # Iterable don't have len()
        num_widgets = len(widgets_list)
        row, col = self.grid.area_adjusted(num_widgets)
        num_items = row * col

        if num_widgets > num_items:
            warnings.warn("!! Not all labels shown. Check n_cols, n_rows args !!")
            return widgets_list[:num_items]

        return widgets

    @debug_output.capture(clear_output=False)
    def load(self, widgets: Iterable, callback: Optional[Callable] = None):
        widgets_filtered = self._filter_widgets(widgets)
        self._fill_widgets(widgets_filtered)

        if callback:
            self.on_click(callback)

    @debug_output.capture(clear_output=False)
    def on_click(self, callback: Callable):
        setattr(self, 'callback', callback)
        self.register_on_click()

    @debug_output.capture(clear_output=False)
    def register_on_click(self):
        if self.widgets:
            for widget in self.widgets:
                widget.reset_callbacks()

                widget.on_click(
                    partial(
                        self.callback,
                        value=widget.value
                    )
                )

    def clear(self):
        self.widgets = None
        self.children = tuple()


# We now can instantiate the grid menu and load widgets on it. For this example we're using the custom widget `ImageButton` to be displayed using the load function. 

# In[11]:


from ipyannotator.custom_input.buttons import ImageButton, ImageButtonSetting
from ipywidgets import HTML
from IPython.display import display


# In[12]:


grid = Grid(width=50, height=75, n_cols=2, n_rows=2)
grid_menu = GridMenu(grid)


# In[13]:


widgets = []
setting = ImageButtonSetting(im_path='../data/projects/capture1/pics/pink25x25.png')
for i in range(4):
    widgets.append(ImageButton(setting))
grid_menu.load(widgets)


# In[14]:


grid_menu


# In[15]:


widgets = []
setting = ImageButtonSetting(im_path='../data/projects/capture1/pics/teal50x50_5.png')
for i in range(2):
    widgets.append(ImageButton(setting))
grid_menu.load(widgets)


# 
# While ipyevents implementation lacks `sender` or `source` in callback args, `functools.partial` used to back element `name` into return value. You can see example of on_click event handler `test_handler` below. 
# name of the button is printed out on click.

# In[16]:


# hide
h = HTML('Event info')
display(h)


def test_handler(event, value=None):
    event.update({'label_name': value})
    h.value = event['label_name']


grid_menu.on_click(test_handler)


# In[17]:


#hide
from ipyannotator.custom_input.buttons import ActionButton


# In[18]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_doesnt_load_more_widgets_than_the_grid_area():\n    with warnings.catch_warnings(record=True) as w:\n        grid = Grid(width=50, height=75, n_cols=1, n_rows=1)\n        grid_menu = GridMenu(grid)\n        widgets = [ActionButton() for _ in range(2)]\n        grid_menu.load(widgets)\n        assert len(grid_menu.widgets) == 1\n    assert bool(w) is True')


# In[19]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_doesnt_throw_warning_if_number_of_widgets_is_small():\n    with warnings.catch_warnings(record=True) as w:\n        grid = Grid(width=100, height=100, n_rows=2, n_cols=2)\n        grid_menu = GridMenu(grid)\n        grid_menu._filter_widgets([1])\n    assert bool(w) is False')


# In[20]:


#hide
from nbdev.export import notebook2script
notebook2script()


# In[ ]:





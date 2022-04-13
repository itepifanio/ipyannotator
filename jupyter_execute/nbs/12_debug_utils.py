#!/usr/bin/env python
# coding: utf-8

# In[1]:


# default_exp debug_utils


# In[2]:


# hide
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


#exporti
import logging
from typing import List, Any

from IPython.core.display import display
from ipywidgets import Output
from pubsub import pub


# # Debug tools

# ## Output

# It's necessary to have a output to where show the `log`. To use the default `logging` lib with `ipywidgets`, one could use the `OutputWidgetHandler` class.

# In[4]:


#exporti
class OutputWidgetHandler(logging.Handler):
    """ Custom logging handler sending logs to an output widget """

    def __init__(self, *args, **kwargs):
        super(OutputWidgetHandler, self).__init__(*args, **kwargs)
        layout = {
            'border': '1px solid black'
        }
        self.out = Output(layout=layout)

    def emit(self, record):
        """ Overload of logging.Handler method """
        formatted_record = self.format(record)
        new_output = {
            'name': 'stdout',
            'output_type': 'stream',
            'text': formatted_record + '\n'
        }
        self.out.outputs = (new_output, ) + self.out.outputs

    def show_logs(self):
        """ Show the logs """
        display(self.out)

    def clear_logs(self):
        """ Clear the current logs """
        self.out.clear_output()


# In[5]:


# example of logger configuration
example_logger = logging.getLogger(__name__)
example_handler = OutputWidgetHandler()
example_handler.setFormatter(logging.Formatter('%(asctime)s  - [%(levelname)s] %(message)s'))
example_logger.addHandler(example_handler)
example_logger.setLevel(logging.INFO)


# ## Debug tools

# In[6]:


#export

class IpyLogger:
    """
    Redirects logging and pubsub messages (if subscribed) to output widget.

    Use `@subscribe` class decorator or `subscribe_to_states` method to listen pubsub events.
    """

    def __init__(self, class_name: str, log_level=logging.INFO):
        self._class_name = class_name

        # config the logger/output
        logger = logging.getLogger(__name__)
        handler = OutputWidgetHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(log_level)
        self._logger = logger
        self._handler = handler

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._logger.critical(msg, *args, **kwargs)

    def subscribe(self, states):
        def wrapper(cls):
            def inside_wrapper(*args, **kwargs):
                self.subscribe_to_states(states=states)
                return cls(*args, **kwargs)
            return inside_wrapper
        return wrapper

    def show_logs(self):
        return self._handler.show_logs()

    def clear_logs(self):
        return self._handler.clear_logs()

    def subscribe_to_states(self, states: List[Any]):
        states = self._validate_states(states)
        for state in states:
            pub.subscribe(self._pub_handler, state)

    def _pub_handler(self, topic_obj=pub.AUTO_TOPIC, *args, **kwargs):
        self._logger.info(f"[{self._class_name} - {topic_obj.getName()}] : {kwargs}")

    @staticmethod
    def _validate_states(states):
        """Avoids errors where string is handled as list"""
        if isinstance(states, str):
            states = [states]
        return states


# In[7]:


logger = IpyLogger("mylogger", log_level=logging.DEBUG)


# In[8]:


logger.show_logs()


# In[9]:


from pubsub import pub


def test_listener(a1, a2=None):
    print('listener:')
    print('a1 =', a1)
    print('a2 =', a2)


pub.subscribe(test_listener, 'rootTopic')

logger.subscribe_to_states(['rootTopic'])


# In[10]:


pub.sendMessage('rootTopic', a1=123, a2=dict(a=456, b='abc'))


# In[11]:


pub.sendMessage('rootTopic', a1=122, a2=334)


# In[12]:


# l.info("Hey info", stack_info=True)
logger.debug("Hey debug")


# In[13]:


#hide
from nbdev.export import notebook2script
notebook2script()


# In[ ]:





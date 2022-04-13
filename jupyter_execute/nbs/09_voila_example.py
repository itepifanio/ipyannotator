#!/usr/bin/env python
# coding: utf-8

# In[1]:


# hide
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().system(' rm -rf ../data/projects/bbox/voila_results')


# # Voila - Using Ipyannotator as a standalone web application
# 
# [Voila](https://github.com/voila-dashboards/voila) is a library that turns jupyter notebooks into standalone web applications.
# 
# Voila can be used alongside with Ipyannotator. This allows professional annotators to create annotations without even running a jupyter notebook.
# 
# This notebook displays a bounding box annotator to exemplify how an organization can use Voila to allow external professional annotators to create datasets. 
# 
# To run this example use `voila nbs/09_voila_example.ipynb --enable_nbextensions=True`

# In[2]:


from pathlib import Path
from ipyannotator.storage import construct_annotation_path
from ipyannotator.mltypes import InputImage, OutputImageBbox
from ipyannotator.bbox_annotator import BBoxAnnotator


# In[3]:


input_item = InputImage(image_dir='pics', image_width=640, image_height=400)
output_item = OutputImageBbox(classes=['Label 01', 'Label 02'])
project_path = Path('../data/projects/bbox')
annotation_file_path = construct_annotation_path(project_path, results_dir='voila_results')


# In[4]:


BBoxAnnotator(
    project_path=project_path,
    input_item=input_item,
    output_item=output_item,
    annotation_file_path=annotation_file_path
)


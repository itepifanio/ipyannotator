#!/usr/bin/env python
# coding: utf-8

# In[1]:


#all_slow


# # Image classification - Real project example with CIFAR-10 dataset
# 
# This notebook will exemplify how to do image classification in Ipyannotator using one of the most commonly used datasets in deep learning: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). The dataset contains 60000 32x32 images in 10 classes.

# ## Setup data for a fictive greenfield project
# 
# The first step is to download the dataset. The next cell will use the [pooch](https://github.com/fatiando/pooch) library to easily fetch the data files from s3.

# In[2]:


from pathlib import Path
import pooch


# In[3]:


file_path = pooch.retrieve(
    # URL to one of Pooch's test files
    url="https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz",
    known_hash="sha256:637c5814e11aefcb6ee76d5f59c67ddc8de7f5b5077502a195b0833d1e3e4441",
)


# Pooch retrieves the data to your local machine. The next cell will display the exact path where the files were downloaded.

# In[4]:


file_path


# Since the CITAR-10 dataset is downloaded as a compressed `tar` file, the next cells will extract the files. 
# 
# Ipyannotator has some internal tools to manipulate data, which is the case of the `_extract_tar` function used below to extract the files and move them to a new folder `tmp`.

# In[5]:


from ipyannotator.datasets.download import _extract_tar


# In[6]:


_extract_tar(file_path, Path('/tmp'))


# Ipyannotator uses the following path setup:
# 
# ```
# project_root
# │
# │─── images
# │
# └─── results
# ```
# 
# The `project root` is the folder that contains folders for the image raw data and the annotation results. `Images` is the folder that contains all images that can displayed by the navigator and are used to create the dataset by the annotator. The `results` folder stores the dataset. The folder names can be chosen by the user. By default Ipyannotator uses `images` and `results`.
# 
# The next cell defines a project root called `user_project` and creates a new folder called `images` inside of it.

# In[7]:


project_root = Path('user_project')
(project_root / 'images').mkdir(parents=True, exist_ok=True)


# Once the folder structure is created, the files are downloaded and extracted, they will be moved to the `images` folder. 
# 
# The next cell copies the 200 random images from the CIFAR-10 dataset to the Ipyannotator path structure.

# In[8]:


import shutil
import random

classes = "airplane  automobile  bird  cat  deer  dog  frog  horse  ship  truck".split()
for i in range(1, 200):
    rand_class = random.randint(0, 9)
    shutil.copy(
        Path('/tmp') / "cifar10/train" / classes[rand_class] / f"{i:04}.png",
        project_root / 'images')


# ## Story

# In the current step we have 200 images from random classes and we need to classify them. The first step is to have a look at the images before checking which classes need to be set in the classification.
# 
# Ipyannotator uses an API to ensure easy access to the annotators. The next cell will import the `Annotator` factory, that provides a simple function `InputImage` to explore images.

# In[9]:


from ipyannotator.mltypes import InputImage
from ipyannotator.annotator import Annotator


# CIFAR-10 uses 32x32 px color images. The small size of the images makes the visualization difficult. Therefore, the `fit_canvas` property will be used in the next cell to improve the visual appearance, displaying the image at the same size of the `InputImage`.

# In[10]:


input_ = InputImage(image_width=100, image_height=100, image_dir='images', fit_canvas=True)


# To use the `Annotator` factory, a simple pair of `Input/Output` is used. Omitting the output, Ipyannotator will use `NoOutput` as default. In this case, the user can only navigate across the input images and labels/classes are not displayed in the explore function. 

# In[11]:


Annotator(input_).explore()


# As mentioned before, the Ipyannotator path setup provides some default names for the folders. These names can be changed using the `Settings` property. The next cells demonstrates how to use the settings property to customize the folder structure.

# In[12]:


from ipyannotator.base import Settings


# In[13]:


settings = Settings(
    project_path=Path('user_project'),
    image_dir='images',
    result_dir='results'
)


# In[14]:


anni = Annotator(input_, settings=settings)


# In[15]:


anni.explore()


# Once the user has gained an overview on the input image dataset, the user can define classes to label the images. Using `OutputLabel` you can define the classes that will be used to label the images. 
# 
# The `class_labels` property at `OutputLabel` allows an array of classes to be used in the classification. Since CIFAR-10 uses 10 classes, these are going to be used in the next cells.

# In[16]:


from ipyannotator.mltypes import OutputLabel
output_ = OutputLabel(class_labels=classes)


# In[17]:


anni = Annotator(input_, output_, settings)


# In[18]:


anni.explore()


# To create your own dataset you just have to call the `create` step at the `Annotator` factory. This step will allow users to associate the classes to a image.

# In[19]:


anni.create()


# In[ ]:





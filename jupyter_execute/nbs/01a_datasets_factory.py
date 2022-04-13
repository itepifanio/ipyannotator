#!/usr/bin/env python
# coding: utf-8

# In[1]:


# default_exp datasets.factory


# In[2]:


# hide
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


#exporti
import json
from enum import Enum, auto
from pathlib import Path


from ipyannotator.base import Settings
from ipyannotator.datasets.download import (get_cifar10,
                                            get_cub_200_2011,
                                            get_oxford_102_flowers)
from ipyannotator.datasets.generators import (
    create_color_classification,
    create_mot_ds
)


# In[4]:


#exporti
class DS(Enum):
    ARTIFICIAL_VIDEO = auto()
    ARTIFICIAL_BBOX = auto()
    CIFAR10 = auto()
    CUB200 = auto()
    OXFORD102 = auto()


# # Dataset factory

# In[5]:


#export
def get_settings(dataset: DS):
    """
    Handle the necessary to dowload and save the datasets.
    Capable of dowloading CIFAR_10, OXFORD_102_FLOWERS, CUB_200 or a artificial dataset.
    """
    if dataset == DS.ARTIFICIAL_BBOX:
        project_path = Path('data/artificial/')
        project_file = project_path / 'annotations.json'
        image_dir = 'images'
        _, annotations = create_color_classification(path=project_path, n_samples=50,
                                                     size=(500, 500))

        anno = {str(project_path / image_dir / k): [f'{v}.jpg'] for k, v in annotations.items()}

        with open(project_file, 'w') as f:
            json.dump(anno, f)

        return Settings(project_path=project_path,
                        project_file=project_file,
                        image_dir=image_dir,
                        label_dir='class_images',
                        # used on create step - should be empty!
                        result_dir='create_results',
                        im_width=50, im_height=50,
                        label_width=30, label_height=30,
                        n_cols=3)
    elif dataset == DS.ARTIFICIAL_VIDEO:
        project_path = Path('data/artificial/')
        project_file = project_path / 'annotations.json'
        image_dir = 'images'
        create_mot_ds(project_path, image_dir, 20, True)
        return Settings(
            project_path=project_path,
            project_file=project_file,
            image_dir=image_dir,
            im_width=200,
            im_height=200,
            result_dir='create_results',
        )
    elif dataset == DS.CIFAR10:
        cifar_train_p, cifar_test_p = get_cifar10(Path('data'))

        return Settings(project_path=Path('data/cifar10/'),
                        project_file=cifar_test_p,
                        image_dir='test',
                        label_dir=None,
                        # used on create step - should be empty!
                        result_dir='create_results',
                        im_width=50, im_height=50,
                        label_width=140, label_height=30,
                        n_cols=2)

    elif dataset == DS.OXFORD102:
        flowers102_train_p, flowers102_test_p = get_oxford_102_flowers(Path('data'))

        return Settings(project_path=Path('data/oxford-102-flowers'),
                        project_file=flowers102_test_p,
                        image_dir='jpg',
                        label_dir=None,
                        # used on create step - should be empty!
                        result_dir='create_results',
                        im_width=50, im_height=50,
                        label_width=40, label_height=30,
                        n_cols=7)

    elif dataset == DS.CUB200:
        cub200_train_p, cub200_test_p = get_cub_200_2011(Path('data'))

        return Settings(project_path=Path('data/CUB_200_2011'),
                        project_file=cub200_test_p,
                        image_dir='images',
                        label_dir=None,
                        # used on create step - should be empty!
                        result_dir='create_results',
                        im_width=50, im_height=50,
                        label_width=50, label_height=50,
                        n_cols=7)
    else:
        raise UserWarning(f"Dataset {dataset} is not supported!")


# In[6]:


#exporti
def _combine_train_test(project_path: Path):
    # combine train/test in one json file.
    # Used to generate all possible class labels
    all_annotations = project_path / "annotations.json"

    with open(project_path / "annotations_train.json", "rb") as train:
        tr = json.load(train)

    with open(project_path / "annotations_test.json", "rb") as test:
        te = json.load(test)

    with open(all_annotations, "w") as outfile:
        json.dump({**tr, **te}, outfile)
    return all_annotations


# In[7]:


#hide
from nbdev.export import notebook2script
notebook2script()


# In[ ]:





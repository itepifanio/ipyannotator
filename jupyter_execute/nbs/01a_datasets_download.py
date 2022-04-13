#!/usr/bin/env python
# coding: utf-8

# In[1]:


# default_exp datasets.download


# In[2]:


# hide
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


#exporti

import pooch
import glob
import json
from pathlib import Path
import os
import tarfile
import zlib
from typing import Dict, List, Any


# In[4]:


#exporti

def _download_url(url: str, root: Path, filename: str, file_hash: str = None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        file_hash (str, optional): Hash of the required file. If None, will always download the file
    """
    root = Path(os.path.expanduser(root))
    if not filename:
        filename = os.path.basename(url)
    os.makedirs(root, exist_ok=True)

    download_setup = pooch.create(
        path=root,
        base_url=url,
        registry={
            filename: file_hash,
        },
    )
    download_setup.fetch(filename)


# In[5]:


#exporti
def _extract_tar(tar_path: Path, output_dir: Path):
    try:
        print('Extracting...')
        with tarfile.open(tar_path) as tar:
            tar.extractall(output_dir)
    except (tarfile.TarError, IOError, zlib.error) as e:
        print('Failed to extract!', e)


# # Dowload datasets

# In[6]:


#export
def get_cifar10(output_dir: Path):
    """
    Download the cifar10 dataset.
    """

    output_dir = Path(output_dir)
    dataset_dir = output_dir / 'cifar10'

    _download_url(url='https://s3.amazonaws.com/fast-ai-imageclas/',
                  root=output_dir, filename="cifar10.tgz",
                  file_hash=("sha256:637c5814e11aefcb6ee76d5f"
                             "59c67ddc8de7f5b5077502a195b0833d1e3e4441"))

    if not dataset_dir.is_dir():
        _extract_tar(output_dir / 'cifar10.tgz', output_dir)
    else:
        print(f'Directory {dataset_dir} already exists, skip extraction.')

    print('Generating train/test data..')
    imdir_train = dataset_dir / 'train'
    imdir_test = dataset_dir / 'test'

    # split train/test
    train = [Path(p) for p in glob.glob(f'{imdir_train}/*/*')]
    test = [Path(p) for p in glob.glob(f'{imdir_test}/*/*')]

    # generate data for annotations.json
    # {'image-file.jpg': ['label1.jpg']}
    annotations_train = dict((str(p), [f'{p.parts[-2]}.jpg']) for p in train)
    annotations_test = dict((str(p), [f'{p.parts[-2]}.jpg']) for p in test)

    train_path = dataset_dir / 'annotations_train.json'
    test_path = dataset_dir / 'annotations_test.json'

    with open(train_path, 'w') as f:
        json.dump(annotations_train, f)

    with open(test_path, 'w') as f:
        json.dump(annotations_test, f)
    print("Done")
    return train_path, test_path


# In[7]:


# hide
cifar_train_p, cifar_test_p = get_cifar10(Path('data'))


# In[8]:


#export
def get_oxford_102_flowers(output_dir: Path):
    """
    Download the oxford flowers dataset.
    """
    output_dir = Path(output_dir)
    dataset_dir = output_dir / 'oxford-102-flowers'

    _download_url(url='https://s3.amazonaws.com/fast-ai-imageclas/',
                  root=output_dir, filename="oxford-102-flowers.tgz",
                  file_hash=("sha256:680a253086535b2c800aada76a45fc1"
                             "89d3dcae4da6da8db36ce6a95f00bf4ad"))

    if not dataset_dir.is_dir():
        _extract_tar(output_dir / 'oxford-102-flowers.tgz', output_dir)
    else:
        print((f'Directory {dataset_dir} already'
               ' exists, skip extraction.'))

    print('Generating train/test data..')
    with open(dataset_dir / 'train.txt', 'r') as f:
        # https://github.com/python/mypy/issues/7558
        _annotations_train: Dict[str, Any] = dict(tuple(
            line.split()) for line in f)  # type: ignore

    annotations_train: Dict[str, List[str]] = {
        str(dataset_dir / k): [v + '.jpg'] for k, v in _annotations_train.items()
    }

    with open(dataset_dir / 'test.txt', 'r') as f:
        _annotations_test: Dict[str, Any] = dict(tuple(
            line.split()) for line in f)  # type: ignore

    annotations_test: Dict[str, List[str]] = {
        str(dataset_dir / k): [v + '.jpg'] for k, v in _annotations_test.items()
    }

    train_path = dataset_dir / 'annotations_train.json'
    test_path = dataset_dir / 'annotations_test.json'

    with open(train_path, 'w') as f:
        json.dump(annotations_train, f)

    with open(test_path, 'w') as f:
        json.dump(annotations_test, f)
    print("Done")
    return train_path, test_path


# In[9]:


# hide
flowers102_train_p, flowers102_test_p = get_oxford_102_flowers(Path('data'))


# In[10]:


#export
def get_cub_200_2011(output_dir: Path):
    """
    Download the CUB 200 2001 dataset.
    """
    output_dir = Path(output_dir)
    dataset_dir = output_dir / 'CUB_200_2011'

    _download_url(url='https://s3.amazonaws.com/fast-ai-imageclas/',
                  root=output_dir, filename='CUB_200_2011.tgz',
                  file_hash=("sha256:0c685df5597a8b24909f6a7c9db6"
                             "d11e008733779a671760afef78feb49bf081"))

    if not dataset_dir.is_dir():
        _extract_tar(output_dir / 'CUB_200_2011.tgz', output_dir)
    else:
        print(f'Directory {dataset_dir} already exists, skip extraction.')

    print('Generating train/test data..')
    with open(dataset_dir / 'images.txt', 'r') as f:
        image_id_map: Dict[str, Any] = dict(tuple(
            line.split()) for line in f)  # type: ignore

    with open(dataset_dir / 'classes.txt', 'r') as f:
        class_id_map: Dict[str, Any] = dict(tuple(
            line.split()) for line in f)  # type: ignore

    with open(dataset_dir / 'train_test_split.txt', 'r') as f:
        splitter: Dict[str, Any] = dict(tuple(
            line.split()) for line in f)  # type: ignore

    # image ids for test/train
    train_k = [k for k, v in splitter.items() if v == '0']
    test_k = [k for k, v in splitter.items() if v == '1']

    with open(dataset_dir / 'image_class_labels.txt', 'r') as f:
        anno_: Dict[str, Any] = dict(tuple(
            line.split()) for line in f)  # type: ignore

    annotations_train = {
        str(dataset_dir / 'images' / image_id_map[k]):
        [class_id_map[v] + '.jpg'] for k, v in anno_.items() if k in train_k}

    annotations_test = {
        str(dataset_dir / 'images' / image_id_map[k]):
        [class_id_map[v] + '.jpg'] for k, v in anno_.items() if k in test_k}

    train_path = dataset_dir / 'annotations_train.json'
    test_path = dataset_dir / 'annotations_test.json'

    with open(train_path, 'w') as f:
        json.dump(annotations_train, f)

    with open(test_path, 'w') as f:
        json.dump(annotations_test, f)
    print("Done")
    return train_path, test_path


# In[11]:


#hide
cub200_train_p, cub200_test_p = get_cub_200_2011(Path('data'))


# In[12]:


#hide
from nbdev.export import notebook2script
notebook2script()


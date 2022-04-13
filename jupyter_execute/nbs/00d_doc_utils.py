#!/usr/bin/env python
# coding: utf-8

# In[1]:


# default_exp doc_utils


# In[2]:


# hide
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


# hide
from nbdev import *


# In[4]:


#exporti
import os


# # Doc Utils
# 
# This notebook develops helper modules to build Ipyannotator's static documentation.

# The next cell design a helper function that check if the documentation it's been built. This is specially helpful to mock some behaviors that doesn't work well on static docs.

# In[5]:


#exporti
def is_building_docs() -> bool:
    return 'DOCUTILSCONFIG' in os.environ


# ## Docs metadata
# 
# The following cells was extracted from [jb-nbdev](https://github.com/fastai/jb-nbdev) and will perform some changes on our metadata to integrate nbdev and mynb-st.

# In[6]:


#exporti
import glob
from fastcore.all import L, compose, Path
from nbdev.export2html import _mk_flag_re, _re_cell_to_collapse_output, check_re
from nbdev.export import check_re_multi
import nbformat as nbf


# In[7]:


#export
def nbglob(fname='.', recursive=False, extension='.ipynb') -> L:
    """Find all files in a directory matching an extension.
    Ignores hidden directories and filenames starting with `_`"""
    fname = Path(fname)
    if fname.is_dir():
        abs_name = fname.absolute()
        rec_path = f'{abs_name}/**/*{extension}'
        non_rec_path = f'{abs_name}/*{extension}'
        fname = rec_path if recursive else non_rec_path
    fls = L(
        glob.glob(str(fname), recursive=recursive)
    ).filter(
        lambda x: '/.' not in x
    ).map(Path)
    return fls.filter(lambda x: not x.name.startswith('_') and x.name.endswith(extension))


# In[8]:


#exporti
def upd_metadata(cell, tag):
    cell_tags = list(set(cell.get('metadata', {}).get('tags', [])))
    if tag not in cell_tags:
        cell_tags.append(tag)
    cell['metadata']['tags'] = cell_tags


# In[9]:


#export
def hide(cell):
    """Hide inputs of `cell` that need to be hidden
    if check_re_multi(cell, [_re_show_doc, *_re_hide_input]): upd_metadata(cell, 'remove-input')
    elif check_re(cell, _re_hide_output): upd_metadata(cell, 'remove-output')
    """
    regexes = ['#(.+|)hide', '%%ipytest']
    if check_re_multi(cell, regexes):
        upd_metadata(cell, 'remove-cell')

    return cell


_re_cell_to_collapse_input = _mk_flag_re(
    '(collapse_input|collapse-input)', 0, "Cell with #collapse_input")


def collapse_cells(cell):
    "Add a collapse button to inputs or outputs of `cell` in either the open or closed position"
    if check_re(cell, _re_cell_to_collapse_input):
        upd_metadata(cell, 'hide-input')
    elif check_re(cell, _re_cell_to_collapse_output):
        upd_metadata(cell, 'hide-output')
    return cell


# In[10]:


#exporti
if __name__ == '__main__':

    _func = compose(hide, collapse_cells)
    files = nbglob('nbs/')

    for file in files:
        nb = nbf.read(file, nbf.NO_CONVERT)
        for c in nb.cells:
            _func(c)
        nbf.write(nb, file)


# In[11]:


#hide
from nbdev.export import notebook2script
notebook2script()


#!/usr/bin/env python
# coding: utf-8

# In[1]:


# default_exp storage


# In[2]:


# hide
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


# hide
from nbdev import *
from fastcore.test import test_eq


# # Storage

# In[4]:


#exporti
import warnings
import copy
import json
import os
from typing import List, Union, Iterable
from collections import defaultdict
from collections.abc import MutableMapping
from pathlib import Path


# # Project Structure setup
# 
# just to folders, first containing object (e.g images) to annotate and second folder
# contains annotation data/results.

# In[5]:


#exporti
def group_files_by_class(annotations_dict):
    grouped = defaultdict(list)
    for file, labels in annotations_dict.items():
        for class_ in labels:
            grouped[class_].append(file)
    return grouped


# In[6]:


#exporti

def construct_annotation_path(project_path=None, file_name=None, results_dir=None):
    if file_name is not None:
        annotation_file_path = Path(file_name)
        results_dir = annotation_file_path.parent
    elif project_path is not None:
        results_dir = Path(
            project_path, 'results') if results_dir is None else Path(project_path, results_dir)

        annotation_file_path = Path(results_dir, 'annotations.json')
        if annotation_file_path.is_file():
            warnings.warn(f"Error: Annotations file already exists in {results_dir}!"
                          "\n If you want to create annotations from scratch"
                          " - use empty dir!")
    else:
        raise ValueError("You must define `project_path` or `file_name`!")

    results_dir.mkdir(parents=True, exist_ok=True)
    return annotation_file_path


# In[7]:


(construct_annotation_path('../data/test_anno_path'),
 construct_annotation_path(file_name='../results/annotations.json'),
 construct_annotation_path(project_path='../test_anno_path', results_dir='outpi'))


# In[8]:


#exporti
from ipyannotator.base import validate_project_path


def setup_project_paths(project_path, file_name=None, image_dir='pics',
                        label_dir=None, results_dir=None):
    project_path = validate_project_path(project_path)

    im_dir = project_path / image_dir

    if file_name is not None:
        annotation_file_path = Path(file_name)
        results_dir = annotation_file_path.parent
        print(f"WARNING: `results_dir` is deduced from `file_name` path: {results_dir}")
    else:
        results_dir = Path(
            project_path, 'results') if results_dir is None else Path(project_path, results_dir)

        annotation_file_path = Path(results_dir, 'annotations.json')
        if annotation_file_path.is_file():
            print(f"WARNING: Annotations file already exists in {results_dir}"
                  "!\n         If you want to create annotations from scratch"
                  " - use empty dir isntead.")

    results_dir.mkdir(parents=True, exist_ok=True)

    project_paths = (im_dir, annotation_file_path)

    if label_dir is not None:
        project_paths += (Path(project_path, label_dir),)

    return project_paths


# In[9]:


# hide
test_proj_path = Path('../data/test')
setup_project_paths(test_proj_path)


# In[10]:


# hide
test_proj_path = Path('../data/test')
setup_project_paths(test_proj_path, image_dir='ims', label_dir='labels')


# In[11]:


# hide
test_proj_path = Path('../data/test')
setup_project_paths(test_proj_path, image_dir='ims', label_dir='labels', results_dir='.')


# In[12]:


#exporti
import glob


def get_image_list_from_folder(image_dir) -> Iterable[Path]:
    ''' Scans <image_dir> to construct list of existing images as <Path> objects
    '''
    # if no files in `image_dir` assume all images are under `class_name` directories
    if all([Path(image_dir, f).is_dir() for f in os.listdir(image_dir)]):
        path_list = [Path(p) for p in glob.glob(f'{image_dir}/*/*')]
    else:
        path_list = [Path(image_dir, f) for f in os.listdir(image_dir) if
                     os.path.isfile(os.path.join(image_dir, f))]

    return path_list


def strip_path(paths: Iterable[Path]) -> Iterable[str]:
    return [p.name for p in paths]


# In[13]:


# hide
get_image_list_from_folder('../data/mock/pics')


# In[14]:


# hide
strip_path(get_image_list_from_folder('../data/mock/pics'))


# In[15]:


# hide
#generate imdir/classdir/class.jpg structure
subfolders = ['one', 'two', 'three']
for subfolder in subfolders:
    directory = '../data/test_step_down'
    if not os.path.exists(directory):
        os.mkdir(directory)

    directory = f'{directory}/{subfolder}'
    if not os.path.exists(directory):
        os.mkdir(directory)

from PIL import Image

for class_p in [Path(p) for p in glob.glob('../data/test_step_down/*')]:
    img_name = f'{class_p.stem}.jpg'
    img = Image.new('RGB', (50, 50))
    img.save(class_p / img_name)


# In[16]:


# hide
get_image_list_from_folder('../data/test_step_down')


# # Generic Storage for Annotations
# 
# key values store
# 
# - key, object_id / file_name
# - value json blob containing annotation

# In[17]:


#export

class MapeableStorage(MutableMapping):
    def __init__(self):
        self.mapping = {}

    def __getitem__(self, key):
        return self.mapping[key]

    def __delitem__(self, key):
        if key in self:
            del self.mapping[key]

    def __setitem__(self, key, value):
        self.mapping[key] = value

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)

    def to_dict(self, only_annotated=True):
        if only_annotated:
            return {k: copy.deepcopy(v) for k, v in self.mapping.items() if v}
        else:
            return copy.deepcopy(self.mapping)


# In[18]:


m = MapeableStorage()
m.update({'test': 1})
m['test']


# In[19]:


#export

class AnnotationStorage(MapeableStorage):
    """
    Represents generic storage for annotations.

    `key` is object_id / file_name and `value` - json blob containing annotation.

    im_paths - list of existing images as <Path> objects

    """

    def __init__(self, im_paths):
        super().__init__()
        self.update({str(p): None for p in im_paths})

    def __repr__(self):
        return f"{type(self).__name__}({self.mapping})"

    def save(self, file_name):
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(self.mapping, f, ensure_ascii=False, sort_keys=True, indent=4)

    def load(self, file_name):
        with open(file_name) as data_file:
            self.mapping = json.load(data_file)


# In[20]:


# hide
im_paths = [Path('some/path', f) for f in ['name1', 'name2', 'name3']]
storage = AnnotationStorage(im_paths)
storage


# In[21]:


# hide
storage['some/path/name5'] = {'x': 5, 'y': 3, 'width': 7, 'height': 1}


# In[22]:


# hide
test_eq(storage['some/path/name5'], {'x': 5, 'y': 3, 'width': 7, 'height': 1})


# In[23]:


# hide
len_before = len(storage)
storage.pop('some/path/name1')
test_eq(len(storage), len_before - 1)


# In[24]:


# hide
storage.to_dict()


# In[25]:


# hide
storage.to_dict(only_annotated=False)


# In[26]:


# hide
storage.save('/tmp/ttest.json')


# In[27]:


# hide
storage_from_file = AnnotationStorage([])
storage_from_file.load('/tmp/ttest.json')
test_eq(storage, storage_from_file)


# In[28]:


# hide
storage


# In[29]:


# hide
test_eq(storage.get('name8', {'dict': 'obj'}), {'dict': 'obj'})


# In[30]:


# hide
storage.values()


# In[31]:


#exporti
from ipyannotator.helpers import flatten, reconstruct_class_images


class JsonLabelStorage(AnnotationStorage):
    def __init__(self, im_dir: Path, label_dir: Union[Iterable[str], Path], annotation_file_path):
        self.annotation_file_path = annotation_file_path
        self.label_dir = label_dir

        self.has_annotation_file = True if (annotation_file_path is not None and
                                            annotation_file_path.is_file()) else False

        self.images = get_image_list_from_folder(im_dir)

        if isinstance(label_dir, Path):
            # artificialy generate labels if no class images given (TODO: temorary workaround)
            if 'class_autogenerated_' in str(label_dir):
                label_dir.mkdir(parents=True, exist_ok=True)

                if self.has_annotation_file:
                    reconstruct_class_images(label_dir, annotation_file_path, lbl_w=50, lbl_h=50)
                else:
                    warnings.warn("Annotation file should be provided"
                                  " to generate labels automatically!")

            self.labels = strip_path(get_image_list_from_folder(label_dir))
        elif isinstance(label_dir, Iterable):
            self.labels = label_dir
        else:
            raise ValueError("label_dir should have str or Path type")

        if self.has_annotation_file:  # init from json
            self.load()
        else:  # init storage from folder
            super().__init__(self.images)
            self.save()

    def get_im_names(self, filter_files=None):
        keys = self.keys()
        images = sorted([k for k in self.images if str(k) in keys])

        if not images:
            raise UserWarning("!! No Images to dipslay !!")

        if filter_files is not None:
            images = [p for p in images if str(p) in filter_files]

        if not images:
            raise UserWarning("!! No image files to display. Check filter !!")
        return images

    def get_labels(self) -> List[Union[Path, str]]:
        if not self.labels:
            warnings.warn("!! No labels to display !!")
            return []

        if self.has_annotation_file and isinstance(self.label_dir, Path):
            values = set(flatten(self.values()))
            return sorted([v for v in self.labels if str(v) in values])

        # create mod -> display all labels from folder, not json
        return sorted(self.labels)

    def save(self):
        super().save(self.annotation_file_path)

    def load(self):
        super().load(self.annotation_file_path)


# In[32]:


# hide
jas = JsonLabelStorage(
    Path('../data/projects/im2im1/pics'),
    Path('../data/projects/im2im1/class_images'),
    annotation_file_path=Path('../data/projects/im2im1/results/annotations_j.json'))


# In[33]:


# hide
jas.get_labels(), jas.get_im_names()


# In[34]:


#exporti

class JsonCaptureStorage(AnnotationStorage):
    def __init__(self, im_dir: Path, annotation_file_path):
        self.im_dir = im_dir
        self.annotation_file_path = annotation_file_path

        self.has_annotation_file = True if (annotation_file_path is not None and
                                            annotation_file_path.is_file()) else False
        self.images = sorted(get_image_list_from_folder(im_dir))

        if self.has_annotation_file:  # init from json
            self.load()
        else:  # init storage from folder
            super().__init__(self.images)
            self.save()

    def get_im_names(self, filter_files=None):
        images = sorted(k for k in self.images if str(k) in self.keys())

        if not images:
            raise UserWarning("!! No Images to dipslay !!")

        if filter_files is not None:
            images = [p for p in images if str(p) in filter_files]

        if not images:
            raise UserWarning("!! No image files to display. Check filter !!")
        return images

    def save(self):
        super().save(self.annotation_file_path)

    def load(self):
        super().load(self.annotation_file_path)


# # DB backed storage
# 
# - Changes in annotation should be tracked in db.
# - db
#   - sqlite memory / disk, how to sync so that race conditons are avoided?
#   - remote db (postgres, mysql etc.) with sqlalchemy layer
#   
# ## write sqlite functions
# 
# - init db
# - write json + timestamp to db BUT only if json has changed!
# - iterate over db
# - iterate over values with latest timestamp
# - get all history for key
# - allow for metadata?
# - check how sqlite write locks work

# In[35]:


# export

import sqlite3


# In[36]:


#exporti
def _list_tables(conn):
    query = """
    SELECT
        name
    FROM
        sqlite_master
    WHERE
        type = 'table' AND
        name NOT LIKE 'sqlite_%';
    """
    c = conn.cursor()
    return c.execute(query).fetchall()


# ```sql
# DROP TABLE suppliers;
# 
# CREATE TABLE suppliers (
#     supplier_id   INTEGER PRIMARY KEY,
#     supplier_name TEXT    NOT NULL,
#     group_id      INTEGER NOT NULL,
#     FOREIGN KEY (group_id)
#        REFERENCES supplier_groups (group_id) 
# );
# ```

# In[37]:


# hide

conn = sqlite3.connect(":memory:")


# In[38]:


#exporti
def _create_tables(conn):
    c = conn.cursor()
    query = """
CREATE TABLE IF NOT EXISTS data (objectID TEXT,
                                 timestamp DATETIME DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                                 data JSON,
                                 author TEXT,
                                 PRIMARY KEY (objectId, timestamp)
                                );
    """
    c.execute(query)
    query = """
    CREATE TABLE IF NOT EXISTS objects (objectID TEXT,
                                        orderID INTEGER PRIMARY KEY AUTOINCREMENT

                                       )
    """
    c.execute(query)
    conn.commit()


# In[39]:


#exporti
def _list_table(conn, table_name='data', latest=True):
    if latest:
        query = """
        SELECT * from {}

        GROUP BY objectID
        ORDER BY timestamp
        """.format(table_name)
    else:
        query = """
        SELECT * from {}
        """.format(table_name)
    c = conn.cursor()
    return c.execute(query).fetchall()


# In[40]:


# hide
_create_tables(conn)


# In[41]:


# hide
_list_tables(conn)


# ## SQL helper functions
# is needed for consistant iteration order

# In[42]:


#export
def _get_order_id(conn, object_id, table_name='objects'):
    query = """
    SELECT orderID from {}
    WHERE objectID = '{}'
    """.format(table_name, object_id)
    c = conn.cursor()
    res = c.execute(query).fetchone()
    if res is not None:
        return res[0]


# In[43]:


# hide
_get_order_id(conn, 'doesnt exist')


# In[44]:


#export
def _create_order_id(conn, object_id, table_name='objects'):
    order_id = _get_order_id(conn, object_id, table_name=table_name)
    if order_id:
        return order_id
    query = """
    INSERT INTO {}('objectID') VALUES('{}')
    """.format(table_name, object_id)
    c = conn.cursor()
    c.execute(query)
    return _get_order_id(conn, object_id, table_name=table_name)


# In[45]:


# hide
_create_order_id(conn, 'lala')


# In[46]:


# hide
_create_order_id(conn, 'lala')


# In[47]:


# hide
_create_order_id(conn, 'lala2')


# In[48]:


# hide
query = """
SELECT * from objects
"""
c = conn.cursor()
res = c.execute(query).fetchall()
res


# In[49]:


#export
def _get(conn, object_id, table_name='data'):
    query = """
    SELECT data FROM {}
    WHERE objectID = '{}'

    GROUP BY objectID
    ORDER BY timestamp
    """.format(table_name, object_id)
    c = conn.cursor()
    res = c.execute(query).fetchone()
    if res is not None:
        return json.loads(res[0])


# In[50]:


#export
def _get_object_id_at_pos(conn, pos, table_name='objects'):
    query = """
    SELECT objectID FROM {}
    ORDER BY orderID
    LIMIT {}, 1
    """.format(table_name, pos)
    c = conn.cursor()
    res = c.execute(query).fetchone()
    if res is not None:
        return res[0]


# In[51]:


# hide
_get_object_id_at_pos(conn, 1)


# In[52]:


#export
def _insert(conn, object_id, data: dict, table_name='data', author='author'):
    # insert if values have been changed

    last = _get(conn, object_id)

#     if last is None:
    _create_order_id(conn, object_id)
    if data == last:
        return
    c = conn.cursor()
    c.execute("insert into {}('objectID', 'author', 'data') values (?, ?, ?)".format(table_name),
              [object_id, author, json.dumps(data)])
    conn.commit()


# In[53]:


# hide
_insert(conn, 'lala3', {'crazy': 44})
_insert(conn, 'lala2', {'crazy': 40})
import time
time.sleep(0.1)
_insert(conn, 'lala3', {'crazy': 44 + 5})
_insert(conn, 'lala2', {'crazy': 40 + 5})


# In[54]:


# hide
_list_table(conn, latest=False)


# In[55]:


# hide
_list_table(conn)


# In[56]:


# hide
# insert existing is ignored
_insert(conn, 'lala2', {'crazy': 40 + 5})


# In[57]:


# hide
_list_table(conn, latest=False)


# In[58]:


# hide
_get(conn, _get_object_id_at_pos(conn, 2))


# In[59]:


#export
def _to_dict(conn, table_name='data'):
    query = """
    SELECT objectID, data from {}

    GROUP BY objectID
    ORDER BY timestamp
    """.format(table_name)
    c = conn.cursor()
    return {key: json.loads(value) for key, value in c.execute(query).fetchall()}


# In[60]:


# hide
_to_dict(conn)


# In[61]:


# hide
_get(conn, object_id="lala3")


# In[62]:


#export
def _row_count(conn, table_name='data'):
    query = """
    SELECT COUNT(DISTINCT objectID) FROM {}
    """.format(table_name)
    c = conn.cursor()
    res = c.execute(query).fetchone()
    return res[0]


# In[63]:


# hide
_row_count(conn)


# In[64]:


#export
def _delete_last(conn, object_id, table_name='data'):
    query = """
    DELETE FROM {}
    WHERE objectId = '{}'
    ORDER BY timestamp
    LIMIT 1
    """.format(table_name, object_id)
    c = conn.cursor()
    c.execute(query)
    conn.commit()


# In[65]:


#export
def _delete_all(conn, object_id, table_name='data'):
    query = """
    DELETE FROM {}
    WHERE objectId = '{}'
    """.format(table_name, object_id)
    c = conn.cursor()
    c.execute(query)
    conn.commit()


# In[66]:


# hide
_list_table(conn, latest=False)


# In[67]:


# hide
_delete_last(conn, 'lala3')


# In[68]:


# hide
_list_table(conn, latest=False)


# In[69]:


# hide
_delete_all(conn, 'lala2')


# In[70]:


# hide
_list_table(conn, latest=False)


# In[71]:


# hide
_row_count(conn)


# ## Persistent Storage with history support

# In[72]:


#export

class AnnotationStorageIterator:
    def __init__(self, annotator_storage):
        self.annotator_storage = annotator_storage
        self.index = 0

    def __next__(self):
        try:
            result = self.annotator_storage.at(self.index)
            self.index += 1
        except IndexError:
            raise StopIteration
        return result

    def next(self):
        return self.__next__()

    def prev(self):
        self.index -= 1
        if self.index < 0:
            raise StopIteration
        return self.annotator_storage.at(self.index)


# In[73]:


#export

class AnnotationDBStorage(MutableMapping):
    def __init__(self, conn_string, im_paths=None):
        self.conn = sqlite3.connect(conn_string)
        _create_tables(self.conn)
        if im_paths:
            self.update({p.name: {} for p in im_paths})

    def update(self, dict_):
        for k, v in dict_.items():
            _insert(self.conn, k, v)

    def __getitem__(self, key):
        item = _get(self.conn, key)
        if item is None:
            raise IndexError
        return item

    def get(self, key, default):
        if _get(self.conn, key) is None:
            return default

    def __delitem__(self, key):
        _delete_last(self.conn, key)

    def delete_all(self, key):
        _delete_all(self.conn, key)

    def at(self, pos):
        # bug fix needed when combined with del operations
        object_id = _get_object_id_at_pos(self.conn, pos)
        if object_id is None or pos < 0:
            raise IndexError
        return _get(self.conn, object_id)

    def __setitem__(self, key, value):
        _insert(self.conn, key, value)

    def __iter__(self):
        return AnnotationStorageIterator(self)

    def __len__(self):
        return _row_count(self.conn)

    def __repr__(self):
        return f"{type(self).__name__}({_list_table(self.conn)[:2] + [' ...']})"

    def to_dict(self):
        return _to_dict(self.conn)


# In[74]:


# hide
im_paths = [Path('some/path', f) for f in ['name1', 'name2', 'name3']]
_storage = AnnotationDBStorage(":memory:", im_paths)
_storage


# In[75]:


# hide
_storage['name5'] = {'x': 5, 'y': 3, 'width': 7, 'height': 1}


# In[76]:


# hide
test_eq(_storage.at(3), {'x': 5, 'y': 3, 'width': 7, 'height': 1})


# In[77]:


# hide
test_eq(len(_storage), 4)


# In[78]:


# hide
test_eq(_storage['name5'], {'x': 5, 'y': 3, 'width': 7, 'height': 1})


# In[79]:


# hide
myiter = iter(_storage)
for i in range(len(_storage)):
    print(i, _storage.at(i))
    test_eq(_storage.at(i), next(myiter))


# In[80]:


# hide

myiter.prev()  # type: ignore


# In[81]:


# hide
myiter.prev()  # type: ignore


# In[82]:


# hide
myiter.next()  # type: ignore


# In[83]:


# hide
for i in _storage:
    print(i)


# In[84]:


# hide
len_before = len(_storage)
_storage.pop('name1')
test_eq(len(_storage), len_before - 1)


# In[85]:


# hide
_storage.to_dict()


# In[86]:


# hide
for i in range(len(_storage)):
    print(i, _storage.at(i))


# In[87]:


# hide
# TODO delete objectID from object table if not anymore in data


# In[88]:


# hide
_storage


# In[89]:


# hide
test_eq(_storage.get('name8', {'dict': 'obj'}), {'dict': 'obj'})


# In[90]:


# hide
_storage.to_dict()


# In[91]:


#hide
from nbdev.export import notebook2script
notebook2script()


# In[ ]:





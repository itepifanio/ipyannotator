#!/usr/bin/env python
# coding: utf-8

# In[1]:


# default_exp annotator


# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # Annotator Factory
# 
# The current notebook will develop the annotator factory. Given an input and output, the factory will return the corresponding annotator. Once called the user can choose between the three actions available: explore, create or improve.

# In[3]:


#exporti
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Type, List

from skimage import io
from tqdm.notebook import tqdm

from ipyannotator.base import generate_subset_anno_json, Settings, AnnotatorStep
from ipyannotator.mltypes import (Input, Output, OutputVideoBbox,
                                  InputImage, OutputImageLabel, OutputLabel,
                                  OutputImageBbox, OutputGridBox, NoOutput)
from ipyannotator.bbox_annotator import BBoxAnnotator
from ipyannotator.bbox_video_annotator import BBoxVideoAnnotator
from ipyannotator.capture_annotator import CaptureAnnotator
from ipyannotator.datasets.generators import draw_bbox
from ipyannotator.im2im_annotator import Im2ImAnnotator
from ipyannotator.explore_annotator import ExploreAnnotator
from ipyannotator.storage import (construct_annotation_path, group_files_by_class)


# The next cell defines the actual factory implementation, expecting the pair of input/output. The following classes defines all supported annotators with correct input/output pairs for internal use.

# In[4]:


#hide
import ipytest
ipytest.autoconfig(raise_on_error=True)


# In[5]:


#exporti
class AnnotatorFactory(ABC):
    io: Tuple[Type[Input], List[Type[Output]]]

    @abstractmethod
    def get_annotator(self):
        pass

    def __new__(cls, input_item, output_item):
        subclass_map = {}

        for subclass in cls.__subclasses__():
            for subclass_output in subclass.io[1]:
                subclass_map[(subclass.io[0], subclass_output)] = subclass

        try:
            subclass = subclass_map[(type(input_item), type(output_item))]
            instance = super(AnnotatorFactory, subclass).__new__(subclass)
            return instance
        except KeyError:
            print(f"Pair {(input_item, output_item)} is not supported!")


class Bboxer(AnnotatorFactory):
    io = (InputImage, [OutputImageBbox])

    def get_annotator(self):
        return BBoxAnnotator


class Im2Imer(AnnotatorFactory):
    io = (InputImage, [OutputImageLabel, OutputLabel])

    def get_annotator(self):
        return Im2ImAnnotator


class Capturer(AnnotatorFactory):
    io = (InputImage, [OutputGridBox])

    def get_annotator(self):
        return CaptureAnnotator


class ImExplorer(AnnotatorFactory):
    io = (InputImage, [NoOutput])

    def get_annotator(self):
        return ExploreAnnotator


class VideoBboxer(AnnotatorFactory):
    io = (InputImage, [OutputVideoBbox])

    def get_annotator(self):
        return BBoxVideoAnnotator


# In[6]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_get_im2im_annotator_from_output_image_label():\n    inp = InputImage()\n    outp = OutputImageLabel()\n    factory = AnnotatorFactory(inp, outp).get_annotator()\n    assert factory == Im2ImAnnotator')


# In[7]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_get_im2im_annotator_from_output_label():\n    inp = InputImage()\n    outp = OutputLabel(class_labels=('A', 'B'))\n    factory = AnnotatorFactory(inp, outp).get_annotator()\n    assert factory == Im2ImAnnotator")


# The following cell uses the factory designed before and define the actions that can be used with the factory.

# In[8]:


#export
class Annotator:
    def __init__(self, input_item: Input, output_item: Output = NoOutput(),
                 settings: Settings = Settings()):
        self.settings = settings
        self.input_item = input_item
        self.output_item = output_item

    def explore(self, k=-1):
        '''
        Lets visualize existing annotated dataset
        As we don't have images for each class we provide label_dir=None for ipyannotator,
        thus class labels will be generated automatically based on annotation.json file.

        To explore a part of dataset set `k` - number of classes to display;
        By default explore `all` (k == -1)
        '''
        subset = generate_subset_anno_json(project_path=self.settings.project_path,
                                           project_file=self.settings.project_file,
                                           number_of_labels=k)

        anno_ = construct_annotation_path(project_path=self.settings.project_path,
                                          file_name=subset, results_dir=None)

        annotator = AnnotatorFactory(self.input_item, self.output_item).get_annotator()

        self.output_item.drawing_enabled = False
        annotator = annotator(project_path=self.settings.project_path,
                              input_item=self.input_item,
                              output_item=self.output_item,
                              annotation_file_path=anno_,
                              n_cols=self.settings.n_cols,
                              question="Classification <explore>",
                              has_border=True)

        annotator.app_state.annotation_step = AnnotatorStep.EXPLORE

        return annotator

    def create(self):
        anno_ = construct_annotation_path(project_path=self.settings.project_path,
                                          file_name=None,
                                          results_dir=self.settings.result_dir)

        annotator = AnnotatorFactory(self.input_item, self.output_item).get_annotator()

        self.output_item.drawing_enabled = True
        annotator = annotator(project_path=self.settings.project_path,
                              input_item=self.input_item,
                              output_item=self.output_item,
                              annotation_file_path=anno_,
                              n_cols=self.settings.n_cols,
                              question="Classification <create>",
                              has_border=True)

        annotator.app_state.annotation_step = AnnotatorStep.CREATE

        return annotator

    def improve(self):
        # open labels from create step
        create_step_annotations = Path(
            self.settings.project_path) / self.settings.result_dir / 'annotations.json'

        with open(create_step_annotations) as infile:
            loaded_image_annotations = json.load(infile)

        # @TODO?
        if type(self.output_item) == OutputImageLabel:

            #Construct multiple Capturers for each class
            out = []
            for class_name, class_anno in tqdm(
                    group_files_by_class(loaded_image_annotations).items()):
                anno_ = construct_annotation_path(project_path=self.settings.project_path,
                                                  results_dir=(f'{self.settings.result_dir}'
                                                               f'/missed/{class_name[:-4]}'))

                out.append(CaptureAnnotator(self.settings.project_path,
                                            input_item=self.input_item,
                                            output_item=OutputGridBox(),
                                            annotation_file_path=anno_,
                                            n_cols=2, n_rows=5,
                                            question=(f'Check incorrect annotation'
                                                      f' for [{class_name[:-4]}] class'),
                                            filter_files=class_anno))

        elif type(self.output_item) == OutputVideoBbox:
            self.output_item.drawing_enabled = False

            out = BBoxVideoAnnotator(
                project_path=self.settings.project_path,
                input_item=self.input_item,
                output_item=self.output_item,
                annotation_file_path=create_step_annotations,
            )

        elif type(self.output_item) == OutputImageBbox:
            out = None
            # back to artificial bbox format ->
            di = {
                k: [
                    v['bbox'][0]['x'],
                    v['bbox'][0]['y'],
                    v['bbox'][0]['width'],
                    v['bbox'][0]['height']
                ] if v else [] for k, v in loaded_image_annotations.items()}

            captured_path = Path(self.settings.project_path) / "captured"

            # Save annotated images on disk
            for im, bbx in tqdm(di.items()):
                # use captured_path instead image_dir, keeping the folder structure
                old_im_path = Path(im)

                index = old_im_path.parts.index(self.input_item.dir) + 1
                new_im_path = captured_path.joinpath(*old_im_path.parts[index:])
                new_im_path.parent.mkdir(parents=True, exist_ok=True)

                _image = io.imread(im)
                if bbx:
                    rect = [bbx[1], bbx[0]]
                    rect_dimensions = [bbx[3], bbx[2]]

                    _image = draw_bbox(rect=rect, rect_dimensions=rect_dimensions,
                                       im=_image, black=True)

                io.imsave(str(new_im_path), _image, check_contrast=False)

            # Construct Capturer
            in_p = InputImage(image_dir="captured", image_width=150, image_height=150)
            out_p = OutputGridBox()
            anno_ = construct_annotation_path(self.settings.project_path,
                                              results_dir=f'{self.settings.result_dir}/missed')
            out = CaptureAnnotator(
                self.settings.project_path, input_item=in_p, output_item=out_p,
                annotation_file_path=anno_, n_cols=3,
                question="Check images with incorrect or empty bbox annotation")

        else:
            raise Exception(f"Improve is not supported for {self.output_item}")
        if isinstance(out, list):
            def update_step(anno):
                anno.app_state.annotation_step = AnnotatorStep.IMPROVE
                return anno
            out = [update_step(anno) for anno in out]
        else:
            out.app_state.annotation_step = AnnotatorStep.IMPROVE

        return out


# In[9]:


#hide
from nbdev.export import notebook2script
notebook2script()


# In[ ]:





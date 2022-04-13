#!/usr/bin/env python
# coding: utf-8

# In[1]:


# hide
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# default_exp bbox_video_annotator


# In[3]:


# hide
from nbdev import *


# In[4]:


#exporti
import attr
import warnings
from pubsub import pub
from attr import asdict
from copy import deepcopy
from typing import List, Dict, Optional, Callable, Iterable
from itertools import groupby
from ipywidgets import HBox, VBox, Layout, Button, Checkbox, ToggleButton
from ipyannotator.bbox_canvas import BBoxVideoCanvas
from ipyannotator.base import AppWidgetState
from ipyannotator.right_menu_widget import BBoxVideoList, BBoxVideoItem
from ipyannotator.mltypes import BboxCoordinate, BboxVideoCoordinate, Coordinate
from ipyannotator.bbox_annotator import (
    BBoxAnnotator, BBoxState, BBoxAnnotatorController,
    BBoxAnnotatorGUI, BBoxCoordinates, BBoxCanvasState
)
from ipyannotator.services.bbox_trajectory import BBoxTrajectory, TrajectoryStore


# In[5]:


# hide
import ipytest
import pytest
ipytest.autoconfig(raise_on_error=True)


# # Bbox video annotator

# In[6]:


#exporti
@attr.define
class BboxVideoCoordSelected:
    index: int
    frame: int
    bbox_video_coordinate: BboxVideoCoordinate


# In[7]:


#exporti
@attr.define
class BboxVideoHistory:
    labels: List[str] = []
    trajectories: TrajectoryStore = TrajectoryStore()
    bbox_coord: Optional[BboxVideoCoordinate] = None


# In[8]:


#exporti

class BBoxVideoState(BBoxState):
    trajectories: TrajectoryStore = TrajectoryStore()
    drawing_trajectory_enabled: bool = True
    right_menu_enabled: bool = True
    bbox_coords_selected: List[BboxVideoCoordSelected] = []
    merged_trajectories: List[str] = []


# In[9]:


#exporti

class BBoxVideoCoordinates(BBoxCoordinates):
    """Connects the BBoxList and the states"""

    def __init__(
        self,
        app_state: AppWidgetState,
        bbox_canvas_state: BBoxCanvasState,
        bbox_state: BBoxVideoState,
        drawing_enabled: bool,
        on_btn_select_clicked: Callable,
        on_label_changed: Callable,
        on_trajectory_enabled_clicked: Callable,
        on_btn_delete_clicked: Callable[[BboxVideoCoordinate], None]
    ):
        self.on_label_changed = on_label_changed
        super().__init__(
            app_state,
            bbox_canvas_state,
            bbox_state,
            on_btn_select_clicked
        )

        setattr(self, 'on_btn_delete_clicked', on_btn_delete_clicked)
        self._bbox_canvas_state = bbox_canvas_state
        self._bbox_state = bbox_state

        self.trajectory_enabled_checkbox = Checkbox(
            description='Enable object tracking',
            value=self._bbox_state.drawing_trajectory_enabled
        )

        if on_trajectory_enabled_clicked:
            self.trajectory_enabled_checkbox.observe(on_trajectory_enabled_clicked, names='value')

        self._bbox_state.unsubscribe('drawing_enabled')
        pub.unsubscribe(super()._sync_labels, f'{bbox_canvas_state.root_topic}.bbox_coords')
        pub.unsubscribe(super()._refresh_children, f'{app_state.root_topic}.index')

        self._init_bbox_list(self._bbox_state.drawing_enabled)

        bbox_canvas_state.subscribe(self._update_max_coord_input, 'image_scale')

        self.children = self._bbox_list.children

    def _init_bbox_list(self, drawing_enabled: bool):
        self._bbox_list = BBoxVideoList(
            btn_delete_enabled=drawing_enabled,
            on_label_changed=self.on_label_changed,
            on_btn_delete_clicked=self._on_btn_delete_clicked,
            on_btn_select_clicked=self.on_btn_select_clicked,
            classes=self._bbox_state.classes,
            on_checkbox_object_clicked=self._on_checkbox_object_clicked
        )

    def _refresh_children(self, index: int):
        self._render(
            self._bbox_canvas_state.bbox_coords,
            self._bbox_state.labels
        )

    def __getitem__(self, index: int) -> BBoxVideoItem:
        return self.children[1][index]

    def _render(self, bbox_coords: Iterable[Coordinate], labels: List[List[str]]):
        # "BBoxState" has no attribute "right_menu_enabled"
        if self._bbox_state.right_menu_enabled:  # type: ignore
            selected = []
            # Item "BboxCoordinate" of "Union[BboxCoordinate, Any]" has no attribute "id"
            all_frame_object_ids = [
                bbox_coord.id for bbox_coord in self._bbox_state.coords]  # type: ignore

            # "BBoxState" has no attribute "bbox_coords_selected"
            for coord in self._bbox_state.bbox_coords_selected:  # type: ignore
                if coord.bbox_video_coordinate.id in all_frame_object_ids:
                    selected.append(coord.index)

            # Unexpected keyword argument "labels" for "render_btn_list" of "BBoxList"
            self._bbox_list.render_btn_list(  # type: ignore
                bbox_video_coords=bbox_coords,
                classes=labels,
                labels=self._bbox_state.labels,
                selected=selected
            )

            self.children = VBox([
                self.trajectory_enabled_checkbox,
                self._bbox_list,
            ]).children

    def _on_btn_delete_clicked(self, index: int):
        bbox_coords = self._bbox_canvas_state.bbox_coords.copy()
        deleted = bbox_coords[index]
        del bbox_coords[index]
        self.remove_label(index)
        self._bbox_canvas_state.bbox_coords = bbox_coords
        self._remove_object_selected(index)
        self.on_btn_delete_clicked(deleted)  # type: ignore

    def _remove_object_selected(self, index: int):
        try:
            self._bbox_state.set_quietly(
                'bbox_coords_selected',
                # "BBoxState" has no attribute "bbox_coords_selected"
                list(filter(lambda x: x.index != index,
                            self._bbox_state.bbox_coords_selected))  # type: ignore
            )
        except Exception:
            warnings.warn("Couldn't unselect object")

    def _on_checkbox_object_clicked(self, change: dict, index: int,
                                    bbox_video_coord: BboxVideoCoordinate):
        if change['new']:
            # "BBoxState" has no attribute "bbox_coords_selected"
            self._bbox_state.bbox_coords_selected.append(  # type: ignore
                BboxVideoCoordSelected(
                    frame=self._app_state.index,
                    index=index,
                    bbox_video_coordinate=bbox_video_coord
                )
            )
        else:
            self._remove_object_selected(index)

    def clear(self):
        self._bbox_list.clear()
        self.children = []


# In[10]:


#exporti

class BBoxAnnotatorVideoGUI(BBoxAnnotatorGUI):
    def __init__(
        self,
        app_state: AppWidgetState,
        bbox_state: BBoxVideoState,
        on_label_changed: Callable,
        on_join_btn_clicked: Callable,
        on_bbox_drawn: Callable,
        drawing_enabled: bool = True,
        on_save_btn_clicked: Callable = None
    ):
        super().__init__(
            app_state=app_state,
            bbox_state=bbox_state,
            on_save_btn_clicked=on_save_btn_clicked,
            fit_canvas=False
        )

        self._app_state = app_state
        self._bbox_state = bbox_state
        self.on_bbox_drawn = on_bbox_drawn
        self.bbox_trajectory = BBoxTrajectory()
        self.history = BboxVideoHistory()
        self.on_label_changed = on_label_changed

        pub.unsubAll(f'{self._image_box.state.root_topic}.bbox_coords')

        self._image_box = BBoxVideoCanvas(
            *self._app_state.size,
            drawing_enabled=drawing_enabled
        )

        self.right_menu = BBoxVideoCoordinates(  # type: ignore
            app_state=self._app_state,
            # Argument "bbox_state" to "BBoxVideoCoordinates" has incompatible
            # type "BBoxState"; expected "BBoxVideoState"
            bbox_canvas_state=self._image_box.state,  # type: ignore
            bbox_state=self._bbox_state,  # type: ignore
            on_btn_select_clicked=self._highlight_bbox,
            on_btn_delete_clicked=self._remove_trajectory_history,
            on_label_changed=self.on_label_changed,
            drawing_enabled=drawing_enabled,
            on_trajectory_enabled_clicked=self.on_trajectory_enabled_clicked
        )

        self._navi.on_navi_clicked = self.view_idx_changed

        self.center = HBox(
            [self._image_box, self.right_menu],
            layout=Layout(
                display='flex',
                flex_flow='row'
            )
        )

        self._join_btn = Button(description="Join",
                                icon="compress",
                                layout=Layout(width='auto'))

        self._join_btn.on_click(on_join_btn_clicked)

        self.btn_right_menu_enabled = ToggleButton(
            description="Menu",
            tooltip="Disable right menu for a faster navigation experience.",
            icon="eye-slash",
            disabled=False,
            # Argument 1 to "render_right_menu" of "BBoxAnnotatorVideoGUI" has incompatible
            # type "List[BboxCoordinate]"; expected "List[BboxVideoCoordinate]"
            value=not self._bbox_state.right_menu_enabled,  # type: ignore
            layout=Layout(width="70px")
        )

        self.btn_right_menu_enabled.observe(
            self.on_right_menu_enabled_clicked,
            "value"
        )

        self.footer = HBox(
            [
                self._navi,
                self._save_btn,
                self._undo_btn,
                self._redo_btn,
                self._join_btn,
                self.btn_right_menu_enabled,
            ],
            layout=Layout(
                display='flex',
                flex_flow='row wrap',
                align_items='center'
            )
        )

        self._app_state.index = 0

        self._image_box.state.subscribe(self.render_right_menu, 'bbox_coords')
        if self._image_box.state.bbox_coords:
            # Argument 1 to "render_right_menu" of "BBoxAnnotatorVideoGUI" has incompatible type
            # "List[BboxCoordinate]"; expected "List[BboxVideoCoordinate]"
            self.render_right_menu(self._image_box.state.bbox_coords)  # type: ignore

    def _remove_trajectory_history(self, bbox_video_coord: BboxVideoCoordinate):
        trajectories = deepcopy(self.history.trajectories)
        del trajectories[bbox_video_coord.id]
        self._bbox_state.trajectories = trajectories

    def on_trajectory_enabled_clicked(self, change: dict):
        # "BBoxState" has no attribute "drawing_trajectory_enabled"
        aux = not self._bbox_state.drawing_trajectory_enabled  # type: ignore
        self._bbox_state.drawing_trajectory_enabled = aux
        self._bbox_state.trajectories.clear()  # type: ignore
        self.refresh_gui()

    def on_right_menu_enabled_clicked(self, change: dict):
        # "BBoxState" has no attribute "right_menu_enabled"
        menu_enabled = self._bbox_state.right_menu_enabled  # type: ignore
        self._bbox_state.right_menu_enabled = not menu_enabled
        self.refresh_gui()

    def refresh_gui(self):
        self._image_box.clear_layer(-1)
        self.clear_right_menu()
        self.render_right_menu(self._image_box.state.bbox_coords)
        # refreshing bbox_coords on canvas
        self._image_box.state.bbox_coords = self._image_box.state.bbox_coords

    def render_right_menu(self, bbox_coords: List[BboxVideoCoordinate]):
        if self.on_bbox_drawn:
            self.on_bbox_drawn(bbox_coords)
            self.right_menu._render(bbox_coords, self._bbox_state.labels)
        # "BBoxState" has no attribute "drawing_trajectory_enabled"
        if self._bbox_state.drawing_trajectory_enabled:  # type: ignore
            self.draw_trajectory(bbox_coords)

    def _redo_clicked(self, event: dict):
        if self.history.labels is not None:
            self._bbox_state.labels.append(self.history.labels)
            self.history.labels = []
        if self.history.trajectories:
            for k, v in self.history.trajectories.items():
                # "BBoxState" has no attribute "trajectory"
                self._bbox_state.trajectories[k] = v  # type: ignore
            self.history.trajectories.clear()
        if self.history.bbox_coord:
            tmp_bbox_coords = deepcopy(self._image_box.state.bbox_coords)
            tmp_bbox_coords.append(self.history.bbox_coord)
            self._image_box.state.bbox_coords = tmp_bbox_coords
            self.history.bbox_coord = None

    def _undo_clicked(self, event: dict):
        if self._bbox_state.labels:
            tmp_labels = deepcopy(self._bbox_state.labels)
            self.history.labels = tmp_labels.pop()
            self._bbox_state.labels = tmp_labels

        tmp_bbox_coords = None
        if self._image_box.state.bbox_coords:
            tmp_bbox_coords = deepcopy(self._image_box.state.bbox_coords)
            # expression has type "BboxCoordinate", variable has type
            # "Optional[BboxVideoCoordinate]"
            self.history.bbox_coord = tmp_bbox_coords.pop()  # type: ignore
            self._image_box.state.bbox_coords = tmp_bbox_coords

        if (self.history.bbox_coord and
                self.history.bbox_coord.id in self._bbox_state.trajectories):  # type: ignore
            tmp_trajectory = deepcopy(self._bbox_state.trajectories)  # type: ignore
            self.history.trajectories.update({
                self.history.bbox_coord.id: tmp_trajectory[self.history.bbox_coord.id]
            })
            del tmp_trajectory[self.history.bbox_coord.id]
            self._bbox_state.trajectories = tmp_trajectory
        self.refresh_gui()

    def view_idx_changed(self, index: int):
        # store the last bbox drawn before index update
        self._bbox_state.set_quietly(
            'coords',
            self._image_box._state.bbox_coords
        )
        self._app_state.index = index

    def clear_right_menu(self):
        self.right_menu.clear()

    def draw_trajectory(self, bbox_coords: List[BboxVideoCoordinate]):
        """Draw trajectory checking if object lives on bbox canvas"""
        coords = self._bbox_video_to_trajectory(bbox_coords)
        coord_ids = [i.id for i in self._image_box.state.bbox_coords]  # type: ignore
        for obj_id, value in coords.items():
            if len(value) > 1 and obj_id in coord_ids:
                self.bbox_trajectory.draw_trajectory(
                    self._image_box.multi_canvas[-1],
                    value,
                    self._image_box.state.image_scale
                )

    def _bbox_video_to_trajectory(
            self, coords: List[BboxVideoCoordinate]) -> Dict[str, List[BboxCoordinate]]:
        """Group objects and stores (in memory) a list of bbox coordinates"""
        def sort(k):
            return k['id']

        coords = [asdict(c) for c in coords]  # type: ignore
        for k, v in groupby(sorted(coords, key=sort), sort):
            value = list(v)
            for i in value:
                bbox_coord = BboxCoordinate(*list(i.values())[:4])  # type: ignore
                try:
                    if bbox_coord not in self._bbox_state.trajectories[k]:  # type: ignore
                        self._bbox_state.trajectories[k].append(bbox_coord)  # type: ignore
                except Exception:
                    self._bbox_state.trajectories[k] = [bbox_coord]  # type: ignore

        return self._bbox_state.trajectories  # type: ignore

    def on_client_ready(self, callback):
        self._image_box.observe_client_ready(callback)


# In[11]:


#exporti

class BBoxVideoAnnotatorController(BBoxAnnotatorController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _update_coords(self, index: int):  # from annotations
        image_path = str(self._storage.images[index])
        coords = self._storage.get(image_path) or {}
        self._bbox_state.labels = coords.get('labels', [])
        self._bbox_state.coords = [BboxVideoCoordinate(**c) for c in coords.get('bbox', [])]

    def _save_annotations(self, index: int, *args, **kwargs):
        image_path = str(self._storage.images[index])
        self._storage[image_path] = {
            # Item "None" of "Optional[List[BboxCoordinate]]" has no attribute "__iter__"
            'bbox': [asdict(bbox) for bbox in self._bbox_state.coords],  # type: ignore
            'labels': self._bbox_state.labels,
        }
        self._storage.save()

    def update_storage_labels(self, change: dict, index: int):
        """Receive an object label update and updates all
        object's labels that share the same id"""
        self._bbox_state.labels[index] = [change['new']]
        # Value of type "Optional[List[BboxCoordinate]]" is not indexable
        bbox_coord_id = self._bbox_state.coords[index].id  # type: ignore
        for image_path, bbox_or_labels in self._storage.items():
            if not bbox_or_labels or 'bbox' not in bbox_or_labels:
                break
            for i, bbox_coord in enumerate(bbox_or_labels['bbox']):
                if bbox_coord['id'] == bbox_coord_id:
                    self._storage[image_path]['labels'][i] = [change['new']]

    def update_storage_id(self, merged_ids: List[str]):
        """Update objects id's once one of them are merged.
        Returns the merged ids trajectory."""
        merge_id = "-".join(merged_ids)

        for image_path, bbox_or_labels in self._storage.items():
            if not bbox_or_labels or 'bbox' not in bbox_or_labels:
                break
            for bbox_coord in bbox_or_labels['bbox']:
                if bbox_coord['id'] in merged_ids:
                    bbox_coord['id'] = merge_id

    def _idx_changed(self, index: int):
        """
        On index change save an old state and update
        current image path and bbox coordinates for
        visualisation
        """
        self._save_annotations(self._last_index)
        self._update_im(index)
        self._update_coords(index)
        self._last_index = index

    def sync_labels(self, bbox_coords: List[BboxVideoCoordinate]):
        """Update labels according to the past frame labels
        or add an empty label to the bbox state. It also avoid to
        add empty labels if its length is the same as the bbox_coords"""
        num_classes = len(self._bbox_state.labels)

        for i, _ in enumerate(bbox_coords[num_classes:], num_classes):
            try:
                image_path = self._storage.images[self._app_state.index - 1]
                coords = self._storage.get(str(image_path)) or {}
                self._bbox_state.labels.append(
                    # Value of type "Optional[Any]" is not indexable
                    coords.get('labels')[i]  # type: ignore
                )
            except Exception:
                self._bbox_state.labels.append([])

    def handle_client_ready(self):
        self._idx_changed(self._last_index)


# In[12]:


#export

class BBoxVideoAnnotator(BBoxAnnotator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**kwargs, 'render_previous_coords': False})
        pub.unsubscribe(self.controller._idx_changed, f'{self.app_state.root_topic}.index')
        pub.unsubAll(f'{self.app_state.root_topic}.index')
        state_params = {**self.bbox_state.dict()}
        state_params.pop('_uuid', [])
        state_params.pop('event_map', [])
        self.bbox_state = BBoxVideoState(
            uuid=self.bbox_state._uuid,
            **state_params
        )

        self.controller = BBoxVideoAnnotatorController(
            app_state=self.app_state,
            bbox_state=self.bbox_state,
            storage=self.storage
        )

        self.view = BBoxAnnotatorVideoGUI(
            app_state=self.app_state,
            bbox_state=self.bbox_state,
            on_save_btn_clicked=self.on_save_btn_clicked,
            drawing_enabled=self._output_item.drawing_enabled,
            on_label_changed=self.update_labels,
            on_join_btn_clicked=self.merge_tracks_selected,
            on_bbox_drawn=self.controller.sync_labels
        )

        self.view.on_client_ready(self.controller.handle_client_ready)

    def update_labels(self, change: dict, index: int):
        """Saves bbox_canvas_state coordinates data
        on bbox_state and save all on storage."""
        self.bbox_state.set_quietly('coords', self.view._image_box._state.bbox_coords)
        # "BBoxAnnotatorController" has no attribute "update_storage_labels"
        self.controller.update_storage_labels(change, index)  # type: ignore

    def on_save_btn_clicked(self, bbox_coords: List[BboxVideoCoordinate]):
        self.controller.save_current_annotations(bbox_coords)  # type: ignore

    def _update_state_id(self, merged_ids: List[str], bbox_coords: List[BboxVideoCoordinate]):
        merged_id = "-".join(merged_ids)

        for i, coord in enumerate(bbox_coords):
            if merged_id and bbox_coords[i].id in merged_ids:
                bbox_coords[i].id = merged_id

    def merge_tracks_selected(self, change: Dict):
        # "BBoxState" has no attribute "bbox_coords_selected"
        selecteds = self.bbox_state.bbox_coords_selected  # type: ignore

        if selecteds:
            merged_ids = [selected.bbox_video_coordinate.id for selected in selecteds]

            self._update_state_id(
                # Argument "bbox_coords" to "_update_state_id" of "BBoxVideoAnnotator" has
                #incompatible type "List[BboxCoordinate]"; expected "List[BboxVideoCoordinate]"
                merged_ids=merged_ids,  # type: ignore
                bbox_coords=self.view._image_box.state.bbox_coords  # type: ignore
            )

            # "BBoxAnnotatorController" has no attribute "update_storage_id"
            self.controller.update_storage_id(  # type: ignore
                merged_ids=merged_ids
            )

            # merge trajectories
            key = "-".join(merged_ids)
            # "BBoxState" has no attribute "trajectory"
            tmp_trajectories = deepcopy(self.bbox_state.trajectories)  # type: ignore
            for id, bbx in tmp_trajectories.items():
                if id in merged_ids:
                    try:
                        self.bbox_state.trajectories[key] += bbx  # type: ignore
                    except Exception:
                        self.bbox_state.trajectories[key] = bbx  # type: ignore

                    if id in self.bbox_state.trajectories:  # type: ignore
                        # delete old trajectory id
                        del self.bbox_state.trajectories[id]  # type: ignore

            # finished merge cleans selected bbox coords
            self.bbox_state.bbox_coords_selected = []

            self.view.clear_right_menu()
            self.view.render_right_menu(self.view._image_box.state.bbox_coords)


# In[13]:


#hide
from ipyannotator.mltypes import InputImage, OutputVideoBbox

in_p = InputImage(image_dir='pics', image_width=200, image_height=200)
out_p = OutputVideoBbox(classes=['Label 01', 'Label 02'])


# In[14]:


#hide
get_ipython().system(' rm -rf ../data/projects/bbox_video/results')


# In[15]:


# hide
from ipyannotator.storage import construct_annotation_path
from ipyannotator.datasets.generators import create_mot_ds
from pathlib import Path

project_path = Path('../data/projects/bbox_video')

anno_file_path = construct_annotation_path(project_path)

create_mot_ds(project_path, 'pics', 4)


# In[16]:


# hide
bb = BBoxVideoAnnotator(
    project_path=Path(project_path),
    input_item=in_p,
    output_item=out_p,
    annotation_file_path=anno_file_path
)

bb


# In[17]:


from ipyannotator.storage import construct_annotation_path


def delete_result_file():
    get_ipython().system(' rm -rf ../data/projects/bbox_video/results')


@pytest.fixture
def bbox_video_fixture():
    delete_result_file()
    project_path = Path('../data/projects/bbox_video')
    anno_file_path = construct_annotation_path(project_path)
    in_p = InputImage(image_dir='pics', image_width=640, image_height=400)
    out_p = OutputVideoBbox(classes=['Label 01', 'Label 02'])

    return BBoxVideoAnnotator(
        project_path=project_path,
        input_item=in_p,
        output_item=out_p,
        annotation_file_path=anno_file_path
    )


@pytest.fixture
def trajectory_fixture(bbox_video_fixture):
    fixture = bbox_video_fixture

    coords = [
        BboxVideoCoordinate(x=371, y=405, width=81, height=249, id='0'),
        BboxVideoCoordinate(x=677, y=186, width=78, height=171, id='1')
    ]

    fixture.view._image_box._state.bbox_coords = coords

    fixture.view._navi._next_btn.click()

    coords = [
        BboxVideoCoordinate(x=374, y=408, width=90, height=189, id='0'),
        BboxVideoCoordinate(x=686, y=189, width=75, height=135, id='1')
    ]

    fixture.view._image_box._state.bbox_coords = coords

    fixture.view._navi._next_btn.click()

    fixture.view._navi._prev_btn.click()
    fixture.view._navi._prev_btn.click()

    assert fixture.app_state.index == 0
    assert len(fixture.view.right_menu.children) == 2

    return fixture


# In[18]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_keeps_checkboxes_alive_on_user_navigation(trajectory_fixture):\n    # select objects to join\n    trajectory_fixture.view.right_menu[0].object_checkbox.value = True\n    assert trajectory_fixture.view.right_menu[1].object_checkbox.value == False\n\n    #next page\n    trajectory_fixture.view._navi._next_btn.click()\n    assert trajectory_fixture.view.right_menu[0].object_checkbox.value == True\n    assert trajectory_fixture.view.right_menu[1].object_checkbox.value == False\n    \n    #previous page\n    trajectory_fixture.view._navi._prev_btn.click()\n    assert trajectory_fixture.view.right_menu[0].object_checkbox.value == True\n    assert trajectory_fixture.view.right_menu[1].object_checkbox.value == False')


# In[19]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_clear_right_menu_on_join_btn_click(bbox_video_fixture):\n    for i in range(4):\n        if not bbox_video_fixture.bbox_state.coords:\n            coords=[\n                BboxVideoCoordinate(x=371+i*10, y=405+i*10, width=81, height=249, id='0'), \n                BboxVideoCoordinate(x=677+i*10, y=186+i*10, width=78, height=171, id='1')\n            ]\n\n            bbox_video_fixture.bbox_state.coords = coords\n        \n        # alternate selects between the first and second element from the right menu\n        elem = 1 if i % 2 != 0 else 0\n        bbox_video_fixture.view.right_menu[elem].object_checkbox.value = True\n        bbox_video_fixture.view._navi._next_btn.click()\n\n    #click on join\n    bbox_video_fixture.view._join_btn.click()\n\n    assert bbox_video_fixture.bbox_state.bbox_coords_selected == []\n    \n    for i in range(4):\n        # select the first element from the right menu\n        elem = 1 if i % 2 != 0 else 0\n        assert bbox_video_fixture.view.right_menu[elem].object_checkbox.description == '0-1'")


# In[20]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_update_state_and_storage_on_label_change(trajectory_fixture):\n    cls = out_p.classes[0]\n#     bbox_video_fixture.view.right_menu[elem].object_checkbox.description == '0-1-0-1'\n    trajectory_fixture.view.right_menu[0].dropdown_classes.value = cls\n    assert trajectory_fixture.view.right_menu[1].dropdown_classes.value == None\n    assert trajectory_fixture.bbox_state.labels == [[cls], []]\n    trajectory_fixture.view._navi._next_btn.click()\n    assert trajectory_fixture.view.right_menu[0].dropdown_classes.value == cls\n    assert trajectory_fixture.view.right_menu[1].dropdown_classes.value == None")


# In[21]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_can_draw_object_on_canvas_and_update_labels(bbox_video_fixture):\n    bbox_video_fixture.view._navi._next_btn.click()\n    coords = [\n        BboxVideoCoordinate(x=371, y=405, width=81, height=249, id='0'),\n        BboxVideoCoordinate(x=211, y=325, width=97, height=97, id='1')\n    ]\n    bbox_video_fixture.view._image_box._state.bbox_coords = coords\n    bbox_video_fixture.view._navi._prev_btn.click()\n    coords = [\n        BboxVideoCoordinate(x=361, y=395, width=71, height=239, id='0'),\n    ]\n    bbox_video_fixture.view._image_box._state.bbox_coords = coords\n    bbox_video_fixture.view.right_menu[0].dropdown_classes.value = out_p.classes[0]\n    for k,v in bbox_video_fixture.storage.items():\n        if v is not None:\n            assert v['labels'][0][0] == out_p.classes[0]")


# In[22]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_can_undo_trajectory_on_undo_click(trajectory_fixture):\n    trajectory = {\n        '0': [\n            BboxCoordinate(x=371, y=405, width=81, height=249), \n            BboxCoordinate(x=374, y=408, width=90, height=189),\n        ], \n        '1': [\n            BboxCoordinate(x=677, y=186, width=78, height=171), \n            BboxCoordinate(x=686, y=189, width=75, height=135),\n        ]\n    }\n    assert trajectory_fixture.bbox_state.trajectories == trajectory\n    trajectory_fixture.view._undo_btn.click()\n    assert trajectory_fixture.bbox_state.trajectories == {\n        '0': [\n            BboxCoordinate(x=371, y=405, width=81, height=249), \n            BboxCoordinate(x=374, y=408, width=90, height=189),\n        ]\n    }\n    assert trajectory_fixture.view.history.trajectories == {\n        '1': [\n            BboxCoordinate(x=677, y=186, width=78, height=171), \n            BboxCoordinate(x=686, y=189, width=75, height=135),\n        ]\n    }")


# In[23]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_can_undo_labels_on_undo_click(trajectory_fixture):\n    labels = [[], []]\n    assert trajectory_fixture.bbox_state.labels == labels\n    trajectory_fixture.view._undo_btn.click()\n    assert trajectory_fixture.bbox_state.labels == [[]]\n    assert trajectory_fixture.view.history.labels == []')


# In[24]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_can_redo_trajectory_on_redo_click(trajectory_fixture):\n    trajectory = {\n        '0': [\n            BboxCoordinate(x=371, y=405, width=81, height=249), \n            BboxCoordinate(x=374, y=408, width=90, height=189),\n        ], \n        '1': [\n            BboxCoordinate(x=677, y=186, width=78, height=171), \n            BboxCoordinate(x=686, y=189, width=75, height=135),\n        ]\n    }\n    trajectory_fixture.view._undo_btn.click()\n    trajectory_fixture.view._redo_btn.click()\n    assert trajectory_fixture.view.history.trajectories == {}")


# In[25]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_can_redo_labels_on_redo_click(trajectory_fixture):\n    labels = [[], []]\n    assert trajectory_fixture.bbox_state.labels == labels\n    trajectory_fixture.view._undo_btn.click()\n    assert trajectory_fixture.bbox_state.labels == [[]]\n    trajectory_fixture.view._redo_btn.click()\n    assert trajectory_fixture.bbox_state.labels == labels')


# In[26]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_change_only_selected_checkboxes_on_join_trajectories(trajectory_fixture):\n    trajectory_fixture.view.right_menu[1].object_checkbox.value = True\n    \n    assert len(trajectory_fixture.bbox_state.bbox_coords_selected) == 1\n    \n    trajectory_fixture.view._join_btn.click()\n    \n    inp1 = trajectory_fixture.view.right_menu[0].object_checkbox\n    inp2 = trajectory_fixture.view.right_menu[1].object_checkbox\n    \n    assert inp1.description != inp2.description')


# In[27]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_merge_storage_trajectories_on_join_trajectories(trajectory_fixture):\n    assert trajectory_fixture.app_state.index == 0\n    \n    second_frame_coords = None\n    for i in range(3):\n        if i == 0:\n            # select object 0 on the first frame\n            trajectory_fixture.view.right_menu[0].object_checkbox.value = True\n        elif i == 1:\n            second_frame_coords = trajectory_fixture.view._image_box._state.bbox_coords[0]\n        \n        trajectory_fixture.view._navi._next_btn.click()\n\n    coords = [\n        BboxVideoCoordinate(x=400, y=455, width=111, height=298, id='0'), \n        BboxVideoCoordinate(x=722, y=210, width=90, height=190, id='1')\n    ]\n    trajectory_fixture.view._image_box._state.bbox_coords = coords\n\n    # select object 0 on the third (current) frame\n    trajectory_fixture.view.right_menu[0].object_checkbox.value = True\n    # join the first and third frame\n    trajectory_fixture.view._join_btn.click()\n    # makes sure that the second frame is in the trajectory\n    exists = False\n    \n    for k, trajectory in trajectory_fixture.bbox_state.trajectories.items():\n        if second_frame_coords.bbox_coord() in trajectory:\n            exists = True\n    \n    assert exists == True")


# In[28]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_render_object_label_on_drawing(bbox_video_fixture):\n    bbox_video_fixture.view._image_box._state.bbox_coords = [\n        BboxVideoCoordinate(x=361, y=395, width=71, height=239, id='0'),\n    ]\n    cls = out_p.classes[0]\n    bbox_video_fixture.view.right_menu[0].dropdown_classes.value = cls\n    bbox_video_fixture.view._navi._next_btn.click()\n    assert bbox_video_fixture.view._image_box._state.bbox_coords == []\n    bbox_video_fixture.view._image_box._state.bbox_coords = [\n        BboxVideoCoordinate(x=371, y=405, width=81, height=249, id='0'),\n    ]\n    assert bbox_video_fixture.view.right_menu[0].dropdown_classes.value == cls")


# In[29]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_can_disable_drawing_trajectory(bbox_video_fixture):\n    bbox_video_fixture.view.right_menu.trajectory_enabled_checkbox.value = False\n    bbox_video_fixture.view._image_box._state.bbox_coords = [\n        BboxVideoCoordinate(x=361, y=395, width=71, height=239, id='0'),\n    ]\n    bbox_video_fixture.view._navi._next_btn.click()\n    bbox_video_fixture.view._image_box._state.bbox_coords = [\n        BboxVideoCoordinate(x=371, y=405, width=81, height=249, id='0'),\n    ]\n    assert bbox_video_fixture.bbox_state.trajectories == {}")


# In[30]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_can_navigate_on_disabled_drawing_trajectory(trajectory_fixture):\n    trajectory_fixture.view._navi._next_btn.click()\n    trajectory_fixture.view.right_menu.trajectory_enabled_checkbox.value = False\n    trajectory_fixture.view._navi._prev_btn.click()\n    assert trajectory_fixture.bbox_state.trajectories == {}\n    trajectory_fixture.view.right_menu.trajectory_enabled_checkbox.value = True\n    trajectory_fixture.view._navi._next_btn.click()\n    assert trajectory_fixture.bbox_state.trajectories != {}')


# In[31]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_sticky_object_selection_across_frames(trajectory_fixture):\n    trajectory_fixture.view.right_menu[0].object_checkbox.value = True\n    trajectory_fixture.view._navi._next_btn.click()\n    assert trajectory_fixture.view.right_menu[0].object_checkbox.value == True')


# In[32]:


get_ipython().run_cell_magic('ipytest', '', 'def test_it_render_right_menu_on_init(trajectory_fixture, bbox_video_fixture):\n    assert len(bbox_video_fixture.storage) > 0\n    assert len(bbox_video_fixture.view.right_menu.children[1].children) > 0')


# In[33]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_can_delete_element(bbox_video_fixture):\n    bbox_video_fixture.view._image_box._state.bbox_coords = [\n        BboxVideoCoordinate(x=361, y=395, width=71, height=239, id='0'),\n         BboxVideoCoordinate(x=381, y=400, width=90, height=250, id='0'),\n         BboxVideoCoordinate(x=391, y=410, width=100, height=260, id='0'),\n    ]\n    bbox_video_fixture.view.right_menu[1].btn_delete.click()\n    assert bbox_video_fixture.view.right_menu[1].btn_delete.value == 1")


# In[34]:


get_ipython().run_cell_magic('ipytest', '', "def test_it_remove_trajectory_on_btn_delete_clicked(trajectory_fixture):\n    del_coord = trajectory_fixture.bbox_state.coords[0]\n    coords = trajectory_fixture.bbox_state.coords\n    trajectory_fixture.view.right_menu[0].btn_delete.click()\n    assert del_coord not in trajectory_fixture.view.right_menu._bbox_canvas_state.bbox_coords\n    assert '0' not in dict(trajectory_fixture.bbox_state.trajectories)")


# In[35]:


#hide
from nbdev.export import notebook2script
notebook2script()


# In[ ]:





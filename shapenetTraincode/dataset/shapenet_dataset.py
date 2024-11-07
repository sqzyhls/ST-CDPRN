from typing import Dict, List, Optional, Tuple
from pathlib import Path
from os import path
import warnings
import json
import numpy as np
import glob
import h5py
import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os
import torch
from PIL import Image
#from .utils import torch_center_and_normalize, sort_jointly, load_obj, load_text, torch_direction_vector
import collections
# from torch_geometric.io import read_off, read_obj
import trimesh
import math
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures
import imageio
from torch import nn


def torch_center_and_normalize(points,p="inf"):
    """
    a helper pytorch function that normalize and center 3D points clouds 
    """
    N = points.shape[0]
    center = points.mean(0)
    if p != "fro" and p!= "no":
        scale = torch.max(torch.norm(points - center, p=float(p),dim=1))
    elif p=="fro" :
        scale = torch.norm(points - center, p=p )
    elif p=="no":
        scale = 1.0
    points = points - center.expand(N, 3)
    points = points * (1.0 / float(scale))
    return points
def sort_jointly(list_of_arrays, dim=0):
    """
    sort all the arrays in `list_of_arrays` according to the sorting of the array `array list_of_arrays`[dim]
    """
    def swapPositions(mylsit, pos1, pos2):
        mylsit[pos1], mylsit[pos2] = mylsit[pos2], mylsit[pos1]
        return mylsit
    sorted_tuples = sorted(zip(*swapPositions(list_of_arrays, 0, dim)))
    combined_sorted = list(zip(*sorted_tuples))
    return [list(ii) for ii in swapPositions(combined_sorted, 0, dim)]
class ShapeNetBase(torch.utils.data.Dataset):
    """
    'ShapeNetBase' implements a base Dataset for ShapeNet and R2N2 with helper methods.
    It is not intended to be used on its own as a Dataset for a Dataloader. Both __init__
    and __getitem__ need to be implemented.
    """

    def __init__(self):
        """
        Set up lists of synset_ids and model_ids.
        """
        self.synset_ids = []
        self.model_ids = []
        self.synset_inv = {}
        self.synset_start_idxs = {}
        self.synset_num_models = {}
        self.shapenet_dir = ""
        self.model_dir = "model.obj"
        self.load_textures = True
        self.texture_resolution = 4

    def __len__(self):
        """
        Return number of total models in the loaded dataset.
        """
        return len(self.model_ids)

    def __getitem__(self, idx) -> Dict:
        """
        Read a model by the given index. Need to be implemented for every child class
        of ShapeNetBase.

        Args:
            idx: The idx of the model to be retrieved in the dataset.

        Returns:
            dictionary containing information about the model.
        """
        raise NotImplementedError(
            "__getitem__ should be implemented in the child class of ShapeNetBase"
        )

    def _get_item_ids(self, idx) -> Dict:
        """
        Read a model by the given index.

        Args:
            idx: The idx of the model to be retrieved in the dataset.

        Returns:
            dictionary with following keys:
            - synset_id (str): synset id
            - model_id (str): model id
        """
        model = {}
        model["synset_id"] = self.synset_ids[idx]
        model["model_id"] = self.model_ids[idx]
        return model

    def _load_mesh(self, model_path) -> Tuple:
        from pytorch3d.io import load_obj

        verts, faces, aux = load_obj(
            model_path,
            create_texture_atlas=self.load_textures,
            load_textures=self.load_textures,
            texture_atlas_size=self.texture_resolution,
        )
        if self.load_textures:
            textures = aux.texture_atlas
            # Some meshes don't have textures. In this case
            # create a white texture map
        else:
            textures = verts.new_ones(
                faces.verts_idx.shape[0],
                self.texture_resolution,
                self.texture_resolution,
                3,
            )

        return verts, faces.verts_idx, textures

    

class ShapeNetCore(ShapeNetBase):
    """
    This class loads ShapeNetCore from a given directory into a Dataset object.
    ShapeNetCore is a subset of the ShapeNet dataset and can be downloaded from
    https://www.shapenet.org/.
    """

    def __init__(
        self,
        data_dir,
        split,
        nb_points,
        synsets=None,
        version: int = 2,
        load_textures: bool = False,
        texture_resolution: int = 4,
        dset_norm: str = "inf",
        simplified_mesh=False,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    ):
        """
        Store each object's synset id and models id from data_dir.
        Args:
            data_dir: Path to ShapeNetCore data.
            synsets: List of synset categories to load from ShapeNetCore in the form of
                synset offsets or labels. A combination of both is also accepted.
                When no category is specified, all categories in data_dir are loaded.
            version: (int) version of ShapeNetCore data in data_dir, 1 or 2.
                Default is set to be 1. Version 1 has 57 categories and verions 2 has 55
                categories.
                Note: version 1 has two categories 02858304(boat) and 02992529(cellphone)
                that are hyponyms of categories 04530566(watercraft) and 04401088(telephone)
                respectively. You can combine the categories manually if needed.
                Version 2 doesn't have 02858304(boat) or 02834778(bicycle) compared to
                version 1.
            load_textures: Boolean indicating whether textures should loaded for the model.
                Textures will be of type TexturesAtlas i.e. a texture map per face.
            texture_resolution: Int specifying the resolution of the texture map per face
                created using the textures in the obj file. A
                (texture_resolution, texture_resolution, 3) map is created per face.
        """
        super().__init__()
        self.shapenet_dir = data_dir
        self.nb_points = nb_points
        self.load_textures = load_textures
        self.texture_resolution = texture_resolution
        self.dset_norm = dset_norm
        self.split = split
        self.simplified_mesh = simplified_mesh

        if version not in [1, 2]:
            raise ValueError("Version number must be either 1 or 2.")
        self.model_dir = "model.obj" if version == 1 else "models/model_normalized.obj"
        if self.simplified_mesh:
            self.model_dir = "models/model_normalized_SMPLER.obj"
        #splits = pd.read_csv(os.path.join(
        #    self.shapenet_dir, "shapenet_split.csv"), sep=",", dtype=str)

        # Synset dictionary mapping synset offsets to corresponding labels.
        dict_file = "shapenet_synset_dict_v%d.json" % version
        with open(path.join(os.path.abspath(pytorch3d.datasets.shapenet.__path__[0]), dict_file), "r") as read_dict:
            self.synset_dict = json.load(read_dict)
        # Inverse dicitonary mapping synset labels to corresponding offsets.
        self.synset_inv = {label: offset for offset,
                           label in self.synset_dict.items()}

        # If categories are specified, check if each category is in the form of either
        # synset offset or synset label, and if the category exists in the given directory.
        if synsets is not None:
            # Set of categories to load in the form of synset offsets.
            synset_set = set()
            for synset in synsets:
                if (synset in self.synset_dict.keys()) and (
                    path.isdir(path.join(data_dir, synset))
                ):
                    synset_set.add(synset)
                elif (synset in self.synset_inv.keys()) and (
                    (path.isdir(path.join(data_dir, self.synset_inv[synset])))
                ):
                    synset_set.add(self.synset_inv[synset])
                else:
                    msg = (
                        "Synset category %s either not part of ShapeNetCore dataset "
                        "or cannot be found in %s."
                    ) % (synset, data_dir)
                    warnings.warn(msg)
        # If no category is given, load every category in the given directory.
        # Ignore synset folders not included in the official mapping.
        else:
            synset_set = {
                synset
                for synset in os.listdir(data_dir)
                if path.isdir(path.join(data_dir, synset))
                and synset in self.synset_dict
            }

        # Check if there are any categories in the official mapping that are not loaded.
        # Update self.synset_inv so that it only includes the loaded categories.
        synset_not_present = set(
            self.synset_dict.keys()).difference(synset_set)
        [self.synset_inv.pop(self.synset_dict[synset])
         for synset in synset_not_present]

        if len(synset_not_present) > 0:
            msg = (
                "The following categories are included in ShapeNetCore ver.%d's "
                "official mapping but not found in the dataset location %s: %s"
                ""
            ) % (version, data_dir, ", ".join(synset_not_present))
            warnings.warn(msg)

        # Extract model_id of each object from directory names.
        # Each grandchildren directory of data_dir contains an object, and the name
        # of the directory is the object's model_id.
        for synset in synset_set:
            self.synset_start_idxs[synset] = len(self.synset_ids)
            for model in os.listdir(path.join(data_dir, synset)):
                if not path.exists(path.join(data_dir, synset, model, self.model_dir)):
                    msg = (
                        "Object file not found in the model directory %s "
                        "under synset directory %s."
                    ) % (model, synset)
                    # warnings.warn(msg)
                    continue
                self.synset_ids.append(synset)
                self.model_ids.append(model)
            model_count = len(self.synset_ids) - self.synset_start_idxs[synset]
            self.synset_num_models[synset] = model_count
        # !!
        self.model_ids, self.synset_ids = sort_jointly([self.model_ids, self.synset_ids], dim=0)

        split_model_ids,split_synset_ids = [] , [] 
        if not os.path.exists(os.path.join(self.shapenet_dir, "shapenet_split.csv")):
            out_dict = {}
            for ii, model in enumerate(self.model_ids):
                rd = np.random.random()
                if rd < train_ratio:
                    out_dict[self.synset_ids[ii]] = ['train']
                    if split == 'train':
                        split_model_ids.append(model)
                        split_synset_ids.append(self.synset_ids[ii])
                        
                elif rd < train_ratio+val_ratio:
                    out_dict[self.synset_ids[ii]] = ['val']
                    if split == 'val':
                        split_model_ids.append(model)
                        split_synset_ids.append(self.synset_ids[ii])
                else:
                    out_dict[self.synset_ids[ii]] = ['test']
                    if split == 'test':
                        split_model_ids.append(model)
                        split_synset_ids.append(self.synset_ids[ii])
            out = pd.DataFrame.from_dict(out_dict)
            out.to_csv(os.path.join(self.shapenet_dir, "shapenet_split.csv"))
        else:
            split_data = pd.read_csv(os.path.join(self.shapenet_dir, "shapenet_split.csv"))
            for ii, model in enumerate(self.model_ids):
                if split_data[self.synset_ids[ii]][0] == split:
                    split_model_ids.append(model)
                    split_synset_ids.append(self.synset_ids[ii])  
        self.model_ids = split_model_ids
        self.synset_ids = split_synset_ids
        self.classes = sorted(list(self.synset_inv.keys()))
        self.label_by_number = {k: v for v, k in enumerate(self.classes)}
        self.dataset = self.model_ids
        # adding train/val/test splits of the data 
        """
        split_model_ids,split_synset_ids = [] , [] 
        for ii, model in enumerate(self.model_ids):
            found = splits[splits.modelId.isin([model])]["split"]
            if len(found) > 0:
                if found.item() in self.split:
                    split_model_ids.append(model)
                    split_synset_ids.append(self.synset_ids[ii])
        self.model_ids = split_model_ids
        self.synset_ids = split_synset_ids
        """
        # self.model_ids = list(splits[splits.modelId.isin(self.model_ids)][splits.split.isin([self.split])]["modelId"])
        # self.synset_ids = list(splits[splits.modelId.isin(self.model_ids)][splits.split.isin([self.split])]["synsetId"])



    def __getitem__(self, idx: int) -> Dict:
        """
        Read a model by the given index.
        Args:
            idx: The idx of the model to be retrieved in the dataset.
        Returns:
            dictionary with following keys:
            - verts: FloatTensor of shape (V, 3).
            - faces: LongTensor of shape (F, 3) which indexes into the verts tensor.
            - synset_id (str): synset id
            - model_id (str): model id
            - label (str): synset label.
        """
        model = self._get_item_ids(idx)
        model_path = path.join(
            self.shapenet_dir, model["synset_id"], model["model_id"], self.model_dir
        )
        print(model_path)
        print(gg)
        verts, faces, textures = self._load_mesh(model_path)
        label_str = self.synset_dict[model["synset_id"]]
        # model["verts"] = verts
        # model["faces"] = faces
        # model["textures"] = textures
        # model["label"] = self.synset_dict[model["synset_id"]]
        verts = torch_center_and_normalize(verts.to(torch.float), p=self.dset_norm)

        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = Textures(verts_rgb=verts_rgb)
        mesh = Meshes(
            verts=[verts],
            faces=[faces],
            textures=textures
        )
        points = trimesh.Trimesh(vertices=verts.numpy(), faces=faces.numpy()).sample(self.nb_points, False)
        points = torch.from_numpy(points).to(torch.float)
        points = torch_center_and_normalize(points, p=self.dset_norm)
        return self.label_by_number[label_str], mesh, points
        # return model
class Shapenet_batch:
    def __init__(self,points,):
        self.sequence_point_cloud = points
        print(points.shape)
        #self.
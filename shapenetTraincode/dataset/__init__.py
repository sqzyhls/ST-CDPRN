from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import pytorch3d
import torch
from torch.utils.data import SequentialSampler
from omegaconf import DictConfig
from pytorch3d.implicitron.dataset.data_loader_map_provider import \
    SequenceDataLoaderMapProvider
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.implicitron.dataset.json_index_dataset import JsonIndexDataset
from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import (
    JsonIndexDatasetMapProviderV2, registry)
from pytorch3d.implicitron.tools.config import expand_args_fields
from pytorch3d.renderer.cameras import CamerasBase
#from .shapenet_dataset import ShapeNetCore
from pytorch3d.datasets import (
    #R2N2,
    ShapeNetCore,
    #collate_batched_R2N2,
    #collate_batched_meshes,
    render_cubified_voxels,
)
from .r2n2 import R2N2
from pytorch3d.renderer.mesh import TexturesAtlas
from pytorch3d.structures import Meshes
from torch.utils.data import DataLoader
from config.structured import CO3DConfig, DataloaderConfig, ProjectConfig
from .exclude_sequence import EXCLUDE_SEQUENCE, LOW_QUALITY_SEQUENCE
from .utils import DatasetMap

def collate_batched_meshes(batch: List[Dict]):  # pragma: no coverW
    if batch is None or len(batch) == 0:
        return None
    collated_dict = {}
    for k in batch[0].keys():
        collated_dict[k] = [d[k] for d in batch]

    collated_dict["mesh"] = None
    if {"verts", "faces"}.issubset(collated_dict.keys()):

        textures = None
        if "textures" in collated_dict and collated_dict["textures"][0] is not None:
            print(collated_dict["textures"])
            textures = TexturesAtlas(atlas=collated_dict["textures"])

        collated_dict["mesh"] = Meshes(
            verts=collated_dict["verts"],
            faces=collated_dict["faces"],
            textures=textures,
        )

    return collated_dict
def collate_batched_R2N2(batch: List[Dict]):  # pragma: no cover
    collated_dict = collate_batched_meshes(batch)

    # If collate_batched_meshes receives R2N2 items with images and that
    # all models have the same number of views V, stack the batches of
    # views of each model into a new batch of shape (N, V, H, W, 3).
    # Otherwise leave it as a list.
    if "images" in collated_dict:
        try:
            collated_dict["images"] = torch.stack(collated_dict["images"])
        except RuntimeError:
            print(
                "Models don't have the same number of views. Now returning "
                "lists of images instead of batches."
            )

    # If collate_batched_meshes receives R2N2 items with camera calibration
    # matrices and that all models have the same number of views V, stack each
    # type of matrices into a new batch of shape (N, V, ...).
    # Otherwise leave them as lists.
    if all(x in collated_dict for x in ["R", "T", "K"]):
        try:
            collated_dict["R"] = torch.stack(collated_dict["R"])  # (N, V, 3, 3)
            collated_dict["T"] = torch.stack(collated_dict["T"])  # (N, V, 3)
            collated_dict["K"] = torch.stack(collated_dict["K"])  # (N, V, 4, 4)
        except RuntimeError:
            print(
                "Models don't have the same number of views. Now returning "
                "lists of calibration matrices instead of a batched tensor."
            )

    # If collate_batched_meshes receives voxels and all models have the same
    # number of views V, stack the batches of voxels into a new batch of shape
    # (N, V, S, S, S), where S is the voxel size.
    if "voxels" in collated_dict:
        try:
            collated_dict["voxels"] = torch.stack(collated_dict["voxels"])
        except RuntimeError:
            print(
                "Models don't have the same number of views. Now returning "
                "lists of voxels instead of a batched tensor."
            )
    return collated_dict
def get_dataset(cfg: ProjectConfig):
    
    if cfg.dataset.type == 'co3dv2':
        dataset_cfg: CO3DConfig = cfg.dataset
        dataloader_cfg: DataloaderConfig = cfg.dataloader

        # Exclude bad and low-quality sequences
        exclude_sequence = []
        exclude_sequence.extend(EXCLUDE_SEQUENCE.get(dataset_cfg.category, []))
        exclude_sequence.extend(LOW_QUALITY_SEQUENCE.get(dataset_cfg.category, []))
        
        # Whether to load pointclouds
        kwargs = dict(
            remove_empty_masks=True,
            n_frames_per_sequence=1,
            load_point_clouds=True,
            max_points=dataset_cfg.max_points,
            image_height=dataset_cfg.image_size,
            image_width=dataset_cfg.image_size,
            mask_images=dataset_cfg.mask_images,
            exclude_sequence=exclude_sequence,
            pick_sequence=() if dataset_cfg.restrict_model_ids is None else dataset_cfg.restrict_model_ids,
        )

        # Get dataset mapper
        dataset_map_provider_type = registry.get(JsonIndexDatasetMapProviderV2, "JsonIndexDatasetMapProviderV2")
        expand_args_fields(dataset_map_provider_type)
        dataset_map_provider = dataset_map_provider_type(
            category=dataset_cfg.category,
            subset_name=dataset_cfg.subset_name,
            dataset_root=dataset_cfg.root,
            test_on_train=False,
            only_test_set=False,
            load_eval_batches=True,
            dataset_JsonIndexDataset_args=DictConfig(kwargs),
        )

        # Get datasets
        datasets = dataset_map_provider.get_dataset_map()

        # PATCH BUG WITH POINT CLOUD LOCATIONS!
        for dataset in (datasets["train"], datasets["val"]):
            for key, ann in dataset.seq_annots.items():
                correct_point_cloud_path = Path(dataset.dataset_root) / Path(*Path(ann.point_cloud.path).parts[-3:])
                assert correct_point_cloud_path.is_file(), correct_point_cloud_path
                ann.point_cloud.path = str(correct_point_cloud_path)

        # Get dataloader mapper
        data_loader_map_provider_type = registry.get(SequenceDataLoaderMapProvider, "SequenceDataLoaderMapProvider")
        expand_args_fields(data_loader_map_provider_type)
        data_loader_map_provider = data_loader_map_provider_type(
            batch_size=dataloader_cfg.batch_size,
            num_workers=dataloader_cfg.num_workers,
        )

        # QUICK HACK: Patch the train dataset because it is not used but it throws an error
        if (len(datasets['train']) == 0 and len(datasets[dataset_cfg.eval_split]) > 0 and 
                dataset_cfg.restrict_model_ids is not None and cfg.run.job == 'sample'):
            datasets = DatasetMap(train=datasets[dataset_cfg.eval_split], val=datasets[dataset_cfg.eval_split], 
                                  test=datasets[dataset_cfg.eval_split])
            print('Note: You used restrict_model_ids and there were no ids in the train set.')

        # Get dataloaders
        dataloaders = data_loader_map_provider.get_data_loader_map(datasets)
        dataloader_train = dataloaders['train']
        dataloader_val = dataloader_vis = dataloaders[dataset_cfg.eval_split]

        # Replace validation dataloader sampler with SequentialSampler
        dataloader_val.batch_sampler.sampler = SequentialSampler(dataloader_val.batch_sampler.sampler.data_source)

        # Modify for accelerate
        dataloader_train.batch_sampler.drop_last = True
        dataloader_val.batch_sampler.drop_last = False
    elif cfg.dataset.type == 'shapenet_r2n2':
        dataset_cfg: SHAPENETConfig = cfg.dataset
        dataloader_cfg: DataloaderConfig = cfg.dataloader
        shapenet_dataset = ShapeNetCore(dataset_cfg.shapenet_dir,synsets=[dataset_cfg.category],version=1)
        r2n2_dataset_train = R2N2("train", dataset_cfg.shapenet_dir, dataset_cfg.r2n2_dir, dataset_cfg.splits_file,subsets=[dataset_cfg.category], return_voxels=False,load_textures=False,return_all_views=False, voxels_rel_path='ShapeNetVox32')
        dataloader_train = DataLoader(r2n2_dataset_train, batch_size=dataloader_cfg.batch_size, collate_fn=collate_batched_R2N2)
        r2n2_dataset_val = R2N2("val", dataset_cfg.shapenet_dir, dataset_cfg.r2n2_dir, dataset_cfg.splits_file,subsets=[dataset_cfg.category],  return_voxels=False,load_textures=False,return_all_views=False, voxels_rel_path='ShapeNetVox32')
        dataloader_val = DataLoader(r2n2_dataset_val, batch_size=dataloader_cfg.batch_size, collate_fn=collate_batched_R2N2)
        r2n2_dataset_test = R2N2("test", dataset_cfg.shapenet_dir, dataset_cfg.r2n2_dir, dataset_cfg.splits_file,subsets=[dataset_cfg.category],  return_voxels=False,load_textures=False,return_all_views=False, voxels_rel_path='ShapeNetVox32')

        dataloader_vis = DataLoader(r2n2_dataset_test, batch_size=dataloader_cfg.batch_size, collate_fn=collate_batched_R2N2)
        
        # Get dataloader mapper
        """
        data_loader_map_provider_type = registry.get(SequenceDataLoaderMapProvider, "SequenceDataLoaderMapProvider")
        expand_args_fields(data_loader_map_provider_type)
        data_loader_map_provider = data_loader_map_provider_type(
            batch_size=dataloader_cfg.batch_size,
            num_workers=dataloader_cfg.num_workers,
        )
                # Get dataset mapper
        dataset_map_provider_type = registry.get(JsonIndexDatasetMapProviderV2, "JsonIndexDatasetMapProviderV2")
        expand_args_fields(dataset_map_provider_type)
        dataset_map_provider = dataset_map_provider_type(
            category=dataset_cfg.category,
            subset_name='sofa',
            dataset_root=dataset_cfg.root,
            test_on_train=False,
            only_test_set=False,
            load_eval_batches=True,
            #dataset_JsonIndexDataset_args=DictConfig(kwargs),
        )
        print('ccccccc')
        print(dataset_map_provider)
        print(dataset_map_provider_type)
        print(data_loader_map_provider)
        print(gg)
        """
    """
    elif cfg.dataset.type == 'shapenet3d':
        dataset_cfg: SHAPENETConfig = cfg.dataset
        dataloader_cfg: DataloaderConfig = cfg.dataloader
        # Get dataloader mapper
        data_loader_map_provider_type = registry.get(SequenceDataLoaderMapProvider, "SequenceDataLoaderMapProvider")
        expand_args_fields(data_loader_map_provider_type)
        data_loader_map_provider = data_loader_map_provider_type(
            batch_size=dataloader_cfg.batch_size,
            num_workers=dataloader_cfg.num_workers,
        )
        dataloader_train = ShapeNetCore(data_dir=dataset_cfg.root,split='train',nb_points=dataset_cfg.max_points,synsets=[dataset_cfg.category],version=2)
        dataloader_val = ShapeNetCore(data_dir=dataset_cfg.root,split='val',nb_points=dataset_cfg.max_points,synsets=[dataset_cfg.category],version=2)
        dataloader_vis = ShapeNetCore(data_dir=dataset_cfg.root,split='test',nb_points=dataset_cfg.max_points,synsets=[dataset_cfg.category],version=2)
        #tmp =  DatasetMap(train=dataloader_train, val=dataloader_val, 
        #                          test=dataloader_vis)
        #dataloaders = data_loader_map_provider.get_data_loader_map(tmp)
        #print(dataloaders)
        #dataloader_train = dataloaders['train']
        #dataloader_val = dataloader_vis = dataloaders[dataset_cfg.eval_split]

        # Replace validation dataloader sampler with SequentialSampler
        #dataloader_val.batch_sampler.sampler = SequentialSampler(dataloader_val.batch_sampler.sampler.data_source)

        # Modify for accelerate
        #dataloader_train.batch_sampler.drop_last = True
        #dataloader_val.batch_sampler.drop_last = False
    else:
        raise NotImplementedError(cfg.dataset.type)
    """
    return dataloader_train, dataloader_val, dataloader_vis

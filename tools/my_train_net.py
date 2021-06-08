#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import json
import logging
import os
from collections import OrderedDict
from pathlib import Path

from clearml import Task

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results
from detectron2.modeling import GeneralizedRCNNWithTTA

from detectron2.utils.logger import setup_logger



from monk.config.classes import DAMAGES, PARTS, TYPES, Classes
from monk.config.models import get_config_centermask, get_config_detectron2
from monk.data.catalog import DatasetCatalog
from monk.data.tables.fill_database import FilesDB
from monk.evaluation.evaluators import CustomF1Evaluator, Instance2SemEvaluator, NumInstancesEvaluator  # TODO: add AP
from monk.evaluation.utils import Monk2DetectronEvaluator



from extend_cfg import get_config_blendmask

import sys
sys.path.append("/shared/perso/jeremy/libs/AdelaiDet/adet")

import adet
#import blendmask


######### IMPORT MONK MAPPER #######

import copy
import logging
import os
from typing import List, Optional, Union

import numpy as np
import torch
from fvcore.transforms.transform import CropTransform

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from monk.structures.boxes import BBox as MonkBBox


def get_crop_car_transform(dataset_dict, extend_pct):
    vehicle_bbox = dataset_dict.get("vehicule_bbox_xyxy")
    if vehicle_bbox is None:
        return None, (int(dataset_dict["width"]), int(dataset_dict["height"]))
    bbox = MonkBBox(xyxy=vehicle_bbox, label="vehicle", image_size=(dataset_dict["width"], dataset_dict["height"]))
    bbox = bbox.extend_pct(extend_pct)
    x0, y0, w, h = bbox.rounded().xywh
    return CropTransform(int(x0), int(y0), int(w), int(h)), (int(w), int(h))


class MonkDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
        crop_car: bool = False,
        crop_car_extend_bbox_pct: float = 0.0
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = augmentations
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        self.crop_car               = crop_car
        self.crop_car_extend_bbox_pct = crop_car_extend_bbox_pct
        # fmt: on
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            "crop_car": cfg.INPUT.CROP_CAR,
            "crop_car_extend_bbox_pct": cfg.INPUT.CROP_CAR_EXTEND_BBOX,
        }
        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        crop_car_tfm = []
        if self.crop_car:
            crop_car_transform, (new_width, new_height) = get_crop_car_transform(
                dataset_dict, self.crop_car_extend_bbox_pct
            )
            if crop_car_transform is not None:
                crop_car_tfm = [crop_car_transform]
                dataset_dict["width"] = new_width
                dataset_dict["height"] = new_height

        augmentations = crop_car_tfm + self.augmentations

        aug_input = T.StandardAugInput(image, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(augmentations)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk)

        if not self.is_train and not os.getenv("DETECTRON2_VAL_LOSS"):
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape, mask_format=self.instance_mask_format)

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict


####################################


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_types = cfg.TEST.EVALUATOR_TYPES
        for evaluator_type in evaluator_types:
            if evaluator_type == "instance2sem":
                evaluator_list.append(
                    Monk2DetectronEvaluator(
                        Instance2SemEvaluator,
                        dataset_name=dataset_name,
                        distributed=cfg.TEST.DISTRIBUTED,
                        output_dir=output_folder,
                        conf_thresholds=cfg.TEST.CONF_THRESHOLDS,
                        contour_ignore=cfg.TEST.INSTANCE2SEM.CONTOUR_IGNORE,
                        downsize_longer_side=cfg.TEST.DOWNSIZE_LONGER_SIDE,
                        crop_car=cfg.INPUT.CROP_CAR,
                        crop_car_extend_pct=cfg.INPUT.CROP_CAR_EXTEND_BBOX,
                        process_mode=cfg.TEST.PROCESS_MODE,
                    )
                )
            elif evaluator_type == "customf1":
                evaluator_list.append(
                    Monk2DetectronEvaluator(
                        CustomF1Evaluator,
                        dataset_name=dataset_name,
                        distributed=cfg.TEST.DISTRIBUTED,
                        output_dir=output_folder,
                        conf_thresholds=cfg.TEST.CONF_THRESHOLDS,
                        iou_thresh=dict(cfg.TEST.CUSTOMF1.IOU_THRESHOLDS),
                        do_avg_metrics=cfg.TEST.CUSTOMF1.AVG_METRICS,
                        weights=dict(cfg.TEST.CUSTOMF1.WEIGHTS),
                        downsize_longer_side=cfg.TEST.DOWNSIZE_LONGER_SIDE,
                        groupping_distance_thresh=cfg.TEST.GROUPPING_THRESH,
                        crop_car=cfg.INPUT.CROP_CAR,
                        crop_car_extend_pct=cfg.INPUT.CROP_CAR_EXTEND_BBOX,
                        process_mode=cfg.TEST.PROCESS_MODE,
                    )
                )
            elif evaluator_type == "ap":
                evaluator_list.append(COCOEvaluator(dataset_name, cfg, cfg.TEST.DISTRIBUTED, output_folder))
            elif evaluator_type == "num_instances":
                evaluator_list.append(
                    Monk2DetectronEvaluator(
                        NumInstancesEvaluator,
                        dataset_name=dataset_name,
                        distributed=cfg.TEST.DISTRIBUTED,
                        output_dir=output_folder,
                        conf_thresholds=cfg.TEST.CONF_THRESHOLDS,
                        crop_car=cfg.INPUT.CROP_CAR,
                        crop_car_extend_pct=cfg.INPUT.CROP_CAR_EXTEND_BBOX,
                        process_mode=cfg.TEST.PROCESS_MODE,
                    )
                )
            else:
                raise NotImplementedError
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA"))
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def build_hooks(self):
        hooks_ = super().build_hooks()
        return hooks_

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        mapper = MonkDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        mapper = MonkDatasetMapper(cfg, False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)


def setup_allegro():
    training_name = os.getenv('TRAINING_NAME', "NO_NAME")
    expfolder = os.getenv('EXPFOLDER', "?")
    return Task.init(
        project_name="Monk Detectron2",
        task_name=" ".join([training_name, expfolder]),
        task_type=Task.TaskTypes.training,
    )


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_config_blendmask()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = str(Path(args.config_file).parent)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")


    return cfg


def main(args):
    cfg = setup(args)
    if comm.is_main_process():
        task = setup_allegro()
        task.connect(cfg, 'cfg')
    register_monk_datasets(cfg)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks([hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))])
    return trainer.train()


def _get_monk_instances_meta(classes):
    continous_classes = classes.to_continous_classes()
    cats = continous_classes.to_detectron2()
    for cat in cats:
        cat["color"] = (0, 0, 0)
    thing_colors = [k["color"] for k in cats if k["isthing"] == 1]

    thing_dataset_id_to_contiguous_id = classes.map(continous_classes)
    thing_classes = [k["name"] for k in cats if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
        "classes": continous_classes,
    }
    return ret


def register_monk_datasets(cfg):
    args_dataset = {}
    if cfg.DATASETS.MIN_SIZE_SCRATCH != 0:
        args_dataset["min_size_scratch"] = cfg.DATASETS.MIN_SIZE_SCRATCH
    if cfg.DATASETS.MIN_SIZE_DENT != 0:
        args_dataset["min_size_dent"] = cfg.DATASETS.MIN_SIZE_DENT

    versions = {"train": cfg.DATASETS.TRAIN_VERSION, "test": cfg.DATASETS.TEST_VERSION}
    datasets = {x: "train" for x in cfg.DATASETS.TRAIN}
    datasets.update({x: "test" for x in cfg.DATASETS.TEST})

    # Read classes metadata from first annotation file
    dataset = cfg.DATASETS.TRAIN[0] if cfg.DATASETS.TRAIN else cfg.DATASETS.TEST[0]
    if dataset in DatasetCatalog.CUSTOM_DATASETS:
        ann_file = DatasetCatalog.CUSTOM_DATASETS[dataset]
    elif dataset.startswith("/"):
        ann_file = dataset
    else:
        dataset_name, dataset_set = dataset.rsplit("_", 1)
        dset = "train" if cfg.DATASETS.TRAIN else "test"
        version = versions[dset]
        ann_file = DatasetCatalog.get_ann_files(
            dataset_name=dataset_name,
            dataset_task=cfg.DATASETS.TASK,
            dataset_set=dataset_set,
            dataset_subcats=cfg.DATASETS.SUBCATS,
            dataset_version=version,
            create_if_not_exists=True,
            **args_dataset,
        )[0]

    with open(ann_file, "r") as f:
        data = json.load(f)
    dataset_classes = Classes.from_coco_categories(data["categories"])
    metadata = _get_monk_instances_meta(dataset_classes)

    if cfg.MODEL.ROI_HEADS.NUM_CLASSES != len(dataset_classes):
        raise ValueError(
            f"Number of classes in dataset: {len(dataset_classes)} but cfg.MODEL.ROI_HEADS.NUM_CLASSES"
            " says {cfg.MODEL.ROI_HEADS.NUM_CLASSES}"
        )
    metadata["classes"].to_file(Path(cfg.OUTPUT_DIR) / "classes.json")

    # Register datasets
    for dataset, mode in datasets.items():
        if dataset in DatasetCatalog.CUSTOM_DATASETS:
            ann_file = DatasetCatalog.CUSTOM_DATASETS[dataset]
            dataset_name_in_catalog = dataset
        elif dataset.startswith("/"):
            ann_file = dataset
            # TODO: that does not work, do somethink like that that works please (anyone)
            # dataset_name_in_catalog = os.path.basename(ann_file)
            dataset_name_in_catalog = dataset
        else:
            dataset_name, dataset_set = dataset.rsplit("_", 1)
            ann_file = DatasetCatalog.get_ann_files(
                dataset_name=dataset_name,
                dataset_task=cfg.DATASETS.TASK,
                dataset_set=dataset_set,
                dataset_subcats=cfg.DATASETS.SUBCATS,
                dataset_version=versions[mode],
                create_if_not_exists=True,
                **args_dataset,
            )[0]
            dataset_name_in_catalog = dataset
        if not os.path.exists(ann_file):
            raise FileNotFoundError(ann_file)
        with open(ann_file, "r") as f:
            data = json.load(f)
            if Classes.from_coco_categories(data["categories"]) != dataset_classes:
                raise ValueError(
                    f"""Wrong Classes in test file {ann_file},
            {dataset_classes} =! {Classes.from_coco_categories(data["categories"])}"""
                )
        register_coco_instances(dataset_name_in_catalog, metadata, str(ann_file), str(FilesDB.IMAGES_ROOT))


def get_parser(epilog=None):
    parser = default_argument_parser(epilog=epilog)
    parser.add_argument(
        "--model", default="default", help="default for detectron2 based models, centermask for centermask"
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

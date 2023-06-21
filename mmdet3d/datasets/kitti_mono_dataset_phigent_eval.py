import copy
import tempfile
from os import path as osp
from mmdet.datasets.api_wrappers.coco_api import COCO

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log
from tqdm import tqdm

from ..core.bbox import Box3DMode, CameraInstance3DBoxes, points_cam2img
from .builder import DATASETS
from .nuscenes_mono_dataset import NuScenesMonoDataset


@DATASETS.register_module()
class KittiMonoDatasetPhiEval(NuScenesMonoDataset):
    """Monocular 3D detection on KITTI Dataset.
    Args:
        data_root (str): Path of dataset root.
        info_file (str): Path of info file.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to False.
        eval_version (str, optional): Configuration version of evaluation.
            Defaults to None.
        version (str, optional): Dataset version. Defaults to None.
        kwargs (dict): Other arguments are the same of NuScenesMonoDataset.
    """

    CLASSES = ('Pedestrian', 'Cyclist', 'Car')

    def __init__(self,
                 data_root,
                 info_file,
                 ann_file,
                 pipeline,
                 load_interval=1,
                 with_velocity=False,
                 eval_version=None,
                 version=None,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            load_interval=load_interval,
            with_velocity=with_velocity,
            eval_version=eval_version,
            version=version,
            **kwargs)
        if info_file is not None:
            self.anno_infos = mmcv.load(info_file)
        self.bbox_code_size = 7

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.
        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.
        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_bboxes_cam3d = []
        centers2d = []
        depths = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
                # 3D annotations in camera coordinates
                bbox_cam3d = np.array(ann['bbox_cam3d']).reshape(-1, )
                gt_bboxes_cam3d.append(bbox_cam3d)
                # 2.5D annotations in camera coordinates
                center2d = ann['center2d'][:2]
                depth = ann['center2d'][2]
                centers2d.append(center2d)
                depths.append(depth)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_cam3d:
            gt_bboxes_cam3d = np.array(gt_bboxes_cam3d, dtype=np.float32)
            centers2d = np.array(centers2d, dtype=np.float32)
            depths = np.array(depths, dtype=np.float32)
        else:
            gt_bboxes_cam3d = np.zeros((0, self.bbox_code_size),
                                       dtype=np.float32)
            centers2d = np.zeros((0, 2), dtype=np.float32)
            depths = np.zeros((0), dtype=np.float32)

        gt_bboxes_cam3d = CameraInstance3DBoxes(
            gt_bboxes_cam3d,
            box_dim=gt_bboxes_cam3d.shape[-1],
            origin=(0.5, 0.5, 0.5))
        gt_labels_3d = copy.deepcopy(gt_labels)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_bboxes_3d=gt_bboxes_cam3d,
            gt_labels_3d=gt_labels_3d,
            centers2d=centers2d,
            depths=depths,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann


    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str): The prefix of submitted files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        result_files = self.bbox2result_kitti(outputs, self.CLASSES,
                                                pklfile_prefix,
                                                submission_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str, optional): The prefix of pkl files, including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str, optional): The prefix of submission data.
                If not specified, the submission data will not be generated.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        metric = None
        result_files, tmp_dir = self.format_results(results, pklfile_prefix)
        from mmdet3d.core.evaluation import phigent_eval
        coco = COCO(self.ann_file)
        img_ids=coco.get_img_ids()
        gt_annos = []
        print('\nPreparing gt annos!')
        for img_id in tqdm(img_ids):
            info = {}
            ann_ids = coco.get_ann_ids(img_ids=[img_id])
            anns = coco.load_anns(ann_ids)
            label_anno = self.get_label_anno(anns)
            info['annos'] = label_anno
            self.add_difficulty_to_annos(info)
            gt_annos.append(info['annos'])

        assert len(gt_annos) == len(result_files)
        eval_types = ['bev']
        ap_dict = phigent_eval(gt_annos, result_files,
                                        self.CLASSES, eval_types=eval_types)

        return ap_dict

    def bbox2result_kitti(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        """Convert 3D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str): The prefix of pkl file.
            submission_prefix (str): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries with the kitti format.
        """

        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to KITTI format')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            img_info = self.data_infos[idx]

            image_shape = np.array((img_info['height'], img_info['width']), dtype=np.int32)

            box_dict = self.convert_valid_bboxes(pred_dicts['img_bbox'], image_shape)
            anno = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': []
            }
            if len(box_dict['box3d_camera']) > 0:

                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                label_preds = box_dict['label_preds']
                for box, score, label in zip(box_preds, scores, label_preds):
                    if int(label) != 0: continue
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(-np.arctan2(box[0], box[2]) + box[6])
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)
                if len(anno['name']) > 0:
                    anno = {k: np.stack(v) for k, v in anno.items()}
                else:
                    anno = {
                        'name': np.array([]),
                        'truncated': np.array([]),
                        'occluded': np.array([]),
                        'alpha': np.array([]),
                        'dimensions': np.zeros([0, 3]),
                        'location': np.zeros([0, 3]),
                        'rotation_y': np.array([]),
                        'score': np.array([]),
                    }
                annos.append(anno)

            else:
                anno = {
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                }
                annos.append(anno)

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print('Result is saved to %s' % out)

        return det_annos


    def get_label_anno(self, anns):
        annotations = {}
        annotations.update({
            'name': [],
            'truncated': [],
            'occluded': [],
            'alpha': [],
            'bbox': [],
            'dimensions': [],
            'location': [],
            'rotation_y': []
        })
        bboxes_cam3d = []
        for ann in anns:
            if ann.get('ignore', False):
                continue
            if ann['category_id'] == 0:
                bboxes_cam3d.append(ann['bbox_cam3d'])
 
        bboxes_cam3d = np.array(bboxes_cam3d).reshape(-1, 7)
        num_gt = bboxes_cam3d.shape[0]

        annotations['name'] = np.array(['vehicle' for x in range(num_gt)])
        annotations['truncated'] = np.array([0. for x in range(num_gt)])
        annotations['occluded'] = np.array([0. for x in range(num_gt)])
        annotations['bbox'] = np.zeros((num_gt, 4))
        # dimensions will convert hwl format to standard lhw(camera) format.
        annotations['dimensions'] = bboxes_cam3d[:,3:6].reshape(-1, 3)
        annotations['location'] = bboxes_cam3d[:,:3].reshape(-1, 3)
        annotations['rotation_y'] = bboxes_cam3d[:,-1].reshape(-1)
        annotations['alpha'] = -np.arctan2(bboxes_cam3d[:, 0], bboxes_cam3d[:, 2]) + bboxes_cam3d[:, 6]
        annotations['score'] = np.zeros((num_gt, ))

        annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)

        return annotations
 
    def add_difficulty_to_annos(self, info):
        annos = info['annos']
        min_height = [0, 15, 30, 60]  # minimum height for evaluated groundtruth/detections
        location = annos['location']

        diff = []

        for loc in location:
            z = loc[-1]
            if z > min_height[0] and z <= min_height[1]:
                diff.append(0)
            elif z > min_height[1] and z <= min_height[2]:
                diff.append(1)
            elif z > min_height[2] and z <= min_height[3]:
                diff.append(2)
            else:
                diff.append(3)

        annos['difficulty'] = np.array(diff, np.int32)
        return diff

    def convert_valid_bboxes(self, box_dict, image_shape):
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.
                - boxes_3d (:obj:`CameraInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (torch.Tensor): Scores of boxes.
                - labels_3d (torch.Tensor): Class labels of boxes.

        Returns:
            dict: Valid predicted boxes.
                - bbox (np.ndarray): 2D bounding boxes.
                - box3d_camera (np.ndarray): 3D bounding boxes in
                    camera coordinate.
                - scores (np.ndarray): Scores of boxes.
                - label_preds (np.ndarray): Class label predictions.
                - sample_idx (int): Sample index.
        """
        box_preds = box_dict['boxes_3d']

        

        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        
        if len(box_preds) == 0:
            return dict(
                box3d_camera=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]))
        loc = box_preds.gravity_center.numpy()
        other = box_preds.tensor.numpy()[:, 3:]
        box3d_camera = np.concatenate([loc, other], axis=1)
        return dict(
            box3d_camera=box3d_camera,
            scores=scores.numpy(),
            label_preds=labels.numpy())
import os.path as osp
from typing import List, Dict, Union
from mmengine.fileio import load
from mmdet.registry import DATASETS
from mmdet.datasets import BaseDetDataset, CocoDataset

@DATASETS.register_module()
class YHDataset(CocoDataset):
    METAINFO = {
        'classes': ('jiye', 'qixiong'),
        'palette': [(220, 20, 60), (255, 0, 0)]
    }

    # def load_data_list(self) -> List[Dict]:
    #     """加载数据集的标注数据"""
    #     # 加载 JSON 文件
    #     ann_file = load(self.ann_file)
        
    #     # 获取图像、标注和类别信息
    #     img_infos = ann_file['images']
    #     annotations = ann_file['annotations']
    #     categories = {cat['id']: cat['name'] for cat in ann_file['categories']}
        
    #     # 数据列表
    #     data_list = []
        
    #     # 处理每张图像
    #     for img_info in img_infos:
    #         data_info = {
    #             'img_id': img_info['id'],
    #             'img_path': osp.join(self.data_root, img_info['file_name']),
    #             'width': img_info['width'],
    #             'height': img_info['height'],
    #             'instances': []
    #         }
            
    #         # 查找对应图像的标注
    #         for ann in annotations:
    #             if ann['image_id'] == img_info['id']:
    #                 instance = {
    #                     'bbox': ann['bbox'],  # [x, y, w, h]
    #                     'bbox_label': ann['category_id'],
    #                     'ignore_flag': ann['iscrowd']
    #                 }
    #                 data_info['instances'].append(instance)
            
    #         data_list.append(data_info)
    #     self.data_list = data_list
    #     return data_list

    # def parse_data_info(self, raw_data_info: Dict) -> Union[Dict, List[Dict]]:
    #     """处理单条数据信息"""
    #     data_info = {
    #         'img_id': raw_data_info['img_id'],
    #         'img_path': raw_data_info['img_path'],
    #         'width': raw_data_info['width'],
    #         'height': raw_data_info['height'],
    #         'instances': []
    #     }
        
    #     for instance in raw_data_info['instances']:
    #         data_info['instances'].append({
    #             'bbox': instance['bbox'],
    #             'bbox_label': instance['bbox_label'],
    #             'ignore_flag': instance['ignore_flag']
    #         })
    #     print(data_info)
    #     raise
    #     return data_info

    # def filter_data(self) -> List[Dict]:
    #     """过滤数据"""
    #     valid_data_infos = []
    #     for data_info in self.data_list:
    #         # 过滤掉没有标注的图像
    #         if len(data_info['instances']) > 0:
    #             valid_data_infos.append(data_info)

    #     return valid_data_infos
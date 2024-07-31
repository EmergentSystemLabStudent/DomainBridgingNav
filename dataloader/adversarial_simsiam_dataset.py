from __future__ import annotations
from typing import Collection, List

import torch 
from torch.utils.data import Dataset
from torchvision.transforms import Compose

import os 
import random 
from PIL import Image 

class AdversarialSimSiamDataset(Dataset):
    
    def __init__(self, dataset_dir: str, transforms: Collection=None) -> None:

        self._dataset_dir = dataset_dir
        if type(transforms) == List:
            self._transforms = Compose(transforms)
        else:
            self._transforms = transforms
        
        self._classes, self._instance_id_to_index = self._get_labels()
        self._lq_img_path_and_labels, self._hq_img_path_and_labels = self._get_image_path_dict()
        
        
    def _get_labels(self):
        # labels: instance id
        labels = [d.name for d in os.scandir(self._dataset_dir)]
        labels.sort()
        instance_id_to_index = {instance_id: i for i, instance_id in enumerate(labels)} # {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
        return labels, instance_id_to_index
    
    def __len__(self):
        return len(self._lq_img_path_and_labels)
    
    def __getitem__(self, index):
        
        domain_label1=0
        domain_label2=0
        lq_image_path, lq_label = self._lq_img_path_and_labels[index]
        hq_image_path_list = [img_path for img_path, hq_label in self._hq_img_path_and_labels if lq_label==hq_label]
        if len(hq_image_path_list)>0:
            other_view = random.choice(hq_image_path_list) 
            domain_label2 = 1
            
        else:
            same_object_images = [img_path for img_path, label in self._lq_img_path_and_labels if lq_label == label]
            other_view = random.choice(same_object_images)
            
        x_lq1, x_hq = Image.open(lq_image_path), Image.open(other_view)
        x_lq1, x_hq= self._transforms(x_lq1), self._transforms(x_hq)
        
        return x_lq1, x_hq,  self._instance_id_to_index[lq_label], domain_label1, domain_label2
    
    def _get_image_path_dict(self):
        
        # dataset_dir
        #   -> 1
        #   -> 2
        #       -> lq
        #           -> 1.png
        #       -> hq
        #           -> 2.png
        #   -> ...
        
        if self._dataset_dir:
            
            lq_img_path_and_labels = []
            hq_img_path_and_labels = []
            directory = os.path.expanduser(self._dataset_dir)
            for target_label in sorted(self._instance_id_to_index):
                label_index = self._instance_id_to_index[target_label]
                target_dir = os.path.join(directory, target_label)
                lq_img_dir = os.path.join(target_dir, "lq")
                hq_img_dir = os.path.join(target_dir, "hq")
                
                for root, _, file_names in sorted(os.walk(target_dir, followlinks = True)):
                    for file_name in file_names:
                        
                        if "lq" in root:
                            img_path = os.path.join(root, file_name)
                            img_path_and_label = img_path, target_label
                            lq_img_path_and_labels.append(img_path_and_label)
                        elif "hq" in root:
                            img_path = os.path.join(root, file_name)
                            img_path_and_label = img_path, target_label
                            hq_img_path_and_labels.append(img_path_and_label)
                            
        random.shuffle(lq_img_path_and_labels)
        random.shuffle(hq_img_path_and_labels)
        
        return lq_img_path_and_labels, hq_img_path_and_labels
            
    @property
    def classes(self):
        return self._classes
            
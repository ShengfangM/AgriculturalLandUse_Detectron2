#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import cv2, random
import numpy as np
import matplotlib.pyplot as plt

from detectron2.data import detection_utils as utils
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

#from detectron2.data import DatasetMapper
from detectron2.data import DatasetMapper   # the default mapper

# In[2]:


root = '/home/shengfangm/'
dirs = 'Agriculture-Vision-2021'


# In[3]:


labels  = ['background',
           'double_plant',
           'drydown',
           'endrow',
           'nutrient_deficiency',
           'planter_skip',
           'water',
           'waterway',
           'weed_cluster']
num_classes = 9


# In[4]:


from detectron2.structures import BoxMode
#get bbox and instance segmentation from mask
def get_label_annos(image, label_id) :
         
    #polys = mask2seg (image) #get all polys(segmentationa) from class data
    
    ret, thresh = cv2.threshold(image, 127, 255, 0)
    polys, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 
    #objs used to store segmentation and 
    objs = []  
    # create annotation for each poly(segmentation)
    for poly in polys:

        hi = len(poly)
        a = poly.reshape((hi,-1))

        px = a[:,0]
        py = a[:,1]

    #check if this is necessary????????????????????????????
        poly1 = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        if(len(poly1) > 2):
            poly1 = [p for x in poly1 for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly1],
                "category_id": label_id,
            }
            objs.append(obj)
    return objs
        


# In[5]:


def  create_segment(root, ds, base_name):
        
    label_file_name = base_name +'.png'
    dir_labels = os.path.join(dirs, ds, 'labels')
    dir_boundary = os.path.join(dirs, ds, 'boundaries')
    dir_masks = os.path.join(dirs, ds, 'masks')
    
    mask_file = os.path.join(root,dir_masks,label_file_name)
    image = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
    img_seg = np.where(image < 1, 255, 0) #set all mask area(0) value of 10, otherwise 0

#     count = np.count_nonzero(img_seg > 0) #count how many un masks area
#     print(count)

    label_id=0
    #objs used to store segmentations from all labels
    objs = []
    for label in labels:
        
        if label == 'background':
            label_file = os.path.join(root,dir_boundary,label_file_name)
            image = cv2.imread(label_file, cv2.IMREAD_UNCHANGED)
            img_seg = np.where(image < 1, 255, img_seg)
            
#             count = np.count_nonzero(img_seg == 10)
#             print(count)

        else:
            
            label_file = os.path.join(root,dir_labels,label, label_file_name)
            image = cv2.imread(label_file, cv2.IMREAD_UNCHANGED) 
            
            obj = get_label_annos(image, label_id)
            objs.extend(obj)
            

            img_seg = np.where(image > 125, label_id, img_seg)
            
#             count = np.count_nonzero(img_seg == label_id)
#             print(count)


        label_id +=1
    
    image = np.where(img_seg == 0, 255, 0)
    image = np.uint8(image)
    obj_all = get_label_annos(image, 0)
    obj_all.extend(objs)
    #objs.append(obj)
    img_seg = np.uint8(img_seg)
    #print("segment", img_seg.dtype)

    seg_path =  os.path.join(root, dirs, ds, 'segment')
    if not os.path.exists(seg_path):
        os.makedirs(seg_path)
        
    seg_filename = os.path.join(seg_path,label_file_name)
    cv2.imwrite(seg_filename, img_seg)
    
    return seg_filename, obj_all


# In[6]:


# Register all_classes in dataset to detectron2
def get_class_dicts(root,ds) :
    
    dir_image = os.path.join(dirs, ds,'images') # image files path    
    rgb_files_path = os.path.join(dir_image,'rgb') #rgb file path

    rgb_files_name = os.listdir(os.path.join(root,rgb_files_path)) #all rgb files 
    
    idx = 0  # index of each image in the path
    dataset_dicts = []
    testnum = len(rgb_files_name)
#testnum = len(rgb_files_name)//20
    #print(testnum)
    for file in rgb_files_name[:testnum]:
   
        img = cv2.imread(os.path.join(root,rgb_files_path,file))
         
        #get base file name
        base_name = file[:-4]
        #print(base_name)
        
        record = {}
        height, width = img.shape[:2]
        idx += 1
        
#        record["file_name"] = base_name

        record["file_name"] = os.path.join(root,rgb_files_path,file)
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        
        seg_file, objs = create_segment(root, ds, base_name)  
        
        record["annotations"] = objs
        record["sem_seg_file_name "] = seg_file
        dataset_dicts.append(record)

    return dataset_dicts


# In[7]:


dataset_base = "Ag_dataset_"

for d in ["train", "val"]:
    
    dataset_name = dataset_base + d
    if dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(dataset_name)

    DatasetCatalog.register(dataset_name, lambda d=d: get_class_dicts(root, d))
    MetadataCatalog.get(dataset_name ).set(thing_classes=labels)

classes_metadata = MetadataCatalog.get("Ag_dataset_train")
evalu_metadata = MetadataCatalog.get("Ag_dataset_val")


# In[8]:


#dataset_dicts = get_class_dicts(root,'train')


# In[9]:


# for d in random.sample(dataset_dicts, 15):

#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=classes_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     #cv2.imshow("result", out.get_image()[:, :, ::-1])
    
#     plt.imshow(out.get_image()[:, :, ::-1])
#     plt.show()
#     plt.close()


# In[10]:


# # Register the Rareplanes dataset to detectron2.
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("Ag_dataset_train", {}, "/register/Ag_dataset_train.json", "/mnt/d/RarePlanes/datasets/synthetic/train/images")
# register_coco_instances("rareplanes_dataset_val", {}, "/mnt/d/RarePlanes/datasets/synthetic/metadata_annotations/instances_test_aircraft.json", "/mnt/d/RarePlanes/datasets/synthetic/test/images")

def valid_img(filename):
    
        #  read image
    base_name = filename[:-4]
    mask_file = base_name.replace('images\rgb', 'labels')
    mask_file = mask_file +'.png'

    boundary_file = base_name.replace('images\rgb', 'boundaries')
    boundary_file = boundary_file +'.png'

    image_mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
    image_mask = image_mask//255
    image_boundary = cv2.imread(boundary_file, cv2.IMREAD_UNCHANGED)
    image_boundary = image_boundary//255

    image = np.multiply(image_mask, image_boundary)

    return image
	
# In[11]:

def readfile(filename):
    
	# mask = valid_img(filename)
        #  read image
    rgb_file = filename
    nir_file = rgb_file.replace('rgb', 'nir')
   
    image_rgb = cv2.imread(rgb_file, cv2.IMREAD_UNCHANGED)
    image_nir = cv2.imread(nir_file, cv2.IMREAD_UNCHANGED)
    image_nir = image_nir.reshape((image_rgb.shape[0], image_rgb.shape[1], -1))
    
    image = np.concatenate((image_nir, image_rgb), axis=2)
	
    return image
	
	
import copy
# torch
import torch
from detectron2.data import transforms as T
def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict) # it will be modified by code below
    #image = utils.read_image(dataset_dict["file_name"], format="BGR")
    #  read image
    image = readfile(dataset_dict["file_name"])
   
    #image = image[:,:,1:]
    # Define a sequence of augmentations:
    transform_list = [
            T.RandomBrightness(0.9, 1.1),
            T.RandomContrast(0.9, 1.1),
  #          T.RandomSaturation(0.9, 1.1),
  #          T.RandomLighting(0.9),
            T.RandomFlip(prob=0.75, horizontal=False, vertical=True),
            T.RandomFlip(prob=0.75, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.25, horizontal=False, vertical=True),
            T.RandomFlip(prob=0.25, horizontal=True, vertical=False),

    ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    imageshape = image.shape[:2] # h,w
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
       # utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        utils.transform_instance_annotations(obj, transforms, imageshape)
        for obj in dataset_dict.pop("annotations")
   #     if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, imageshape)
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


# In[12]:


# Evaluator
# Taken from https://www.kaggle.com/theoviel/competition-metric-map-iou
from detectron2.evaluation.evaluator import DatasetEvaluator
import pycocotools.mask as mask_util

def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    return np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)

def score(pred, targ):
    pred_masks = pred['instances'].pred_masks.cpu().numpy()
    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    enc_targs = list(map(lambda x:x['segmentation'], targ))
    ious = mask_util.iou(enc_preds, enc_targs, [0]*len(enc_targs))
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, ious)
        p = tp / (tp + fp + fn)
        prec.append(p)
    return np.mean(prec)

class MAPIOUEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name):
        dataset_dicts = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {item['image_id']:item['annotations'] for item in dataset_dicts}
            
    def reset(self):
        self.scores = []

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            if len(out['instances']) == 0:
                self.scores.append(0)    
            else:
                targ = self.annotations_cache[inp['image_id']]
                self.scores.append(score(out, targ))

    def evaluate(self):
        return {"MaP IoU": np.mean(self.scores)}


# In[13]:

class AugTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return MAPIOUEvaluator(dataset_name)
#     @classmethod
#     def build_test_loader(cls, cfg, name="Ag_dataset_test"):
#         return build_detection_test_loader(cfg, name, mapper=custom_mapper(False))
    
		#return build_detection_test_loader(cfg, name, mapper=DatasetMapper(cfg, is_train=False, augmentations=transform_list))
'''
transform_list = [
            T.RandomBrightness(0.9, 1.1),
            T.RandomContrast(0.9, 1.1),
            T.RandomSaturation(0.9, 1.1),
            T.RandomLighting(0.9),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
]

class AugTrainer(DefaultTrainer):   
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, 
                                            mapper=DatasetMapper(cfg, is_train=True, augmentations=transform_list),)
                                            # sampler=SAMPLER)
    
#    @classmethod
#    def build_test_loader(cls, cfg, name="Ag_dataset_val"):
#        return build_detection_test_loader(cfg, name, mapper=DatasetMapper(cfg, is_train=False, augmentations=transform_list))
# In[14]:
'''

cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
#cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.DATASETS.TRAIN = ("Ag_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2 # The speed of dataload to the ram once at a time. I reduced it from the genral value 4 to 2 and 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
#cfg.SOLVER.WARMUP_ITERS = 2000 
cfg.SOLVER.MAX_ITER = 10000 # and a good number of iterations #adjust up if val mAP is still rising, adjust down if overfit
#cfg.SOLVER.STEPS = (2000, 2500)# milestones where the LR is reduced
#cfg.SOLVER.WEIGHT_DECAY = 0.0005
#cfg.SOLVER.GAMMA = 0.05

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9  # 9 classes 
cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530, 123.675]
#cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
#cfg.MODEL.PIXEL_STD = [123.675, 116.280, 103.530, 123.675]
cfg.MODEL.PIXEL_STD = [58.395, 57.375, 57.120, 58.395]
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.


# In[15]:


# dataloder = build_detection_train_loader(cfg, mapper=custom_mapper)


# In[16]:


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


# In[17]:


#trainer = DefaultTrainer(cfg) 
trainer = AugTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


# In[ ]:

'''
from detectron2.engine import DefaultPredictor

def cfg_test():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    cfg.DATASETS.TEST = ("Ag_dataset_val",)
    cfg.MODEL.RETINANET.NUM_CLASSES = 9
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.6
    
    return cfg

cfg = cfg_test()
#predict = DefaultPredictor(cfg)

#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)
'''
# In[ ]:


evaluator = COCOEvaluator("Ag_dataset_val", cfg, False, output_dir="./output/")
#cfg.MODEL.WEIGHTS="./output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
#val_loader = build_detection_test_loader(cfg, "Ag_dataset_val")
#val_loader = build_detection_test_loader(cfg, "Ag_dataset_val", mapper=DatasetMapper(cfg, is_train=False, augmentations=transform_list))
#val_loder = build_detection_train_loader(cfg,"Ag_dataset_val", mapper=custom_mapper)
val_loader = build_detection_test_loader(cfg, "Ag_dataset_val", mapper=custom_mapper) 
inference_on_dataset(trainer.model, val_loader, evaluator)



# In[ ]:

'''
from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_class_dicts(root,'val')

for d in random.sample(dataset_dicts, 5): 

    img = readfile(d["file_name"])
    outputs = predictor(img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
       
    im = cv2.imread(d["file_name"])
   # outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=evalu_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2_imshow(out.get_image()[:, :, ::-1])
    
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.show()
    plt.close()
    for d in random.sample(dataset_dicts, 5):    
    im = cv2.imread(d["file_name"])
    print(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=evalu_metadata, 
                   scale=0.5, 
                  # instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    
    v2 = Visualizer(im[:, :, ::-1], metadata=evalu_metadata, scale=0.5)
    out2 = v2.draw_dataset_dict(d)

    plt.imshow(out2.get_image()[:, :, ::-1])
    plt.show()
    plt.close()
'''

# In[ ]:


# from detectron2.utils.visualizer import ColorMode
# json_file_test = "/mnt/d/RarePlanes/datasets/synthetic/metadata_annotations/instances_test_aircraft.json"
# coco_test=COCO(json_file_test)

# catIds = coco_test.getCatIds(catNms=['aircraft']);
# imgIds = coco_test.getImgIds(catIds=catIds);

# img = coco_test.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
# im = cv2.imread("/mnt/d/RarePlanes/datasets/synthetic/test/images/" + img['file_name'])
# plt.axis('off')
# # plt.imshow(im)
# # plt.show()
# plt.imsave('input.png',im)
# outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
# v = Visualizer(im[:, :, ::-1],
#                metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), 
#                scale=0.5, 
#                instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
# )
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# # plt.imshow(out.get_image()[:, :, ::-1])
# # plt.show()
# plt.imsave('out_put.png',out.get_image()[:, :, ::-1])


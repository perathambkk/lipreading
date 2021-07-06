import os.path
basepath = '/cfarhomes/peratham/lrcodebase'

# datasets #
GRID_path = '/vulcan/scratch/peratham/GRID-data'
LRW_path = '/vulcan/scratch/peratham/lrw'
LRS2_path = '/vulcan/scratch/peratham/lrs2'
LRS3_path = '/vulcan/scratch/peratham/lrs3'

GRID_frames = os.path.join(GRID_path, 'frames')
GRID_faces = os.path.join(GRID_path, 'faces')
GRID_mouths = os.path.join(GRID_path, 'mouths')

LRW_class_path = os.path.join(basepath, 'LRW_classes.yaml')
LRW_videos = os.path.join(LRW_path, 'lipread_mp4')
LRW_frames = os.path.join(LRW_path, 'frames')
LRW_mouths = os.path.join(LRW_path, 'mouths')
LRW_mouth_npz = os.path.join(LRW_path, 'mouth_npz')
LRW_mouths_rois = os.path.join(LRW_path, 'mouth_rois')
LRW_audios = os.path.join(LRW_path, 'audios')
LRW_flow_npz = os.path.join(LRW_path, 'flow_npz')
LRW_flows = os.path.join(LRW_path, 'flows')
LRW_flow_ims = os.path.join(LRW_path, 'flowims')
LRW_brox = os.path.join(LRW_path, 'brox')

#LRW OSL
OSLLRW_path = '/vulcan/scratch/peratham/lrw-osl'
OSLLRW_mouth_npz = os.path.join(OSLLRW_path, 'e2e_npz')

LRS2_videos = os.path.join(LRS2_path, 'mvlrs_v1')
LRS2_pretrained = os.path.join(LRS2_videos, 'pretrain')
LRS2_main = os.path.join(LRS2_videos, 'main')
LRS2_frames = os.path.join(LRS2_path, 'frames')
LRS2_audios = os.path.join(LRS2_path, 'audios')
LRS2_mouths = os.path.join(LRS2_path, 'mouths')

LRS3_trainval = os.path.join(LRS3_path, 'trainval')
LRS3_test = os.path.join(LRS3_path, 'test')
LRS3_pretrain = os.path.join(LRS3_path, 'pretrain')
LRS3_frames = os.path.join(LRS3_path, 'frames')
LRS3_mouths = os.path.join(LRS3_path, 'mouths')

############


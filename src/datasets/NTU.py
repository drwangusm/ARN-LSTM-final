import pandas as pd
import numpy as np
import glob

DATA_DIR = '/usr/local/inter-rel-net-hockey/data02/ntu-rgbd/'

""" Folder structure
    descs.csv
    skl.csv
"""

""" Folder structure 2
    nturgb+d_skeletons/
        ...
        S002C002P013R002A039.skeleton
        ...
        
    
    File Format:
        <number of frames>
        frame 1 info:
            <number of people>
            People 1 info:
                <header info: NOT SURE WHAT IS>
                <number of joints>
                <Joint i info: NOT SURE WHAT EACH COLUMN IS>
                ...
            People 2 info:
                ...
        frame 2 info:
            ...
"""

### Full names
# ACTIONS = ["drink water","eat meal/snack","brushing teeth","brushing hair","drop","pickup","throw","sitting down","standing up (from sitting position)","clapping","reading","writing","tear up paper","wear jacket","take off jacket","wear a shoe","take off a shoe","wear on glasses","take off glasses","put on a hat/cap","take off a hat/cap","cheer up","hand waving","kicking something","put something inside pocket / take out something from pocket","hopping (one foot jumping)","jump up","make a phone call/answer phone","playing with phone/tablet","typing on a keyboard (pc/laptop)","pointing to something with finger","taking a selfie","check time (from watch)","rub two hands together","nod head/bow (Japanese)","shake head","wipe face (using napkin/towel)","salute (policeman)","put the palms together (Indians/Thai) / make a bow with hands folded in front (Chinese)","cross hands in front (say stop)","sneeze/cough","staggering","falling","touch head (headache)","touch chest (stomach-ache/heart pain)","touch back (backache)","touch neck (neck-ache)","nausea or vomiting condition","use a fan (with hand or paper)/feeling warm","punching/slapping other person","kicking other person","pushing other person","pat on back of other person","point finger at the other person","hugging other person","giving something to other person","touch other person's pocket","handshaking","walking towards each other","walking apart each other"]

### 60 种动作，涵盖了日常活动、人际交互、与物体的交互等多个类别。总序列数: 56,880 个序列,每个动作类型的平均序列数: 约 948 个序列
ACTIONS = [
    "drink water","eat meal/snack","brushing teeth","brushing hair","drop","pickup","throw","sitting down","standing up (from sitting position)","clapping","reading","writing","tear up paper","wear jacket","take off jacket","wear a shoe","take off a shoe","wear on glasses","take off glasses","put on a hat/cap","take off a hat/cap","cheer up","hand waving","kicking something","put something inside pocket / take out something from pocket","hopping (one foot jumping)","jump up","make a phone call/answer phone","playing with phone/tablet","typing on a keyboard (pc/laptop)","pointing to something with finger","taking a selfie","check time (from watch)","rub two hands together","nod head/bow (Japanese)","shake head","wipe face (using napkin/towel)","salute (policeman)","put the palms together (Indians/Thai) / make a bow with hands folded in front (Chinese)","cross hands in front (say stop)","sneeze/cough","staggering","falling","touch head (headache)","touch chest (stomach-ache/heart pain)","touch back (backache)","touch neck (neck-ache)","nausea or vomiting condition","use a fan (with hand or paper)/feeling warm","Punch/slapping","Kicking","Pushing","PattingOnBack","PointingFinger","Hugging","GiveSomething","TouchingPocket","Handshaking","WalkingTowards","WalkingApart"]

# https://github.com/shahroudy/NTURGB-D/blob/master/Matlab/NTU_RGBD_samples_with_missing_skeletons.txt
# "302 of the captured samples in the "NTU RGB+D" dataset have missing or incomplete skeleton data.
# If you are working on skeleton-based analysis, please ignore these files in your training and testing procedures."

# S：设置号，共有17组设置；
# C：相机ID，共有3架相机；
# P：人物ID，共有40个人；
# R：同一个动作的表演次数；
# A：动作类别，共有60个。

# 数据样本的时候，是先对每个样本进行帧填充，以及规范化，中心化等操作，然后利用pickle操作，
# 把处理后的文件打包成npy文件，以及对应的标签文件，依据基于xsub和基于xview的不同beachmark,
# 进行数据生成。真正网络输入的是一个5维度向量（N,V,T,C,M） 分别代表一个batch内样本数，
# 25分关节点，多少帧，多少个通道，以及几个人物主体活动。


#问题数据302个，忽略
IGNORE_LIST = [
    'S001C002P005R002A008','S001C002P006R001A008','S001C003P002R001A055','S001C003P002R002A012','S001C003P005R002A004','S001C003P005R002A005','S001C003P005R002A006','S001C003P006R002A008','S002C002P011R002A030','S002C003P008R001A020','S002C003P010R002A010','S002C003P011R002A007','S002C003P011R002A011','S002C003P014R002A007','S003C001P019R001A055','S003C002P002R002A055','S003C002P018R002A055','S003C003P002R001A055','S003C003P016R001A055','S003C003P018R002A024','S004C002P003R001A013','S004C002P008R001A009','S004C002P020R001A003','S004C002P020R001A004','S004C002P020R001A012','S004C002P020R001A020','S004C002P020R001A021','S004C002P020R001A036','S005C002P004R001A001','S005C002P004R001A003','S005C002P010R001A016','S005C002P010R001A017','S005C002P010R001A048','S005C002P010R001A049','S005C002P016R001A009','S005C002P016R001A010','S005C002P018R001A003','S005C002P018R001A028','S005C002P018R001A029','S005C003P016R002A009','S005C003P018R002A013','S005C003P021R002A057','S006C001P001R002A055','S006C002P007R001A005','S006C002P007R001A006','S006C002P016R001A043','S006C002P016R001A051','S006C002P016R001A052','S006C002P022R001A012','S006C002P023R001A020','S006C002P023R001A021','S006C002P023R001A022','S006C002P023R001A023','S006C002P024R001A018','S006C002P024R001A019','S006C003P001R002A013','S006C003P007R002A009','S006C003P007R002A010','S006C003P007R002A025','S006C003P016R001A060','S006C003P017R001A055','S006C003P017R002A013','S006C003P017R002A014','S006C003P017R002A015','S006C003P022R002A013','S007C001P018R002A050','S007C001P025R002A051','S007C001P028R001A050','S007C001P028R001A051','S007C001P028R001A052','S007C002P008R002A008','S007C002P015R002A055','S007C002P026R001A008','S007C002P026R001A009','S007C002P026R001A010','S007C002P026R001A011','S007C002P026R001A012','S007C002P026R001A050','S007C002P027R001A011','S007C002P027R001A013','S007C002P028R002A055','S007C003P007R001A002','S007C003P007R001A004','S007C003P019R001A060','S007C003P027R002A001','S007C003P027R002A002','S007C003P027R002A003','S007C003P027R002A004','S007C003P027R002A005','S007C003P027R002A006','S007C003P027R002A007','S007C003P027R002A008','S007C003P027R002A009','S007C003P027R002A010','S007C003P027R002A011','S007C003P027R002A012','S007C003P027R002A013','S008C002P001R001A009','S008C002P001R001A010','S008C002P001R001A014','S008C002P001R001A015','S008C002P001R001A016','S008C002P001R001A018','S008C002P001R001A019','S008C002P008R002A059','S008C002P025R001A060','S008C002P029R001A004','S008C002P031R001A005','S008C002P031R001A006','S008C002P032R001A018','S008C002P034R001A018','S008C002P034R001A019','S008C002P035R001A059','S008C002P035R002A002','S008C002P035R002A005','S008C003P007R001A009','S008C003P007R001A016','S008C003P007R001A017','S008C003P007R001A018','S008C003P007R001A019','S008C003P007R001A020','S008C003P007R001A021','S008C003P007R001A022','S008C003P007R001A023','S008C003P007R001A025','S008C003P007R001A026','S008C003P007R001A028','S008C003P007R001A029','S008C003P007R002A003','S008C003P008R002A050','S008C003P025R002A002','S008C003P025R002A011','S008C003P025R002A012','S008C003P025R002A016','S008C003P025R002A020','S008C003P025R002A022','S008C003P025R002A023','S008C003P025R002A030','S008C003P025R002A031','S008C003P025R002A032','S008C003P025R002A033','S008C003P025R002A049','S008C003P025R002A060','S008C003P031R001A001','S008C003P031R002A004','S008C003P031R002A014','S008C003P031R002A015','S008C003P031R002A016','S008C003P031R002A017','S008C003P032R002A013','S008C003P033R002A001','S008C003P033R002A011','S008C003P033R002A012','S008C003P034R002A001','S008C003P034R002A012','S008C003P034R002A022','S008C003P034R002A023','S008C003P034R002A024','S008C003P034R002A044','S008C003P034R002A045','S008C003P035R002A016','S008C003P035R002A017','S008C003P035R002A018','S008C003P035R002A019','S008C003P035R002A020','S008C003P035R002A021','S009C002P007R001A001','S009C002P007R001A003','S009C002P007R001A014','S009C002P008R001A014','S009C002P015R002A050','S009C002P016R001A002','S009C002P017R001A028','S009C002P017R001A029','S009C003P017R002A030','S009C003P025R002A054','S010C001P007R002A020','S010C002P016R002A055','S010C002P017R001A005','S010C002P017R001A018','S010C002P017R001A019','S010C002P019R001A001','S010C002P025R001A012','S010C003P007R002A043','S010C003P008R002A003','S010C003P016R001A055','S010C003P017R002A055','S011C001P002R001A008','S011C001P018R002A050','S011C002P008R002A059','S011C002P016R002A055','S011C002P017R001A020','S011C002P017R001A021','S011C002P018R002A055','S011C002P027R001A009','S011C002P027R001A010','S011C002P027R001A037','S011C003P001R001A055','S011C003P002R001A055','S011C003P008R002A012','S011C003P015R001A055','S011C003P016R001A055','S011C003P019R001A055','S011C003P025R001A055','S011C003P028R002A055','S012C001P019R001A060','S012C001P019R002A060','S012C002P015R001A055','S012C002P017R002A012','S012C002P025R001A060','S012C003P008R001A057','S012C003P015R001A055','S012C003P015R002A055','S012C003P016R001A055','S012C003P017R002A055','S012C003P018R001A055','S012C003P018R001A057','S012C003P019R002A011','S012C003P019R002A012','S012C003P025R001A055','S012C003P027R001A055','S012C003P027R002A009','S012C003P028R001A035','S012C003P028R002A055','S013C001P015R001A054','S013C001P017R002A054','S013C001P018R001A016','S013C001P028R001A040','S013C002P015R001A054','S013C002P017R002A054','S013C002P028R001A040','S013C003P008R002A059','S013C003P015R001A054','S013C003P017R002A054','S013C003P025R002A022','S013C003P027R001A055','S013C003P028R001A040','S014C001P027R002A040','S014C002P015R001A003','S014C002P019R001A029','S014C002P025R002A059','S014C002P027R002A040','S014C002P039R001A050','S014C003P007R002A059','S014C003P015R002A055','S014C003P019R002A055','S014C003P025R001A048','S014C003P027R002A040','S015C001P008R002A040','S015C001P016R001A055','S015C001P017R001A055','S015C001P017R002A055','S015C002P007R001A059','S015C002P008R001A003','S015C002P008R001A004','S015C002P008R002A040','S015C002P015R001A002','S015C002P016R001A001','S015C002P016R002A055','S015C003P008R002A007','S015C003P008R002A011','S015C003P008R002A012','S015C003P008R002A028','S015C003P008R002A040','S015C003P025R002A012','S015C003P025R002A017','S015C003P025R002A020','S015C003P025R002A021','S015C003P025R002A030','S015C003P025R002A033','S015C003P025R002A034','S015C003P025R002A036','S015C003P025R002A037','S015C003P025R002A044','S016C001P019R002A040','S016C001P025R001A011','S016C001P025R001A012','S016C001P025R001A060','S016C001P040R001A055','S016C001P040R002A055','S016C002P008R001A011','S016C002P019R002A040','S016C002P025R002A012','S016C003P008R001A011','S016C003P008R002A002','S016C003P008R002A003','S016C003P008R002A004','S016C003P008R002A006','S016C003P008R002A009','S016C003P019R002A040','S016C003P039R002A016','S017C001P016R002A031','S017C002P007R001A013','S017C002P008R001A009','S017C002P015R001A042','S017C002P016R002A031','S017C002P016R002A055','S017C003P007R002A013','S017C003P008R001A059','S017C003P016R002A031','S017C003P017R001A055','S017C003P020R001A059']

#按照人物ID（P）进行划分,将这些id的样本作为训练集20人，共40320个样本；剩余的作为测试集（20人），共16560个样本。
TRAIN_SUBJECTS = [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38] 

def get_ground_truth(data_dir=DATA_DIR, only_mutual=True, only_non_mutual=False):
    ground_truth = pd.read_csv(data_dir+'/descs.csv', index_col=False, header=None).T
    ground_truth.columns = ['setup','camera','subject','duplicate','action','start_frame_pt','end_frame_pt',]
    
    if only_mutual:
        ground_truth = ground_truth[ground_truth.action >= 50]
    elif only_non_mutual:
        ground_truth = ground_truth[ground_truth.action < 50]
    
    ground_truth.action = ground_truth.action - 1
    ground_truth['DATA_DIR'] = data_dir
    
    return ground_truth

def get_folds():
    folds = ['cross_subject','cross_view']
    
    return folds

def get_train(fold_num, **kwargs):
    from misc import data_io
    gt_split = get_train_gt(fold_num)
    
    X, Y = data_io.get_data(gt_split, pose_style='NTU', **kwargs)
    
    return X, Y

def get_val(fold_num, **kwargs):
    from misc import data_io
    gt_split = get_val_gt(fold_num)
    
    X, Y = data_io.get_data(gt_split, pose_style='NTU', **kwargs)
    
    return X, Y

def get_train_gt(fold_num):
    only_mutual = ('_all' not in fold_num)
    fold_num = fold_num.replace('_all','')
    
    ground_truth = get_ground_truth(only_mutual=only_mutual)
    if fold_num == 'cross_subject':
        gt_split = ground_truth[ground_truth.subject.isin(TRAIN_SUBJECTS)]
    elif fold_num == 'cross_subject_subset':
        ground_truth = ground_truth[ground_truth.setup.isin([1,8,17])]
        gt_split = ground_truth[ground_truth.subject.isin(TRAIN_SUBJECTS)]
    elif fold_num == 'cross_view':
        gt_split = ground_truth[ground_truth.camera != 1]
    
    return gt_split

def get_val_gt(fold_num):
    only_mutual = ('_all' not in fold_num)
    fold_num = fold_num.replace('_all','')
    
    ground_truth = get_ground_truth(only_mutual=only_mutual)
    if fold_num == 'cross_subject':
        gt_split = ground_truth[~ground_truth.subject.isin(TRAIN_SUBJECTS)]
    elif fold_num == 'cross_subject_subset':
        ground_truth = ground_truth[ground_truth.setup.isin([1,8,17])]
        gt_split = ground_truth[~ground_truth.subject.isin(TRAIN_SUBJECTS)]
    elif fold_num == 'cross_view':
        gt_split = ground_truth[ground_truth.camera == 1]
    
    return gt_split

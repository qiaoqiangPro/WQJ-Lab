
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import pdb
import argparse
import configparser
import torch

# 图像预处理
from torchvision import transforms
from torchvision import datasets
# 定义数据加载器DataLoader
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import time
from torch.cuda.amp import autocast, GradScaler
from torch import nn as nn
import albumentations as A
import torch.nn.functional as F
from timm.models.resnet import resnet50
import timm
import random
import warnings
# warnings.filterwarnings("ignore")
config = configparser.ConfigParser() # 类实例化

# 定义文件路径
config.read(r'config.ini')

# task_name   = config['task']['task_name']
# task_id     = config['task']['task_id']
# task_base = '/home/qiaoqiang/data/animals_split'
# basepath = '/home/qiaoqiang/data/animals_split'
task_base = config['task']['task_base']
basepath = config['task']['task_base']

debug = False
def set_seed(seed=19840807):
    random.seed(seed) # python
    np.random.seed(seed) # numpy
    torch.manual_seed(seed) # pytorch
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True #模型固化
    torch.backends.cudnn.benchmark = False #模型初始优化
    os.environ['PYTHONHASHSEED'] = str(seed)
###############################################################
##### part1: build_transforms & build_dataset & build_dataloader
###############################################################
def build_transforms(CFG):
    data_transforms = {
        #防止结构扭曲特征被误判因此取消形变模糊，只用水平和垂直反转 再加入一些模糊模拟伪影
        #TODO：scale VerticalFlip HorizontalFlip 做对比消融实验
        "train": transforms.Compose([
            
            #改成CenterCrop 水平反转
            #改成CenterCrop 水平反转
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            #A.RandomGridShuffle(grid=(3, 3), p=1), #按网格shuffle
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # A.OneOf([
            #     A.Cutout(),
            #     A.CoarseDropout(),
            #     A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            #     A.GridDropout(),
            # ]),
            # A.CenterCrop(random.randint(int(height*0.7),int(height*0.9)),random.randint(int(weight*0.7),int(weight*0.9)))
            # A.CenterCrop(always_apply=False, p=1.0, height=CFG.img_size, width=CFG.img_size)
            #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.25),
            ]),
        "valid_test": transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])
        }
    return data_transforms


def build_dataloader(save_folder, fold, data_transforms, CFG):
    train_path = os.path.join(save_folder, f'fold_{fold}', 'train_images')
    val_path = os.path.join(save_folder, f'fold_{fold}', 'val_images')
    print('训练集路径', train_path)
    print('验证集路径', val_path)

    # 载入训练集
    train_dataset = datasets.ImageFolder(train_path, data_transforms['train'])

    # 载入验证集
    val_dataset = datasets.ImageFolder(val_path, data_transforms['valid_test'])

    # 训练集的数据加载器
    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.train_bs,
                              shuffle=True,
                              num_workers=CFG.num_workers
                              )

    # 测试集的数据加载器
    test_loader = DataLoader(val_dataset,
                             batch_size=CFG.valid_bs,
                             shuffle=False,
                             num_workers=CFG.num_workers
                             )
    print(f'{fold}折，训练集图像数量', len(train_dataset))
    print('类别个数', len(train_dataset.classes))
    print('各类别名称', train_dataset.classes)

    print(f'{fold}折，验证集图像数量', len(val_dataset))
    print('类别个数', len(val_dataset.classes))
    print('各类别名称', val_dataset.classes)

    # 类别和索引号 一一对应
    # 各类别名称
    class_names = train_dataset.classes
    n_class = len(class_names)

    # 映射关系：类别 到 索引号
    print(train_dataset.class_to_idx)
    # 映射关系：索引号 到 类别
    idx_to_labels = {y: x for x, y in train_dataset.class_to_idx.items()}

    # 保存为本地的 npy 文件
    np.save(f'idx_to_labels_{fold}.npy', idx_to_labels)
    np.save(f'labels_to_idx_{fold}.npy', train_dataset.class_to_idx)


    return train_loader, test_loader,class_names


###############################################################
##### >>>>>>> part2: build_model <<<<<<
###############################################################
class PupilVisionNetwork(nn.Module):
    def __init__(self,cfg):
        super(PupilVisionNetwork,self).__init__()
        
        self.cfg = cfg
        self.n_classes = len(cfg.classes)
                
        # Define Feature part (IMAGE)
        self.features =  timm.create_model(cfg.modelname, 
                                          pretrained=cfg.pretrained, 
                                          num_classes=2, 
                                          global_pool="", 
                                          in_chans=1)
        
        backbone_out = self.features.feature_info[-1]['num_chs'] #输出特征提取层

        # self.global_pool = GeM(p_trainable=False) #使用GeM 基础可以使用Maxpool

        self.BNNeck = nn.BatchNorm1d(backbone_out)
        
        # Define Classification part
        self.classification = torch.nn.Linear(backbone_out, self.n_classes) #分类头
          
    def forward(self, x):   
        x = self.features(x)

        return x
def build_model(CFG):
    model = timm.create_model(model_name = CFG.backbone, pretrained=True, num_classes = 90, in_chans = 3)
    # if CFG.modelname == 'resnet':
    #     model = resnet50()
    # elif CFG.modelname == 'VGG':
    #     model = VGG()
    # model = PupilVisionNetwork(CFG).to(CFG.device)
    # if CFG.finetune == "":
    #     return model
    # else:
    #     model.load_state_dict(torch.load(CFG.finetune))

    #     # Freeze
    #     freeze = ['features.features']  # parameter names to freeze (full or partial)
    #     if any(freeze):
    #         for k, v in model.named_parameters():
    #             if any(x in k for x in freeze):
    #                 # print('freezing %s' % k)
    #                 v.requires_grad = False
    model = model.to(CFG.device)
    return model

###############################################################
##### >>>>>>> part3: build_optim <<<<<<
###############################################################
def build_optim(model,CFG):
    if CFG.optim=='adamW':
        return torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    elif CFG.optim == 'adam':
        return torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    else:
        return torch.optim.SGD(model.parameters(),lr=CFG.lr,momentum=0.9,weight_decay=CFG.wd)

    ###############################################################


##### >>>>>>> part4: build_loss <<<<<<
###############################################################
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp, sn):
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


def build_loss(CFG):
    if CFG.loss == 'bce':
        #BCELoss     = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4]).cuda())#
        # BCELoss     = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.0204]).cuda())#
        BCELoss   =  nn.CrossEntropyLoss()
        return {"BCELoss":BCELoss,}

    if CFG.loss == 'circle':
        BCELoss     = CircleLoss(m=0.25, gamma=256)#
        return {"BCELoss":BCELoss,}

    elif CFG.loss == 'circle+focal':
        Focal = FocalLoss()
        CirceLoss     = CircleLoss(m=0.25, gamma=256)
        return {"FocalLoss":Focal,'CircleLoss':CirceLoss,}

    elif CFG.loss == 'bce+circle':
        BCELoss     = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]).cuda())
        CirceLoss     = CircleLoss(m=0.25, gamma=256)
        return {"BCELoss":BCELoss,'CircleLoss':CirceLoss,}



###############################################################
##### >>>>>>> part5: EPOCH Train & Valid <<<<<<
###############################################################
def train_one_batch(model, images, targets, losses_dict,CFG):
    '''
    运行一个 batch 的训练，返回当前 batch 的训练日志
    '''

    # with torch.cuda.amp.autocast():

        # 获得一个 batch 的数据和标注
    images = images.to(CFG.device)
    targets = targets.to(CFG.device)

    y_preds = model(images)  # 输入模型，执行前向预测

    debug = False
    if debug: print(y_preds.shape)
    if debug: print(targets.shape)

    if CFG.loss == 'bce+circle':
        bce_loss = losses_dict["BCELoss"](y_preds, targets)
        circle_loss = losses_dict["CircleLoss"](y_preds, targets)
        total_loss = 0.7 * bce_loss + 0.3 * circle_loss
        losses = total_loss

    if CFG.loss == 'circle+focal':
        focal_loss = losses_dict["FocalLoss"](y_preds, targets)
        circle_loss = losses_dict["CircleLoss"](y_preds, targets)
        total_loss = 0.7 * circle_loss + 0.3 * focal_loss
        losses = total_loss

    else:
        bce_loss = losses_dict["BCELoss"](y_preds, targets)
        losses = bce_loss

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
    # 获取当前 batch 的标签类别和预测类别
    _, preds = torch.max(y_preds, 1)  # 获得当前 batch 所有图像的预测类别
    preds = preds.cpu().numpy()

    # !!!!losses 选择

    loss = losses.detach().cpu().numpy()
    outputs = y_preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    log_train = {}
    log_train['epoch'] = epoch
    log_train['batch'] = batch_idx
    # 计算分类评估指标
    log_train['train_loss'] = loss
    log_train['train_accuracy'] = accuracy_score(targets, preds)
    # log_train['train_precision'] = precision_score(labels, preds, average='macro')
    # log_train['train_recall'] = recall_score(labels, preds, average='macro')
    # log_train['train_f1-score'] = f1_score(labels, preds, average='macro')

    return log_train


@torch.no_grad()
def valid_one_batch(model, valid_loader, losses_dict, CFG,class_names):
    loss_list = []
    labels_list = []
    preds_list = []
    losses_all, bce_all = 0, 0
    for images, labels in valid_loader:  # 生成一个 batch 的数据和标注
        images = images.to(CFG.device)
        targets = labels.to(CFG.device)
        y_preds = model(images)  # 输入模型，执行前向预测

        # 获取整个测试集的标签类别和预测类别
        _, preds = torch.max(y_preds, 1)  # 获得当前 batch 所有图像的预测类别
        preds = preds.cpu().numpy()

        if CFG.loss == 'bce+circle':
            bce_loss = losses_dict["BCELoss"](y_preds, targets)
            circle_loss = losses_dict["CircleLoss"](y_preds, targets)
            total_loss = 0.7 * bce_loss + 0.3 * circle_loss
            losses = total_loss

        if CFG.loss == 'circle+focal':
            focal_loss = losses_dict["FocalLoss"](y_preds, targets)
            circle_loss = losses_dict["CircleLoss"](y_preds, targets)
            total_loss = 0.7 * circle_loss + 0.3 * focal_loss
            losses = total_loss

        else:
            bce_loss = losses_dict["BCELoss"](y_preds, targets)
            losses = bce_loss

        # loss = criterion(y_preds, labels)  # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值

        losses = losses.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        loss_list.append(losses)
        labels_list.extend(labels)
        preds_list.extend(preds)



    log_val = {}
    log_val['epoch'] = epoch

    # macro：不考虑类别数量，不适用于类别不均衡的数据集，其计算方式为： 各类别的P求和/类别数量

    # weighted:各类别的P × 该类别的样本数量（实际值而非预测值）/ 样本总数量

    # 计算分类评估指标
    log_val['test_loss'] = np.mean(losses)
    log_val['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_val['test_precision'] = precision_score(labels_list, preds_list, average='weighted')
    # log_val['test_auc'] = roc_auc_score(labels_list, preds_list)
    log_val['test_recall'] = recall_score(labels_list, preds_list, average='weighted')
    log_val['test_f1-score'] = f1_score(labels_list, preds_list, average='weighted')
    # 混淆矩阵
    # conf_mat = confusion_matrix(labels_list, preds_list)
    # log_val['confusion_matrix'] = conf_mat

    # 计算 P-R 曲线
    # precision, recall, _ = precision_recall_curve(labels_list, preds_list)
    # pr_auc = auc(recall, precision)

    # 计算 ROC 曲线
    # fpr, tpr, _ = roc_curve(labels_list, preds_list)
    # roc_auc = auc(fpr, tpr)


    # 绘制混淆矩阵图
    # fig, ax = plt.subplots(figsize=(8, 6))
    # sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt='g', ax=ax)
    # ax.set_xlabel('Predicted Labels')
    # ax.set_ylabel('True Labels')
    # ax.set_title('Confusion Matrix')
    # wandb.log({"Confusion Matrix": wandb.Image(fig)})


    # # 将 P-R 曲线、ROC 曲线和 AUC 记录到日志中
    # log_val['pr_curve'] = {
    #     "precision": precision.tolist(),
    #     "recall": recall.tolist(),
    #     "auc": pr_auc
    # }
    #
    # log_val['roc_curve'] = {
    #     "fpr": fpr.tolist(),
    #     "tpr": tpr.tolist(),
    #     "auc": roc_auc
    # }
    #
    # # 将 P-R 曲线和 ROC 曲线记录到 wandb 中
    # wandb.log({"PR Curve": wandb.plot.line_series(
    #     xs=[0, 1],
    #     ys=[precision, recall],
    #     keys=["precision", "recall"],
    #     title="PR Curve",
    #     xname="Recall",
    # )})
    # wandb.log({"ROC Curve": wandb.plot.line_series(
    #     xs=[0, 1],
    #     ys=[fpr, tpr],
    #     keys=["fpr", "tpr"],
    #     title="ROC Curve",
    #     xname="False Positive Rate",
    #
    # )})

    return log_val



def parse_args():
    parser = argparse.ArgumentParser(description='BC breast cancer classifican')
    # basic
    parser.add_argument('--debug', default='debug',
                        help='use debug mode')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('-m', '--model', default='resnet',
                        help='model:like resnet50,efff0')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='The num of epochs')
    parser.add_argument('--optim',default='adamW',
                        help='adamW, adam or sgd The optimizer')
    parser.add_argument('--loss',default='bce',
                        help='loss func')
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='learning rate')
    parser.add_argument('--lr_drop', default=5, type=float,
                        help='lr lr_drop to drop')
    # parser.add_argument('--tfboard', action='store_true', default=False,
    #                     help='use tensorboard')
    parser.add_argument('--flag', default=876, type=float,
                        help='different model')
    # parser.add_argument('-ft', '--finetune', default='',
    #                     help='finetune checkpoint path')
    # parser.add_argument('-ft', '--finetune', default='',
    #                     help='finetune checkpoint path')
    return parser.parse_args()


if __name__ == '__main__':
    ###############################################################
    ##### >>>>>>> config <<<<<<
    ###############################################################

    BASE_PATH = task_base
    print(BASE_PATH)

    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    # model name
    print('model: ', args.model)
    print('batch_size: ', args.batch_size)
    print('num_epochs: ', args.num_epochs)
    print('optim: ', args.optim)
    print('lr: ', args.lr)
    print('lr_drop: ', args.lr_drop)
    print('loss: ', args.loss)
    # print('finetune: ', args.finetune)
    print('debug: ', args.debug)

    # cfg = SimpleNamespace(**{}) #SimpleNamespace 可以作为 class 的替代品 您可以添加和删除属性
    # cfg.img_size = 1024
    # cfg.backbone = 'seresnext50_32x4d'
    # cfg.pretrained=False
    # cfg.in_channels = 1
    # cfg.classes = ['cancer']
    # cfg.batch_size = 8
    # cfg.data_folder = "/tmp/output/"
    # cfg.val_aug = A.CenterCrop(always_apply=False, p=1.0, height=cfg.img_size, width=cfg.img_size)
    # cfg.device = DEVICE

    models = {'resnet': 'resnet50', 'densenet': 'densenet169', 'seresnet': 'seresnet50',
              'swin': 'swin_base_patch4_window7_224', 'efficient':'efficientnet_b0',
              'convit':'convit','vgg':'vgg16','maxxvit':'maxxvit','seresnext': 'seresnext50_32x4d',
              'pvt': 'pvt_v2_b2','vit':'vit_base_patch16_224'}


    class CFG:
        # step1: hyper-parameter
        seed = 19840807
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ckpt_fold = "logs"
        debug = args.debug

        # step2: data
        n_fold = 5
        img_size = 224
        train_bs = args.batch_size
        valid_bs = args.batch_size
        batch_size = args.batch_size
        num_workers = 16  # 16

        # step3: model
        backbone = models[args.model]
        modelname = args.model
        pretrained = True
        in_channels = 3
        classes = ['cancer']
        # output_size = 1 #输出的类别数
        # no_columns = 4 #元数据的个数
        # no_features = 2048 #输出特征图的通道数
        # in_channels = 3

        # step4: optimizer
        epoch = args.num_epochs
        optim = args.optim
        lr = args.lr
        wd = 1e-5
        lr_drop = args.lr_drop
        tblogger = None

        # step5: loss
        loss = args.loss

        # step6: infer
        thr = 0.4  # hyper-parameter

        # data_folder = "/home/qiaoqiang/nnUNetFrame/nao_xue_2dclassi/2dtwo_class"
        # val_aug = A.CenterCrop(always_apply=False, p=1.0, height=img_size, width=img_size)

        ckpt_name = "model-{}_opt-{}_bs-{}_ephs-{}_lr-{}_lrdrop-{}_loss-{}_fold-{}_flag-{}".format(args.model,
                                                                                                   args.optim,
                                                                                                   args.batch_size,
                                                                                                   args.num_epochs,
                                                                                                   args.lr,
                                                                                                   args.lr_drop,
                                                                                                   args.loss, n_fold,
                                                                                                   args.flag)  # for submit. #
        print('ckpt_name: ', ckpt_name)

        # step7: finetune
        # finetune = args.finetune

        tblogger = None


    set_seed(CFG.seed)
    ckpt_path = f"{BASE_PATH}/{CFG.ckpt_fold}/{CFG.ckpt_name}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    log = open(ckpt_path + '/log.train.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (CFG.ckpt_name, '-' * 12))
    log.write('\n')
    log.write('model:%s\n ' % args.model)
    log.write('batch_size: %s\n' % args.batch_size)
    log.write('num_epochs: %s\n' % args.num_epochs)
    log.write('optim: %s\n' % args.optim)
    log.write('lr: %s\n' % args.lr)
    log.write('lr_drop: %s\n' % args.lr_drop)
    log.write('loss: %s\n' % args.loss)
    log.write('debug: %s\n' % args.debug)

    # path to save model
    path_to_save = f'logs'
    os.makedirs(path_to_save, exist_ok=True)

    train_val_flag = True
    if train_val_flag:
        ###############################################################
        ##### part0: data preprocess
        ###############################################################
        # newdf = pd.read_csv(os.path.join(BASE_PATH,f'test.csv'))
        # newdf['age'] = newdf['age'].fillna(58)
        # 使用了balance sampler
        newdf = ""

        ###############################################################
        ##### >>>>>>> trick1: cross validation train <<<<<<
        ###############################################################

        # skf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        for fold in range(CFG.n_fold):
            fold = fold+1

            print('\n')
            print('    __________            .__.__          .__       .__ ')
            print('    \______   \__________ |__|  |   ___  _|__| _____|__| ____   ____')
            print('    |     ___/  |  \____ \|  |  |   \  \/ /  |/  ___/  |/  _ \ /    \ ')
            print('    |    |   |  |  /  |_> >  |  |__  \   /|  |\___ \|  (  <_> )   |  \\')
            print('    |____|   |____/|   __/|__|____/   \_/ |__/____  >__|\____/|___|  /')
            print('                   |__|                           \/               \/ \n')


            # if CFG.finetune != "":
            #     print('====FINETUNE MODE====\n')
            print(f'----FOLD:{fold}----|------------------ VALID------------|---- TRAIN/BATCH --------')
            print('ep     lr     |   @th    AUC    P      R      F1P  |  loss    best  | time(s)')
                   #1 0.000100|0.4469 0.1777 0.8011 0.0842 0.7247  | 5.3901  0.4469 | 80.98
            print('-----------------------------------------------------------------------------')


            log.write('\n###### [Fold %s] %s\n\n' % (fold, '#' * 12))
            log.write('\n')
            log.write('** start training here! **\n')
            log.write('              |----------------- VALID----------------|---- TRAIN/BATCH -----------\n')
            log.write(
                'ep     lr       |    @th       AUC       P        R        F1P    |    loss       best   | time           \n')
            log.write('------------------------------------------------------------------------------\n')
            log.flush()

            ###############################################################
            ##### >>>>>>> step2: combination <<<<<<
            ##### build_transforme() & build_dataset() & build_dataloader()
            ##### build_model() & build_loss()
            ###############################################################

            data_transforms = build_transforms(CFG)
            train_loader, valid_loader,class_names = build_dataloader(basepath, fold, data_transforms, CFG)  # dataset & dtaloader



            model = build_model(CFG)# model

            optimizer = build_optim(model, CFG)

            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, CFG.lr_drop)
            # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200,T_mult=2, eta_min=0.00001, last_epoch=-1, verbose=False)
            # lr_scheduler = PolynomialLR(optimizer, total_iters=CFG.epoch, power=1.0)
            # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5,T_mult=2, eta_min=1e-6)
            # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=0.001, total_steps = CFG.epoch)
            # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=0.001,epochs=100,steps_per_epoch=185,div_factor=0.1,final_div_factor=1000,three_phase=True,last_epoch=-1,verbose=False)
            # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CFG.epoch, eta_min=0.00001, last_epoch=-1, verbose=False)
            losses_dict = build_loss(CFG)  # loss



            # 训练开始之前，记录日志
            epoch = 0
            batch_idx = 0
            best_test_accuracy = 0

            # 训练日志-训练集
            df_train_log = pd.DataFrame()
            log_train = {}
            # log_train['epoch'] = 0
            # log_train['batch'] = 0
            # images, labels = next(iter(train_loader))
            #
            # # log_train.update(train_one_batch(model, images, labels, losses_dict, CFG))
            #
            # df_train_log = df_train_log.append(log_train, ignore_index=True)
            #
            # print(df_train_log)
            #
            # # 训练日志-验证集
            df_val_log = pd.DataFrame()
            log_val = {}
            # log_val['epoch'] = 0
            # # log_val.update(valid_one_batch())
            # df_val_log = df_val_log.append(log_val, ignore_index=True)
            #
            # print(df_val_log)

            import wandb

            wandb.init(project='2dclass', name= CFG.modelname+'lr_'+str(CFG.lr)+'_'+str(fold)+'_'+time.strftime('%m-%d-%H-%M-%S'))

            # scaler = GradScaler()
            for epoch in range(1, CFG.epoch+1):

                print(f'Epoch {epoch}/{CFG.epoch}')
                start_time = time.time()
                ###############################################################
                ##### >>>>>>> step3: train & val <<<<<<
                ###############################################################
                model.train()
                for images, labels in tqdm(train_loader):  # 获得一个 batch 的数据和标注
                    batch_idx += 1
                    log_train = train_one_batch(model, images, labels, losses_dict, CFG)
                    # df_train_log = df_train_log.append(log_train, ignore_index=True)
                    # 将字典转换为 DataFrame
                    df_train = pd.DataFrame.from_dict(log_train, orient='index', columns=['epoch'])
                    df_train_log = pd.concat([df_train_log, df_train], ignore_index=True)
                    df_train_log.index = pd.RangeIndex(len(df_train_log))

                    wandb.log(log_train)

                lr_scheduler.step()

                ## 验证阶段
                model.eval()
                log_val = valid_one_batch(model, valid_loader, losses_dict, CFG, class_names)
                # df_val_log = df_val_log.append(log_val, ignore_index=True)

                df_val = pd.DataFrame.from_dict(log_val, orient='index', columns=['epoch'])
                df_val_log = pd.concat([df_val_log, df_val], ignore_index=True)
                df_val_log.index = pd.RangeIndex(len(df_val_log))

                wandb.log(log_val)



                if epoch==1:
                    root_f = os.path.join(ckpt_path, f'fold_{fold}')

                    if not os.path.exists(root_f):
                        os.makedirs(root_f)

                # 保存最新的最佳模型文件
                if log_val['test_accuracy'] > best_test_accuracy:
                    # 删除旧的最佳模型文件(如有)
                    old_best_checkpoint_path = os.path.join(ckpt_path,f'fold_{fold}','best-{:.3f}.pth'.format(best_test_accuracy))
                    if os.path.exists(old_best_checkpoint_path):
                        os.remove(old_best_checkpoint_path)
                    # 保存新的最佳模型文件

                    new_best_checkpoint_path= os.path.join(ckpt_path, f'fold_{fold}','best-{:.3f}.pth'.format(log_val['test_accuracy']))
                    torch.save(model, new_best_checkpoint_path)
                    print('保存新的最佳模型', new_best_checkpoint_path)
                    best_test_accuracy = log_val['test_accuracy']

                epoch_time = time.time() - start_time
                # print('epoch_time: ',epoch_time)
                current_lr = optimizer.param_groups[0]['lr']
                # pdb.set_trace()
                # print("epoch:{}, time:{:.2f}s, best:{}\n".format(epoch, epoch_time, best_val_dice), flush=True)

                # log.write("{} {:.6f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f}  {:.4f} | {:.2f}   \n ".format(epoch,
                #                                                                                                   current_lr,
                #                                                                                                   val_dices[
                #                                                                                                       4],
                #                                                                                                   val_dices[
                #                                                                                                       0],
                #                                                                                                   val_dices[
                #                                                                                                       1],
                #                                                                                                   val_dices[
                #                                                                                                       2],
                #                                                                                                   val_dices[
                #                                                                                                       3],
                #                                                                                                   loss_all,
                #                                                                                                   best_val_dice,
                #                                                                                                   epoch_time))
                # log.flush()


            df_train_log.to_csv(os.path.join(ckpt_path, f'fold_{fold}', f'{CFG.modelname}_{fold}折训练日志-训练集.csv'),
                            index=False)
            df_val_log.to_csv(os.path.join(ckpt_path, f'fold_{fold}', f'{CFG.modelname}_{fold}折训练日志-测试集.csv'),
                          index=False)
            wandb.finish()
        '''
            # 在测试集上评价
            # 载入最佳模型作为当前模型
            model = torch.load('checkpoints/best-{:.3f}.pth'.format(best_test_accuracy))
            model.eval()
            df_test_log = evaluate_testset()
            print(df_test_log)
            # 保存字典文件
            import json
        
            df_test_data = convert_to_float(df_test_log)
        
            json_str = json.dumps(df_test_data)
        
            with open('test_result.json', 'w') as json_file:
                json_file.write(json_str)
        
        '''



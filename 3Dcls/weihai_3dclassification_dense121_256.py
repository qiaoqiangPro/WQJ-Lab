import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import logging
import pdb
import random
import sys
import shutil
import tempfile
import read_data
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset, CSVSaver
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
)


def data_shuffle(self, x, random=None):
    res = [i for i in range(len(x))]
    if random is None:
        randbelow = self._randbelow
        for i in reversed(range(1, len(x))):
            # pick an element in x[:i+1] with which to exchange x[i]
            j = randbelow(i + 1)
            x[i], x[j] = x[j], x[i]
            res[i], res[j] = res[j], res [i]
    return res



pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print_config()

# Set data directory
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

data_root = r'/home/qiaoqiang/data/data_only_mask_area'
extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', '.gz')

_,images,labels,_,_ = read_data.make_dataset(data_root,extensions=extensions)


labels = torch.nn.functional.one_hot(torch.as_tensor(labels)).float()


# Define transforms
reshape = 256
train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((reshape,reshape, 32)), RandRotate90()])

val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((reshape, reshape, 32))])

# Define nifti dataset, data loader
check_ds = ImageDataset(image_files=images, labels=labels)
check_loader = DataLoader(check_ds, batch_size=3, num_workers=2, pin_memory=pin_memory)

im, label = monai.utils.misc.first(check_loader)
print(type(im), im.shape, label, label.shape)




length = len(images)
# 得到位置下标
indices = np.arange(length)
np.random.seed(123)
np.random.shuffle(indices)
# 划分训练、验证、测试集
val_frac = 0.1
test_frac = 0.1

#划分训练、测试集、验证集下标
test_split = int(test_frac * length)
val_split = int(val_frac * length) + test_split
test_indices = indices[:test_split]
val_indices = indices[test_split:val_split]
train_indices = indices[val_split:]

train_x = [images[i] for i in train_indices]
train_y = [labels[i] for i in train_indices]
val_x = [images[i] for i in val_indices]
val_y = [labels[i] for i in val_indices]
test_x = [images[i] for i in test_indices]
test_y = [labels[i] for i in test_indices]


print(
    f"Training count: {len(train_x)}, Validation count: "
    f"{len(val_x)}, Test count: {len(test_x)}")

batch_size = 8

# create a training data loader
train_ds = ImageDataset(image_files=train_x, labels=train_y, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)

# create a validation data loader
val_ds = ImageDataset(image_files=val_x, labels=val_y, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4, pin_memory=pin_memory)


# Create DenseNet121, CrossEntropyLoss and Adam optimizer
model_name = 'DenseNet121'
model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)

loss_function = torch.nn.CrossEntropyLoss()
# loss_function = torch.nn.BCEWithLogitsLoss()  # also works with this data

optimizer = torch.optim.Adam(model.parameters(), 1e-4)
#
# start a typical PyTorch training
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
save_path = 'logs/'+ model_name+'_'+str(reshape)
writer = SummaryWriter(log_dir=save_path)
max_epochs = 500


for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()

        num_correct = 0.0
        metric_count = 0
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            with torch.no_grad():
                val_outputs = model(val_images)
                value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
                metric_count += len(value)
                num_correct += value.sum().item()

        metric = num_correct / metric_count
        metric_values.append(metric)


        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), f"best_metric_model_classification3d_{model_name}_{reshape}.pth")
            print("saved new best metric model")

        print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f} ")
        print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
        writer.add_scalar("val_accuracy", metric, epoch + 1)

print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()


# create a validation data loader
test_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((reshape, reshape, 32))])
test_ds = ImageDataset(image_files=test_x, labels=test_y, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())



# 测试
model.eval()
with torch.no_grad():
    num_correct = 0.0
    metric_count = 0
    saver = CSVSaver(output_dir="./output")
    # pdb.set_trace()
    for test_data in test_loader:
        test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
        test_outputs = model(test_images).argmax(dim=1)
        value = torch.eq(test_outputs, test_labels.argmax(dim=1))
        metric_count += len(value)
        num_correct += value.sum().item()
        saver.save_batch(test_outputs, test_images.meta)
    metric = num_correct / metric_count
    print("evaluation metric:", metric)
    saver.finalize()







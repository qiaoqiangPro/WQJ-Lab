import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple


# 从 data 根目录自动获取不同的类别文件夹，并自动给文件夹标签
def find_classes(directory: str):
    """Finds the class folders in a dataset.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


# 检查 file 的后缀是不是在允许的扩展中
def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


# 从根目录中获取 图像的类别，以及自动为类别设置类标签，返回【图像-标签对， 类别名， 类别对应的索引等】
def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        classes, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    img_label_dict = []
    imgs = []
    labels = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                if is_valid_file(fname):
                    path = os.path.join(root, fname)
                    item = {'img': path, 'label': class_index}
                    img_label_dict.append(item)
                    imgs.append(path)
                    labels.append(class_index)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return img_label_dict, imgs, labels, classes, class_to_idx


if __name__ == '__main__':
    data_root = r'F:\brain_3dclassifition'
    # classes, class_to_idx = find_classes(data_root)
    # 允许的扩展名
    extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', '.gz')
    img_label_dict, imgs, labels, classes, class_to_idx= make_dataset(data_root, extensions=extensions)
    print(img_label_dict)

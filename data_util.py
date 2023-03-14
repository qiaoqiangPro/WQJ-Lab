from sklearn.model_selection import KFold
import os
import shutil


parent_folder = "/home/qiaoqiang/data/animals/animals"
save_folder = "animals_5folds"



class_folders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]

k_folds = 5

for class_dir in class_folders:
    print(f"Processing data in {class_dir}...")

    # Get all patient IDs in the current class folder
    patient_ids = set()
    for filename in os.listdir(class_dir):
        patient_id = filename.split("_")[0]
        patient_ids.add(patient_id)

    # Generate a dictionary to store the data paths of each patient
    data = {}
    for patient_id in patient_ids:
        data[patient_id] = []
        for filename in os.listdir(class_dir):
            if filename.startswith(patient_id):
                filepath = os.path.join(class_dir, filename)
                data[patient_id].append(filepath)









    # Split the data using k-fold cross validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    train_data_fold1 = []
    val_data_fold1 = []

    train_data_fold2 = []
    val_data_fold2 = []

    train_data_fold3 = []
    val_data_fold3 = []

    train_data_fold4 = []
    val_data_fold4 = []

    train_data_fold5 = []
    val_data_fold5 = []


    for fold, (train_indices, test_indices) in enumerate(kf.split(data)):
        print(f"Processing fold {fold+1}...")

        if fold==0:
            for idx in train_indices:
                train_data_fold1.extend(data[list(data.keys())[idx]])

            for idx in test_indices:
                val_data_fold1.extend(data[list(data.keys())[idx]])



            # 创建新文件夹用于存储k折的训练集和测试集图像
            train_path = f"fold_{fold + 1}/train_images"
            val_path = f"fold_{fold + 1}/val_images"


            # 将训练集图像复制到train_images文件夹中
            for train_data in train_data_fold1:

                new_path = os.path.join(save_folder, train_path, class_dir.split('/')[-1])
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                new_path = os.path.join(new_path, os.path.basename(train_data))

                shutil.copy(train_data, new_path)

            # 将测试集图像复制到test_images文件夹中
            for test_data in val_data_fold1:
                new_path = os.path.join(save_folder, val_path, class_dir.split('/')[-1])
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                new_path = os.path.join(new_path, os.path.basename(test_data))

                shutil.copy(test_data, new_path)

            print(f"Train data fold1: {len(train_data_fold1)} samples")
            print(f"Test data fold1: {len(val_data_fold1)} samples")

        if fold==1:
            for idx in train_indices:
                train_data_fold2.extend(data[list(data.keys())[idx]])

            for idx in test_indices:
                val_data_fold2.extend(data[list(data.keys())[idx]])



            # 创建新文件夹用于存储k折的训练集和测试集图像
            train_path = f"fold_{fold + 1}/train_images"
            val_path = f"fold_{fold + 1}/val_images"

            # 将训练集图像复制到train_images文件夹中
            for train_data in train_data_fold2:

                new_path = os.path.join(save_folder, train_path, class_dir.split('/')[-1])
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                new_path = os.path.join(new_path, os.path.basename(train_data))

                shutil.copy(train_data, new_path)

            # 将测试集图像复制到test_images文件夹中
            for test_data in val_data_fold2:
                new_path = os.path.join(save_folder, val_path, class_dir.split('/')[-1])
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                new_path = os.path.join(new_path, os.path.basename(test_data))

                shutil.copy(test_data, new_path)

            print(f"Train data fold2: {len(train_data_fold2)} samples")
            print(f"Test data fold2: {len(val_data_fold2)} samples")

        if fold==2:
            for idx in train_indices:
                train_data_fold3.extend(data[list(data.keys())[idx]])

            for idx in test_indices:
                val_data_fold3.extend(data[list(data.keys())[idx]])

            import os
            import shutil

            # 创建新文件夹用于存储k折的训练集和测试集图像
            train_path = f"fold_{fold + 1}/train_images"
            val_path = f"fold_{fold + 1}/val_images"

            # 将训练集图像复制到train_images文件夹中
            for train_data in train_data_fold3:

                new_path = os.path.join(save_folder, train_path, class_dir.split('/')[-1])
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                new_path = os.path.join(new_path, os.path.basename(train_data))

                shutil.copy(train_data, new_path)

            # 将测试集图像复制到test_images文件夹中
            for test_data in val_data_fold3:
                new_path = os.path.join(save_folder, val_path, class_dir.split('/')[-1])
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                new_path = os.path.join(new_path, os.path.basename(test_data))

                shutil.copy(test_data, new_path)

            print(f"Train data fold3: {len(train_data_fold3)} samples")
            print(f"Test data fold3: {len(val_data_fold3)} samples")


        if fold==3:
            for idx in train_indices:
                train_data_fold4.extend(data[list(data.keys())[idx]])

            for idx in test_indices:
                val_data_fold4.extend(data[list(data.keys())[idx]])



            # 创建新文件夹用于存储k折的训练集和测试集图像
            train_path = f"fold_{fold + 1}/train_images"
            val_path = f"fold_{fold + 1}/val_images"

            # 将训练集图像复制到train_images文件夹中
            for train_data in train_data_fold4:

                new_path = os.path.join(save_folder, train_path, class_dir.split('/')[-1])
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                new_path = os.path.join(new_path, os.path.basename(train_data))

                shutil.copy(train_data, new_path)

            # 将测试集图像复制到test_images文件夹中
            for test_data in val_data_fold4:
                new_path = os.path.join(save_folder, val_path, class_dir.split('/')[-1])
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                new_path = os.path.join(new_path, os.path.basename(test_data))

                shutil.copy(test_data, new_path)

            print(f"Train data fold4: {len(train_data_fold4)} samples")
            print(f"Test data fold4: {len(val_data_fold4)} samples")



        if fold==4:
            for idx in train_indices:
                train_data_fold5.extend(data[list(data.keys())[idx]])

            for idx in test_indices:
                val_data_fold5.extend(data[list(data.keys())[idx]])



            # 创建新文件夹用于存储k折的训练集和测试集图像
            train_path = f"fold_{fold + 1}/train_images"
            val_path = f"fold_{fold + 1}/val_images"

            # 将训练集图像复制到train_images文件夹中
            for train_data in train_data_fold5:

                new_path = os.path.join(save_folder, train_path, class_dir.split('/')[-1])
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                new_path = os.path.join(new_path, os.path.basename(train_data))

                shutil.copy(train_data, new_path)

            # 将测试集图像复制到test_images文件夹中
            for test_data in val_data_fold5:
                new_path = os.path.join(save_folder, val_path, class_dir.split('/')[-1])
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                new_path = os.path.join(new_path, os.path.basename(test_data))

                shutil.copy(test_data, new_path)

            print(f"Train data fold5: {len(train_data_fold5)} samples")
            print(f"Test data fold5: {len(val_data_fold5)} samples")


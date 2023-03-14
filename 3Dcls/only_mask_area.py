
import nibabel as nib
import numpy as np
import os

def converse(path,root,patient,save_path):
    # 加载NIfTI数据的数据
    img = nib.load(path)
    data = img.get_fdata()
    # 加载NIfTI数据的mask
    p_label_path = os.path.join(root,'label',patient)
    for i in os.listdir(p_label_path):
        if i.split("_")[1].startswith('t1c'):

            mask = nib.load(os.path.join(p_label_path,i))
            mask_data = mask.get_fdata()

            # 将mask应用于原始图像，提取肿瘤图像
            tumor_img = np.multiply(data, mask_data)
            # 将numpy数组转换NIfTI格式的数据
            tumor_nii = nib.Nifti1Image(tumor_img, img.affine, img.header)

            # 保存NIfTI数据文件(nii.gz)
            save_path = os.path.join(save_path,bing,source,patient)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            nib.save(tumor_nii, os.path.join(save_path,i))


if __name__ == '__main__':

    bing = 'xueguan'
    source = 'niidata'
    modal = 't1c'

    root = rf'F:\weihai\{bing}\data_crop\{bing}'
    root_data = rf'F:\weihai\{bing}\data_crop\{bing}\{source}'

    save_path = rf'F:\weihai\{bing}\data_only_mask_area'

    for patient in os.listdir(root_data):
        patient_path = os.listdir(os.path.join(root_data, patient))
        for f in patient_path:
            if f.split("_")[1].startswith(modal):
                path = os.path.join(root_data,patient,f)
                converse(path,root,patient,save_path)





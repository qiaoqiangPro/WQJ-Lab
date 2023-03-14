import os
import shutil


def refold(root, class_name,savepath):
    for c in class_name:
        every_class_root = os.path.join(root,c,'data_crop',c,'niidata')
        for p in os.listdir(every_class_root):
            p_path = os.path.join(every_class_root, p)
            for file in os.listdir(p_path):
                if file.split('_')[1]=='t1c.nii.gz':
                    end = os.path.join(savepath,c)
                    if not os.path.exists(end):
                        os.makedirs(end)
                    shutil.copy(os.path.join(p_path,file),os.path.join(end,file))


if __name__ == '__main__':
    root = r'F:\weihai'
    class_name = ['naomo','xueguan']
    savepath = r'F:\brain_3dclassifition'
    refold(root,class_name,savepath)
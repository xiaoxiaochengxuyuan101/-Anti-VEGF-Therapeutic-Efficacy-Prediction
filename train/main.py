import os
from PIL import Image


def check_files(data_dir):
    # 定义三个主要目录
    main_dirs = ["疗效稳定", "疗效好", "疗效差"]

    for main_dir in main_dirs:
        main_dir_path = os.path.join(data_dir, main_dir)

        if not os.path.isdir(main_dir_path):
            print(f"目录 {main_dir_path} 不存在")
            continue

        # 遍历主目录中的所有样本文件夹
        for sample_folder in os.listdir(main_dir_path):
            sample_folder_path = os.path.join(main_dir_path, sample_folder)

            if not os.path.isdir(sample_folder_path):
                continue

            # 遍历样本文件夹中的所有文件
            for file_name in os.listdir(sample_folder_path):
                file_path = os.path.join(sample_folder_path, file_name)

                if not os.path.isfile(file_path):
                    continue

                # 检查文件名是否为 H 或 V
                file_name_no_ext, file_ext = os.path.splitext(file_name)
                if file_name_no_ext not in ["H", "V"]:
                    print(f"文件名不是以 H 或 V 为全名: {file_path}")

                # 检查文件格式是否为 jpg
                if file_ext.lower() != ".jpg":
                    print(f"文件格式不是 jpg: {file_path}")


# 示例用法
data_dir = 'E:\\预测数据'
check_files(data_dir)

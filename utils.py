
import os
from datetime import datetime
def log_dir():
    # 获取当前日期和时间
    now = datetime.now()
    date_time_str = now.strftime("%y%m%d_%H%M")

    # 创建路径
    log_dir = "./log"
    new_folder_path = os.path.join(log_dir, date_time_str)

    # 创建log文件夹（如果不存在的话）
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建新文件夹
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    return new_folder_path+'/'
    
import torch

def model_size(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()  # nelement: number of elements, element_size: size of each element in bytes
        param_sum += param.nelement()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = (param_size + buffer_size) / 1024**2  # Convert to MB
    print(f"Model parameters: {param_sum}")
    print(f"Model size: {total_size:.2f} MB")

import shutil
def copy_py_files(source_path,save_path):
    if not os.path.exists(source_path):
        print(f"Source path {source_path} does not exist.")
        return
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if file.endswith('.py'):
                source_file = os.path.join(root, file)
                destination_file = os.path.join(save_path, file)
                try:
                    shutil.copy2(source_file, destination_file)
                except Exception as e:
                    print(f"Failed to copy {source_file} to {destination_file}: {e}")
                    
def copy_py_files(source_path, save_path):
    if not os.path.exists(source_path):
        print(f"Source path {source_path} does not exist.")
        return
    files = os.listdir(source_path)
    for file in files:
        if file.endswith('.py') or file.endswith('.sh'):
            source_file = os.path.join(source_path, file)
            destination_file = os.path.join(save_path, file)
            try:
                shutil.copy2(source_file, destination_file)
            except Exception as e:
                print(f"Failed to copy {source_file} to {destination_file}: {e}")
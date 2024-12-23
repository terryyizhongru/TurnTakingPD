import os
import shutil

def merge_directories(src_dir, dest_dir, overwrite=False):
    """
    将 src_dir 中的所有文件和目录复制到 dest_dir 中，并保留目录结构。
    :param src_dir: 源目录路径
    :param dest_dir: 目标目录路径
    :param overwrite: 是否覆盖同名文件, 默认为 False 表示跳过同名文件。
    """
    for root, dirs, files in os.walk(src_dir):
        # 计算相对 src_dir 的路径，用于在 dest_dir 创建相同结构的子目录
        relative_path = os.path.relpath(root, src_dir)
        # 在目标目录中创建相应的子目录
        target_subdir = os.path.join(dest_dir, relative_path)
        if not os.path.exists(target_subdir):
            os.makedirs(target_subdir, exist_ok=True)
        
        # 复制文件到对应的目标目录
        for filename in files:
            src_file = os.path.join(root, filename)
            dest_file = os.path.join(target_subdir, filename)
            
            # 如果文件已存在且不想覆盖，则跳过
            if os.path.exists(dest_file) and not overwrite:
                print(f"[SKIP] {dest_file} 已存在，跳过复制。")
                continue
            
            # 如果 overwrite=True 或者目标文件不存在，则复制
            shutil.copy2(src_file, dest_file)
            print(f"[COPY] {src_file} -> {dest_file}")

            
def merge_two_trees(directory_a, directory_b, directory_c, overwrite=False):
    """
    先合并目录 A，再合并目录 B 到同一个目标目录 C。
    :param directory_a: 源目录 A
    :param directory_b: 源目录 B
    :param directory_c: 目标目录 C
    :param overwrite: 是否覆盖同名文件
    """
    # 先合并目录 A
    print(f"开始合并 {directory_a} -> {directory_c}")
    merge_directories(directory_a, directory_c, overwrite=overwrite)
    
    # 再合并目录 B
    print(f"开始合并 {directory_b} -> {directory_c}")
    merge_directories(directory_b, directory_c, overwrite=overwrite)
    
    print("合并完成！")


if __name__ == "__main__":
    # 示例：根据实际路径修改
    dir_a = r"/data/storage025/Turntaking/wavs_single_channel_normalized_nosil"
    dir_b = r"/data/storage025/Turntaking/PD_batch3_raw_single_channel_normalized_nosil"
    dir_c = r"/data/storage025/Turntaking/all_batch123"

    # merge_two_trees 会依次将 A、B 中的所有文件复制到 C 中
    # 如果需要覆盖已存在的同名文件，请将 overwrite=True
    merge_two_trees(dir_a, dir_b, dir_c, overwrite=False)

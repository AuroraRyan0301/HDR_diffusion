import os 
import re
from tqdm import tqdm
import json
def _list_path_recursively(data_dir, cache_file="rgbd_pairs.json"):
    print("read exr")
    scene_txt_path = os.path.join(data_dir, 'scene.txt')
    if not os.path.exists(scene_txt_path):
        raise FileNotFoundError(f"scene.txt not found in {data_dir}")
    
    with open(scene_txt_path, 'r') as f:
        scene_paths = [line.strip() for line in f if line.strip()]
    all_depth_paths_pair = []
    for scene_path in tqdm(scene_paths):
        scene_path = os.path.join(data_dir,scene_path)
        for filename in os.listdir(scene_path):
            if filename.startswith('depth') and filename.endswith('.exr'):
                numbers_match = re.findall(r'\d+', filename)
                if not numbers_match:
                    continue  # 跳过未匹配到数字的文件
                numbers = int(numbers_match[0]) // 10000
                # 构造对应的 RGB 文件路径
                depth_path = os.path.join(scene_path, filename)
                rgb_path = os.path.join(scene_path, f"rgb_{numbers}.exr")
                # 检查文件是否存在
                if os.path.exists(depth_path) and os.path.exists(rgb_path):
                    all_depth_paths_pair.append([depth_path, rgb_path])
    with open(cache_file, 'w') as f:
        json.dump(all_depth_paths_pair, f, indent=4)
    print(f"Saved file pairs to {cache_file}")
    return all_depth_paths_pair


pairs = _list_path_recursively("/data2/infinigen_processed_v0", cache_file="/data2/infinigen_processed_v0/rgbd_pairs.json")


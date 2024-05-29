import torch
import os
from tqdm import tqdm, trange
import math
import json
from multiprocessing import Process, Value, Array, Queue
from collections import OrderedDict
import argparse

input_dirs = None
data_dir = None
output_dir = None

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        # 使用有序字典来保存缓存，维护元素的访问顺序
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            # 将访问的元素移到字典的最前面
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        else:
            value = self.init(key)
            self.put(key, value)
            return value

    def put(self, key, value):
        if key in self.cache:
            # 如果键已经存在，更新值并将其移到字典的最前面
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            # 如果缓存已满，删除最近最少使用的元素（字典中的第一个元素）
            last_key, last_value = self.cache.popitem(last=False)
            self.delete(last_key, last_value)

        self.cache[key] = value
        
    def init(self, key):
        pass
    
    def delete(self, key, value):
        pass
    

class FileLRUCache(LRUCache):
    def init(self, key):
        return open(key, 'rb')
    
    def delete(self, key, value):
        value.close()
        
                
def read_worker(tid, mapping_queue, data_queue):
    fp_cache = FileLRUCache(10)
    
    while True:
        mapping = mapping_queue.get()
        if mapping is None:
            mapping_queue.put(None)
            break
        file_path = mapping['file_path']
        base_name = os.path.basename(file_path)
        file_path = os.path.join(data_dir, base_name)
        fp = fp_cache.get(file_path)
        fp.seek(mapping['start_pos'])
        length = mapping['end_pos'] - mapping['start_pos']
        data = json.loads(fp.read(length))
        data['label'] = mapping['label']
        data['score'] = mapping['score']
        data['file_path'] = file_path
        data_queue.put(data)
        

def read_mapping_worker(tid, fn_list, mapping_queue, mapping_dir, label_dir):
    for fn in fn_list:
        mappings = []
        basename = fn.split('.')[0]
        with open(os.path.join(mapping_dir, basename + '.jsonl'), 'rb') as fp:
            for line in fp:
                mappings.append(json.loads(line))
        states = torch.load(os.path.join(label_dir, fn), map_location='cpu')
        labels = states['label'].tolist()
        scores = states['score'].tolist()
        for i in range(len(mappings)):
            mappings[i]['label'] = labels[i]
            mappings[i]['score'] = scores[i]
            mapping_queue.put(mappings[i])
    print('FINISHED read_mapping_worker')
    mapping_queue.put(None)

def write_worker(tid, data_queue, n_clusters, max_length_per_file, write_counter):
    cluster_data = {i: [] for i in range(n_clusters)}
    progress_bar = tqdm(total=1, position=tid, disable=True)
    
    while True:
        data = data_queue.get()
        if data is None:
            data_queue.put(None)
            break            
        index = data['label']
        cluster_data[index].append(data)
        progress_bar.update()
        if len(cluster_data[index]) > max_length_per_file:
            with write_counter.get_lock():
                wid = write_counter[index]
                write_counter[index] += 1
                if wid == 0:
                    os.makedirs(os.path.join(output_dir, f'cluster_{index}'), exist_ok=True)
            with open(os.path.join(output_dir, f'cluster_{index}', f'data_{wid}.jsonl'), 'w', encoding='utf-8') as fp:
                for data in cluster_data[index]:
                    fp.write(json.dumps(data, ensure_ascii=False))
                    fp.write('\n')
                
            cluster_data[index] = []
            print('writing', index, wid)
                
    for index in cluster_data:
        if len(cluster_data[index]) == 0:
            continue
        with write_counter.get_lock():
            wid = write_counter[index]
            write_counter[index] += 1
            if wid == 0:
                os.makedirs(os.path.join(output_dir, f'cluster_{index}'), exist_ok=True)
        with open(os.path.join(output_dir, f'cluster_{index}', f'data_{wid}.jsonl'), 'w', encoding='utf-8') as fp:
            for data in cluster_data[index]:
                fp.write(json.dumps(data, ensure_ascii=False))
                fp.write('\n')
    print(f'FINISHED FOR RANK {tid}')
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', type=str)
    parser.add_argument('--data_dirs', type=str)
    parser.add_argument('--output_dirs', type=str)
    parser.add_argument('--num_clusters', type=int)
    args = parser.parse_args()
    
    global input_dirs, data_dir, output_dir
    input_dirs = args.input_dirs
    data_dir = args.data_dirs
    output_dir = args.output_dirs
    
    print('input_dir', input_dirs)
    print('output_dir', output_dir)
    
    n_read_workers = 64
    n_write_workers = 8
    max_length_per_file = 5000000
    input_root_dir = input_dirs
    emb_dir = os.path.join(input_root_dir, 'doc_embedding')
    label_dir = os.path.join(input_root_dir, 'kmeans_torch', f'n_cluster_{args.num_clusters}')
    mapping_dir = os.path.join(input_root_dir, 'doc_mapping')
    fn_list = list(sorted(os.listdir(emb_dir)))
    
    n_clusters = torch.load(os.path.join(label_dir, 'centroids.pt'), map_location='cpu').shape[0]
    os.makedirs(output_dir, exist_ok=True)
    
    chunk_size = math.ceil(len(fn_list) / n_read_workers)
    mapping_queue = Queue()
    data_queue = Queue()
    write_counter = Array('i', [0 for _ in range(n_clusters)])
    
    mapping_workers = []
    read_workers = []
    write_workers = []
    
    t = Process(target=read_mapping_worker, args=(0, fn_list, mapping_queue, mapping_dir, label_dir))
    t.start()
    mapping_workers.append(t)
    
    
    for i in range(n_read_workers):
        start_idx = i * chunk_size
        end_idx = min((i+1) * chunk_size, len(fn_list))
        if end_idx <= start_idx:
            break
        t = Process(target=read_worker, args=(i, mapping_queue, data_queue))
        t.start()
        read_workers.append(t)
    
    for i in range(n_write_workers):
        t = Process(target=write_worker, args=(i, data_queue, n_clusters, max_length_per_file, write_counter))
        t.start()
        write_workers.append(t)
    
    for workers in [mapping_workers, read_workers]:
        for w in workers:
            w.join()
    
    data_queue.put(None)
    
    for w in write_workers:
        w.join()
    
if __name__ == '__main__':
    main()

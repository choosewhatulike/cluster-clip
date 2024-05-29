import os
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import trange, tqdm
import json
import time
import torch.distributed as dist
from transformers import Pipeline, AutoModel, AutoTokenizer
import langid
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dirs', type=str)
parser.add_argument('--output_dirs', type=str)
parser.add_argument('--model_name', type=str, default='jinaai/jina-embeddings-v2-base-en')
parser.add_argument('--pooling_strategy', type=str, default='mean', help='mean, first')
parser.add_argument('--support_languages', type=str, default='all', help='en, zh, all')
parser.add_argument('--max_length', type=int, default=4096)
parser.add_argument('--batch_size', type=int, default=32)

args = parser.parse_args()

input_dirs = args.input_dirs
output_dirs = args.output_dirs
model_name = args.model_name
pooling_strategy = args.pooling_strategy
support_languages = args.support_languages
MAX_LENGTH = args.max_length
batch_size = args.batch_size
chunk_size = None
block_size = 102400

proxy_url = "100.66.27.20:3128"
def set_proxy(url=proxy_url):
    os.environ["http_proxy"] = url
    os.environ["https_proxy"] = url
    os.environ["HTTP_PROXY"] = url
    os.environ["HTTPS_PROXY"] = url
    
def unset_proxy():
    os.environ.pop('http_proxy', None)
    os.environ.pop('https_proxy', None)
    os.environ.pop('HTTP_PROXY', None)
    os.environ.pop('HTTPS_PROXY', None)

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as fp:
        for ex in data:
            print(json.dumps(ex, ensure_ascii=False), file=fp)
            
            
def traverse_jsonl_files(directory):
    # 遍历文件夹下的所有JSONL文件
    for root, _, files in os.walk(directory, followlinks=True):
        for file in files:
            if file.endswith('.jsonl'):
                file_path = os.path.join(root, file)
                yield file_path
                
                
class MultiProcJsonlDataset(IterableDataset):
    def __init__(self, input_dirs, log_dir, rank, world_size) -> None:
        super().__init__()
        self.input_dirs = input_dirs
        self.total_paths = list(traverse_jsonl_files(input_dirs))
        self.rank = rank
        self.world_size = world_size
        self.log_file = os.path.join(log_dir, f'rank{rank}.json')
            
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            worker_id = worker_info.id % worker_info.num_workers
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1
            
        rank = self.rank * num_workers + worker_id
        world_size = self.world_size * num_workers
            
        for file_path in self.total_paths:
            with open(self.log_file, 'a', encoding='utf-8') as fp:
                json.dump({'file_path': file_path, 'status': 'started'}, fp, ensure_ascii=False)
            fsize = os.path.getsize(file_path)
            chunck_size = fsize // world_size
            start_pos = chunck_size * rank
            end_pos = min(chunck_size * (rank + 1), fsize)
            with open(file_path, 'rb') as f:
                f.seek(start_pos)
                while f.tell() < end_pos:
                    start_pos = f.tell()
                    line = f.readline()
                    try:
                        data = json.loads(line.strip())
                    except Exception as e:
                        continue
                    obj = {
                        'file_path': file_path,
                        'start_pos': start_pos,
                        'end_pos': f.tell(),
                        'data': data,
                        'id': data['id'],
                    }
                    yield obj
                    
            with open(self.log_file, 'a', encoding='utf-8') as fp:
                json.dump({'file_path': file_path, 'status': 'finished'}, fp, ensure_ascii=False)
                

class TokenizeDataset(IterableDataset):
    def __init__(self, dataset, tokenizer) -> None:
        super().__init__()
        self.dataset = dataset
        self.rank = dataset.rank
        self.world_size = dataset.world_size
        self.total_paths = dataset.total_paths
        self.tokenizer = tokenizer
        
    def __iter__(self):
        for obj in self.dataset:
            data = obj.pop('data')
            if 'content' in data:
                text = data['content']
            else:
                text = data['prompt'] + data['output']
            language = langid.classify(text[:10000])[0]
            obj['language'] = language
            tokens = self.tokenizer.encode(text, truncation=True, max_length=MAX_LENGTH)
            obj['input_ids'] = tokens
            if support_languages == 'en' and obj['language'] != 'zh':
                yield obj
            elif support_languages == 'zh' and obj['language'] == 'zh':
                yield obj
            elif support_languages == 'all':
                yield obj
                
    def collate_fn(self, batch):
        max_length = max([len(x['input_ids']) for x in batch])
        pad_input_ids = torch.full((len(batch), max_length), fill_value=self.tokenizer.pad_token_id)
        pad_attn = torch.full((len(batch), max_length), fill_value=0)
        for i, x in enumerate(batch):
            pad_input_ids[i][:len(x['input_ids'])] = torch.tensor(x['input_ids'])
            pad_attn[i][:len(x['input_ids'])] = 1
        return {
            'input_ids': pad_input_ids,
            'attention_mask': pad_attn,
        }, batch
    
    
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def first_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    return token_embeddings[:, 0]
    
    
@torch.no_grad()
def encode_and_save_emb(data_loader, model, tokenizer, doc_embedding_dir, doc_mapping_dir):
    rank = dist.get_rank()
    local_rank = os.getenv('LOCAL_RANK', 0)
    world_size = dist.get_world_size()
    output_idx = 0
    
    mappings = []
    embeddings = []
    emb_path = os.path.join(doc_embedding_dir, f'{output_idx}-world{world_size}-rank{rank}.pt')
    pooling_fn = mean_pooling if pooling_strategy == 'mean' else first_pooling
    
    for batch_idx, (encoded_input, batch) in tqdm(enumerate((data_loader)), disable=rank!=0, desc='encoding'):
        # input_ids = []
        # attention_mask = []
        for emb_idx, item in enumerate(batch):
            mappings.append({
                'language': item['language'],
                'file_path': item['file_path'],
                'start_pos': item['start_pos'],
                'end_pos': item['end_pos'],
                'emb_path': emb_path,
                'emb_local_idx': len(mappings),
                'id': item['id']
            })
        encoded_input['input_ids'] = encoded_input['input_ids'].to(torch.device(f'cuda:{local_rank}'))
        encoded_input['attention_mask'] = encoded_input['attention_mask'].to(torch.device(f'cuda:{local_rank}'))
        outputs = model(**encoded_input)
        batch_embeddings = pooling_fn(outputs, encoded_input['attention_mask'])
        batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        embeddings.append(batch_embeddings)
        if len(mappings) >= block_size:
            save_jsonl(mappings, os.path.join(doc_mapping_dir, f'{output_idx}-world{world_size}-rank{rank}.jsonl'))
            embeddings = torch.cat(embeddings, dim=0)
            torch.save(embeddings, emb_path)
            output_idx += 1
            emb_path = os.path.join(doc_embedding_dir, f'{output_idx}-world{world_size}-rank{rank}.pt')
            mappings= []
            embeddings = []
    if len(mappings):
        save_jsonl(mappings, os.path.join(doc_mapping_dir, f'{output_idx}-world{world_size}-rank{rank}.jsonl'))
        embeddings = torch.cat(embeddings, dim=0)
        torch.save(embeddings, emb_path)
   
def main():
    doc_mapping_dir = os.path.join(output_dirs, 'doc_mapping')
    doc_embedding_dir = os.path.join(output_dirs, 'doc_embedding')
    log_dir = os.path.join(output_dirs, 'progress')
    os.makedirs(doc_mapping_dir, exist_ok=True)
    os.makedirs(doc_embedding_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    local_rank = os.getenv('LOCAL_RANK', 0)
    print(f"Initialzing rank:{os.getenv('RANK', 0)} world_size:{os.getenv('WORLD_SIZE', 1)}, local_rank: {local_rank}")
    dist.init_process_group(backend='nccl', init_method='env://')
    
    # uncomment the following lines if you need to set proxy
    set_proxy()
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    unset_proxy()
    
    model.to(torch.device(f'cuda:{local_rank}'))
    
    dataset = MultiProcJsonlDataset(input_dirs, log_dir, dist.get_rank(), dist.get_world_size())
    dataset = TokenizeDataset(dataset, tokenizer)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False, drop_last=False, pin_memory=True, collate_fn=dataset.collate_fn)
    
    if dist.get_rank() == 0:
        print('Args:')
        print(f'input_dirs: {input_dirs}')
        print(f'output_dirs: {output_dirs}')
        print(f'batch_size: {batch_size}')
        print(f'max_length: {MAX_LENGTH}')
        print(f'chunk_size: {chunk_size}')
        print(f'block_size: {block_size}')
        print(f'model_name: {model_name}')
        print(f'pooling_strategy: {pooling_strategy}')
        print('Num of files:', len(dataset.total_paths))
        print('================== Total File List ==================')
        print(dataset.total_paths)
    
    encode_and_save_emb(data_loader, model, tokenizer, doc_embedding_dir, doc_mapping_dir)
    print(f"Finished rank:{os.getenv('RANK', 0)} world_size:{os.getenv('WORLD_SIZE', 1)}, local_rank: {local_rank}")
    dist.barrier()
    
if __name__ == '__main__':
    main()

import argparse
import os
import random
from pathlib import Path
from typing import List, Optional, Union

import torch
from tqdm import tqdm
import pandas as pd

# from transformers import BertTokenizer
from transformers_cutoff import AutoTokenizer, AutoConfig, glue_tasks_num_labels
from transformers_cutoff.data.processors.glue import glue_processors
from transformers_cutoff.data.processors.utils import InputFeatures
from transexp_orig.RobertaForSequenceClassification import RobertaForSequenceClassification

from transexp_orig.ExplanationGenerator import Generator
from transexp_orig.BertForSequenceClassification import BertForSequenceClassification

TASKS = ["CoLA", "SST-2", "MRPC", "QQP", "STS-B", "MNLI", "QNLI", "RTE", "WNLI"]
random.seed(1)

def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=Path, 
                      default=Path('/home/jovyan/work/datasets'))
  parser.add_argument('--pretrain_dir', type=Path, 
                      default=Path('/home/jovyan/work/checkpoint'))
  parser.add_argument('--save_dir', type=Path, 
                      default=Path('./cutoff_idx'))
  parser.add_argument('--tasks', help='tasks to download data for as a comma separated string',
                      type=str, default='all')
  parser.add_argument('--cutoff_ratio', help='cutoff ratio',
                      type=float, default=0.1)
  parser.add_argument('--gpu', help='gpu number',
                      type=str, default="1")
  
  return parser.parse_args()


def main():
  args = parse()
  
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
  os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
  
  tasks = TASKS if args.tasks == "all" else args.tasks.replace(" ", "").split(",")
  tokenizer = AutoTokenizer.from_pretrained("roberta-base")
  
  # cls, pad, sep, unk, period
  # TOKENS_EXCLUDE = list(set([
  #   tokenizer.cls_token_id,
  #   tokenizer.pad_token_id,
  #   tokenizer.sep_token_id,
  #   tokenizer.unk_token_id,
  #   tokenizer.bos_token_id,
  #   tokenizer.eos_token_id,
  #   tokenizer.convert_tokens_to_ids(".")]))
  TOKENS_EXCLUDE = [0,1,2,3,4]
  
  os.makedirs(str(args.save_dir), exist_ok=True)
  
  for task in tasks:
    task = task.upper()
    if task == "COLA":
      task = "CoLA"
      
    print("+"*20 + task + "+"*20)
    
    pretrain_path = str(args.pretrain_dir / task / "checkpoint_token/")
    print(f"Get pretrained model from {pretrain_path}")  
    # tokenizer = AutoTokenizer.from_pretrained(f"textattack/roberta-base-{task}")
    model = RobertaForSequenceClassification.from_pretrained(
              pretrain_path,
            ).to("cuda")
    model.eval()
      
    processor = glue_processors[task.lower()]()
    data_dir = str(args.data_dir / task)
    examples = (processor.get_train_examples(data_dir))
    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples], max_length=128, pad_to_max_length=True, #truncation=True
    )
    
    label_list = processor.get_labels()
    label_map = {label: i for i, label in enumerate(label_list)}
    
    num_labels = glue_tasks_num_labels[task.lower()]
    if task == "STS-B":
      labels = [float(example.label) for example in examples]
      print("label: continuous label")
    else:
      labels = [label_map[example.label] for example in examples]
      print(f"label: {label_list}")
    print()
    # labels = [label_map[example.label] for example in examples]

    features = []
    for i in range(len(examples)):
      inputs = {k: batch_encoding[k][i] for k in batch_encoding}

      feature = InputFeatures(**inputs, example_index=i, label=labels[i])
      features.append(feature)
      
    
    explanations = Generator(model)
    
    cutoff_indices_exclude = []
    cutoff_indices_all_tokens = []
    example_indices = []
    cutoff_tokens=[]
    print("Compute attribution")
    print(f"len features: {len(features)}")
    for i in range(10):
    # for i in tqdm(range(len(features))):
      input_ids = torch.tensor(features[i].input_ids, dtype=int).reshape(1,-1).cuda()
      attention_mask = torch.tensor(features[i].attention_mask, dtype=torch.float32).reshape(1,-1).cuda()
      example_indices.append(features[i].example_index)
      
      expl = explanations.generate_LRP(
              input_ids=input_ids, 
              attention_mask=attention_mask, 
              start_layer=0
            )[0]
      # with torch.no_grad():
      # normalize scores
      
      input_len = int(attention_mask.sum())
      expl = expl[:input_len]
      print(expl.shape)
      expl = (expl - expl.min()) / (expl.max() - expl.min())
      
      cutoff_ratio = int(input_len*args.cutoff_ratio)
      if cutoff_ratio <= 1:
        cutoff_ratio = 2
      
      lowest_indices_all_token = torch.topk(expl, cutoff_ratio, 0, largest=False).indices
      cutoff_indices_all_tokens.append(lowest_indices_all_token.tolist())
      
      expl_excluded = []
      
      for i in range(input_len):
        if input_ids[0][i] in TOKENS_EXCLUDE:
        # if input_ids[0][i] in [0,1,2,3,4]:
          expl_excluded.append(100)
        else:
          expl_excluded.append(expl[i])
          
      expl_excluded = torch.tensor(expl_excluded)
      

      lowest_indices = torch.topk(expl_excluded, cutoff_ratio, 0, largest=False).indices
      cutoff_indices_exclude.append(lowest_indices.tolist())
      
      print(torch.topk(expl_excluded, cutoff_ratio, 0, largest=False).values)
      print(lowest_indices)
      expl_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][lowest_indices])
      print([s[1:] for s in expl_tokens])
      cutoff_tokens.append(expl_tokens)
      
      del input_ids, attention_mask, expl
      
    pd.DataFrame({
      "example_index": example_indices, 
      "cutoff_idx_excluded": cutoff_indices_exclude,
      "cutoff_idx_all_tokens": cutoff_indices_all_tokens,
      "cutoff_tokens": cutoff_tokens,
    }).to_csv(str(args.save_dir / task)+".tsv", index=False, sep="\t")
    print(f"Saved: {task}")
    print()

      
  
if __name__=="__main__":
  main()
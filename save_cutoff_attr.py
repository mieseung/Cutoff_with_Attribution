import argparse
import os
from pathlib import Path
from typing import List, Optional, Union

import torch
from tqdm import tqdm
import pandas as pd

# from transformers import BertTokenizer
from transformers_cutoff import AutoTokenizer
from transformers_cutoff.data.processors.glue import glue_processors
from transformers_cutoff.data.processors.utils import InputFeatures
from transformers_cutoff import RobertaForSequenceClassification

from transexp_orig.ExplanationGenerator import Generator

TASKS = ["CoLA", "SST-2", "MRPC", "QQP", "STS-B", "MNLI", "QNLI", "RTE", "WNLI"]
TOKENS_EXCLUDE = [0, 1, 2, 3, 4] # cls, pad, sep, unk, period


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

  os.makedirs(str(args.save_dir), exist_ok=True)
  
  for task in tasks:
    task = task.upper()
    if task == "COLA":
      task = "CoLA"
      
    print("+"*20 + task + "+"*20)
      
    processor = glue_processors[task.lower()]()
    data_dir = str(args.data_dir / task)
    examples = (processor.get_train_examples(data_dir))
    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples], max_length=128, pad_to_max_length=True, #truncation=True
    )
    
    label_list = processor.get_labels()
    label_map = {label: i for i, label in enumerate(label_list)}
    
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

      feature = InputFeatures(**inputs, label=labels[i])
      features.append(feature)
      
    print("Get pretrained model")  
    model = RobertaForSequenceClassification.from_pretrained(
              str(args.pretrain_dir / task / "checkpoint_token"),
            ).to("cuda")
    model.eval()
    explanations = Generator(model)
    
    cutoff_indices = []
    
    print("Compute attribution")
    for i in tqdm(range(len(features))):
      input_ids = torch.tensor(features[i].input_ids, dtype=int).reshape(1,-1).cuda()
      attention_mask = torch.tensor(features[i].attention_mask, dtype=torch.float32).reshape(1,-1).cuda()

      expl = explanations.generate_LRP(
              input_ids=input_ids, 
              attention_mask=attention_mask, 
              start_layer=0
            )[0]
      # with torch.no_grad():
      # normalize scores
      expl = (expl - expl.min()) / (expl.max() - expl.min())
      
      input_len = int(attention_mask.sum())
      
      expl_excluded = []
      for i in range(input_len):
        if input_ids[0][i] in [0,1,2,3,4]:
          expl_excluded.append(100)
        else:
          expl_excluded.append(expl[i])
          
      expl_excluded = torch.tensor(expl_excluded)
      
      cutoff_ratio = int(input_len*args.cutoff_ratio)
      if cutoff_ratio < 1:
        cutoff_ratio = 1
        
      lowest_indices = torch.topk(expl_excluded, cutoff_ratio, 0, largest=False).indices
      cutoff_indices.append(lowest_indices.tolist())
      
      del input_ids, attention_mask, expl
      
    pd.DataFrame({"idx": range(len(features)), "cutoff_idx": cutoff_indices}).to_csv(str(args.save_dir / task)+".tsv", index=False, sep="\t")
    print(f"Saved: {task}")
    print()

      
  
if __name__=="__main__":
  main()
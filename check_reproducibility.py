from transexp_orig.BertForSequenceClassification import BertForSequenceClassification
from transexp_orig.ExplanationGenerator import Generator
from transformers_cutoff.data.processors.utils import InputFeatures
from transformers_cutoff.data.processors.glue import glue_processors
from transformers import RobertaTokenizer

import torch

if __name__ == '__main__':
    model = BertForSequenceClassification.from_pretrained("roberta-base").to("cuda")
    model.eval()
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    explanations = Generator(model)
    
    
    # WNLI
    processor = glue_processors["wnli"]()
    data_dir = '/home/jovyan/work/datasets/WNLI'
    examples = processor.get_train_examples(data_dir)
    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples], max_length=128, pad_to_max_length=True, #truncation=True
    )
    
    label_list = processor.get_labels()
    label_map = {label: i for i, label in enumerate(label_list)}
    labels = [label_map[example.label] for example in examples]
    
    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeatures(**inputs, example_index=i, label=labels[i])
        features.append(feature)
        
    
    for i in range(3):
        attention_mask = torch.tensor(features[i].attention_mask, dtype=torch.float32).unsqueeze(0).to('cuda')
        input_len = int(torch.sum(attention_mask).item())
                
        expl_attention_mask = torch.tensor(features[i].attention_mask[:input_len], dtype=torch.float32).unsqueeze(0).to('cuda')
        expl_input_ids = torch.tensor(features[i].input_ids[:input_len], dtype=int).unsqueeze(0).to('cuda')
        
        expl = explanations.generate_LRP(
            input_ids = expl_input_ids,
            attention_mask = expl_attention_mask,
            start_layer=0
        )[0]
        
        expl = (expl - expl.min()) / (expl.max() - expl.min())
        
        print(f" {i} : {expl}\n")
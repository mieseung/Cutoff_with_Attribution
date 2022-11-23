import argparse
import numpy as np
import torch
import glob
import time
import copy

# compute rollout between attention layers
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    
    joint_attention_cp = joint_attention.data.cpu()
        
    del all_layer_matrices
    del eye
    del matrices_aug
    del joint_attention
    torch.cuda.empty_cache()
    
    return joint_attention_cp

class Generator:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def generate_LRP(self, input_ids, attention_mask,
                     index=None, start_layer=11):
        
        # print("\n---------------------------")
        # print(torch.cuda.memory_allocated())
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0] # 234299392 -> 3260589568
        kwargs = {"alpha": 1}
        # print("self.model done")
        # print(torch.cuda.memory_allocated())
        
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        # print("make a one hot vector")
        # print(torch.cuda.memory_allocated())

        self.model.zero_grad()
        # one_hot.backward(create_graph=True, retain_graph=False)
        one_hot.backward(retain_graph=True)
        # print("back propagation with graph")
        # print(torch.cuda.memory_allocated())

        self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)
        # print("relprop")
        # print(torch.cuda.memory_allocated())
        torch.cuda.empty_cache()

        cams = []
        blocks = self.model.bert.encoder.layer
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients()
            cam = blk.attention.self.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cams.append(cam.unsqueeze(0))
            del grad
            del cam
        
        # print("calculate cams [ DONE ]")
        # print(torch.cuda.memory_allocated())
            
        rollout = compute_rollout_attention(cams, start_layer=start_layer)
        rollout[:, 0, 0] = rollout[:, 0].min()
        rollout_copy = rollout.data.cpu()
        
        one_hot.backward(retain_graph=False) # to remove retained graph
        del one_hot
        del one_hot_vector
        del rollout
        del cams
        torch.cuda.empty_cache()
        # print("rollout attention [ DONE ] ")
        # print(torch.cuda.memory_allocated())
        
        # print("++++++++++++++++++++++++")
        
        return rollout_copy[:, 0]

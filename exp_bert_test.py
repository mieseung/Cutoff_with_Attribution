import torch

from transformers_cutoff.modeling_exp_bert import BertSelfAttention

if __name__ == '__main__':
    class Config:
      def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.output_attentions = True

    model = BertSelfAttention(Config(1024, 4, 0.1))
    x = torch.rand(2, 20, 1024)
    x.requires_grad_()

    model.eval()

    y = model.forward(x)
    
    kwargs = {"alpha" : 0.1}

    relprop = model.relprop(torch.rand(2, 20, 1024), **kwargs)

    print(relprop[1][0].shape)
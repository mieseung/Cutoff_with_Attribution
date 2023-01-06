# Cutoff_with_Attribution
2022-2 창의적 통합 설계 1 LDI LAB

### Activating virtual environment
```shell
$ conda activate cutoff   # activate
$ conda deactivate
```

### Training
```shell
$ ./run_glue {dataset name} {GPU number} {train batch size} {token type}
# ./run_glue CoLA 0 16 token_exp
```

### Cutoff types
```
span, token, dim, token_exp
```

## Subprojects
**Attention Attribution + Cutoff** [AttAttr-Cutoff](https://github.com/footprinthere/AttAttr-Cutoff)

## References
**Cutoff** [github](https://github.com/dinghanshen/Cutoff)

**Attention Attribution** [github](https://github.com/YRdddream/attattr)

**Transformer Explainability** [github](https://github.com/hila-chefer/Transformer-Explainability) / [Colab](https://colab.research.google.com/github/hila-chefer/Transformer-Explainability/blob/main/BERT_explainability.ipynb)

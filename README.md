## ProteinBERT - Pytorch (wip)

Implementation of <a href="https://www.biorxiv.org/content/10.1101/2021.05.24.445464v1">ProteinBERT</a> in Pytorch.

<a href="https://github.com/nadavbra/protein_bert">Original Repository</a>

## Install

```bash
$ pip install protein-bert-pytorch
```

## Usage

```python
import torch
from protein_bert_pytorch import ProteinBERT

model = ProteinBERT(
    num_tokens = 21,
    num_annotation = 8943,
    dim = 512,
    dim_global = 256,
    depth = 6,
    narrow_conv_kernel = 9,
    wide_conv_kernel = 9,
    wide_conv_dilation = 5,
    attn_heads = 8,
    attn_dim_head = 64
)

seq = torch.randint(0, 21, (2, 2048))
mask = torch.ones(2, 2048).bool()
annotation = torch.randint(0, 1, (2, 8943)).float()

seq_logits, annotation_logits = model(seq, annotation, mask = mask) # (2, 2048, 21), (2, 8943)
```

## Citations

```bibtex
@article {Brandes2021.05.24.445464,
    author      = {Brandes, Nadav and Ofer, Dan and Peleg, Yam and Rappoport, Nadav and Linial, Michal},
    title       = {ProteinBERT: A universal deep-learning model of protein sequence and function},
    year        = {2021},
    doi         = {10.1101/2021.05.24.445464},
    publisher   = {Cold Spring Harbor Laboratory},
    URL         = {https://www.biorxiv.org/content/early/2021/05/25/2021.05.24.445464},
    eprint      = {https://www.biorxiv.org/content/early/2021/05/25/2021.05.24.445464.full.pdf},
    journal     = {bioRxiv}
}
```

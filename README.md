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

To use for pretraining

```python
import torch
from protein_bert_pytorch import ProteinBERT, PretrainingWrapper

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
    attn_dim_head = 64,
    local_to_global_attn = False,
    local_self_attn = True,
    num_global_tokens = 2,
    glu_conv = False
)

learner = PretrainingWrapper(
    model,
    random_replace_token_prob = 0.05,    # what percentage of the tokens to replace with a random one, defaults to 5% as in paper
    remove_annotation_prob = 0.25,       # what percentage of annotations to remove, defaults to 25%
    remove_all_annotations_prob = 0.5,   # what percentage of batch items to remove annotations for completely, defaults to 50%
    seq_loss_weight = 1.,                # weight on loss of sequence
    annotation_loss_weight = 1.,         # weight on loss of annotation
    exclude_token_ids = (0, 1, 2)        # for excluding padding, start, and end tokens from being masked
)

# do the following in a loop for a lot of sequences and annotations

seq        = torch.randint(0, 21, (2, 2048))
annotation = torch.randint(0, 1, (2, 8943)).float()
mask       = torch.ones(2, 2048).bool()

loss = learner(seq, annotation, mask = mask) # (2, 2048, 21), (2, 8943)
loss.backward()

# save your model and evaluate it

torch.save(model, './improved-protein-bert.pt')
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

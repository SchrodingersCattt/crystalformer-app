
[![arXiv](https://img.shields.io/badge/arXiv-2403.15734-b31b1b.svg)](https://arxiv.org/abs/2403.15734)

_CrystalFormer_ is a transformer-based autoregressive model specifically designed for space group-controlled generation of crystalline materials. The space group symmetry significantly simplifies the
crystal space, which is crucial for data and compute efficient generative modeling of crystalline materials.


## Installation


```bash
  pip install .
```

Then command `crystalgpu-app` will create a gradio link.


## Model card

The model is an autoregressive transformer for the space group conditioned crystal probability distribution `P(C|g) = P (W_1 | ... ) P ( A_1 | ... ) P(X_1| ...) P(W_2|...) ... P(L| ...)`, where

- `g`: space group number 1-230
- `W`: Wyckoff letter ('a', 'b',...,'A')
- `A`: atom type ('H', 'He', ..., 'Og')
- `X`: factional coordinates
- `L`: lattice vector [a,b,c, alpha, beta, gamma]
- `P(W_i| ...)` and `P(A_i| ...)`  are categorical distributuions.
- `P(X_i| ...)` is the mixture of von Mises distribution.
- `P(L| ...)`  is the mixture of Gaussian distribution.

We only consider symmetry inequivalent atoms. The remaining atoms are restored based on the space group and Wyckoff letter information. Note that there is a natural alphabetical ordering for the Wyckoff letters, starting with 'a' for a position with the site-symmetry group of maximal order and ending with the highest letter for the general position. The sampling procedure starts from higher symmetry sites (with smaller multiplicities) and then goes on to lower symmetry ones (with larger multiplicities). Only for the cases where discrete Wyckoff letters can not fully determine the structure, one needs to further consider factional coordinates in the loss or sampling.



```bibtex
@misc{cao2024space,
      title={Space Group Informed Transformer for Crystalline Materials Generation}, 
      author={Zhendong Cao and Xiaoshan Luo and Jian Lv and Lei Wang},
      year={2024},
      eprint={2403.15734},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci}
}
```


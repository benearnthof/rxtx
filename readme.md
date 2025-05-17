# RXTX: XXᵀ Can Be Faster

This repository provides a reference implementation of the RXTX algorithm as presented in [https://arxiv.org/pdf/2505.09814](https://arxiv.org/pdf/2505.09814)

---

## Reference

This implementation is based on the following paper:

> **"XXᵀ Can Be Faster"**  
> Dmitry Rybin, Yushun Zhang, and Zhi-Quan Luo  
> The Chinese University of Hong Kong, Shenzhen  
> Shenzhen Research Institute of Big Data  
> *May 16, 2025*  
>  
> [arXiv:2505.09814](https://arxiv.org/pdf/2505.09814)


---

## Features

- 4×4 block implementation of RXTX
- Supports arbitrary square matrices (zero-padded if needed)

---

## Requirements

- Python 3.8+
- `numpy >= 1.24`
- tensorhue >= 0.1.0 (For quick visualization in terminal, not needed for algorithm.)

Install via:

```bash
pip install -e .
```

## Citation
This repo does not modify or extend the RXTX algorithm beyond reproducing it in code. All credit goes to the original authors. Please cite their paper if using this code in academic work:

```bibtex
@misc{rybin2025xxt,
  title={XXᵀ Can Be Faster},
  author={Dmitry Rybin and Yushun Zhang and Zhi-Quan Luo},
  year={2025},
  eprint={2505.09814},
  archivePrefix={arXiv},
  primaryClass={cs.NA}
}
```

### A Simple Codebase for Clothes-Changing Person Re-identification.
####  [Cloth-aware Augmentation for Cloth-generalized Person Re-identification (ACMMM, 2024)]
#### We provide the essential code snippets for domain augmentation and feature augmentation, which can be easily integrated into various baseline methods to enhance their generalization ability. We give an example code integrated with CAL(Clothes-Changing Person Re-identification with RGB Modality Only (CVPR, 2022)).

#### Requirements
- Python 3.9
- Pytorch 1.11.0
- yacs
- apex
- yaml
- h5py
- scipy

#### Get Started
- Replace `_C.DATA.ROOT` and `_C.OUTPUT` in `configs/default_img.py&default_vid.py`with your own `data path` and `output path`, respectively.
- Run `script.sh`


#### Citation

If you use our code in your research or wish to refer to the baseline results, please use the following BibTeX entry.
    
    @inproceedings{liu2022CaAug,
        title={Cloth-aware Augmentation for Cloth-generalized Person Re-identification},
        author={Liu, Fangyi and Ye, Mang and Du, Bo},
        booktitle={ACMMM},
        year={2024},
    }

#### Related Repos

- [Simple-ReID](https://github.com/guxinqian/Simple-ReID)
- [fast-reid](https://github.com/JDAI-CV/fast-reid)
- [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)
- [Pytorch ReID](https://github.com/layumi/Person_reID_baseline_pytorch)
- [CAL](https://github.com/guxinqian/Simple-CCReID)


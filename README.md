# Probabilistic Uncertainty Modeling (PUM)

Gengcong Yang*, Jingyi Zhang*, Yong Zhang, Baoyuan Wu, and Yujiu Yang, [Probabilistic Modeling of Semantic Ambiguity for Scene Graph Generation](https://arxiv.org/abs/2103.05271), CVPR, 2021. (* co-first authors)

This repository contains PyTorch version code for the proposed **Probabilistic Uncertainty Modeling (PUM)** in the above paper.

## Setup and usage

1. Refer to [neural-motifs](https://github.com/rowanz/neural-motifs) for the installation, except that we use PyTorch 0.4.1 instead of 0.3.0 here. That is, use `conda install pytorch=0.4.1 torchvision=0.2.0 cuda90 -c pytorch` in the first step. You may also need to use `pip install -r requirements.txt` to install additional third-party modules.
1. Train a model with PUM by such a command:
    ```bash
    python models/train_rels.py
        -m predcls  # choose a task from 'predcls', 'sgcls' and 'sgdet'
        -ckpt path/to/vg-faster-rcnn.tar  # path to the pre-trained checkpoint
        -b 8  # batch size
        -ngpu 1  # number of GPUs
        -model motifs  # choose a model from 'motifs', 'imp', 'kern' and 'vctree'
        -visual_gaussian  # apply PUM
        -run_name predcls-motifs-vis_gaussian  # name of this training
    ```
1. Evaluate the model by such a command:
    ```bash
    python models/eval_rels.py
        -m predcls  # choose a task from 'predcls', 'sgcls' and 'sgdet'
        -ckpt path/to/ckpt.tar  # path to the saved checkpoint after training
        -ngpu 1  # number of GPUs
        -model motifs  # choose a model from 'motifs', 'imp', 'kern' and 'vctree'
        -visual_gaussian  # apply PUM
        -run_name predcls-motifs-vis_gaussian  # name of this training
    ```

## Acknowledgments

The codes borrowed a lot from [neural-motifs](https://github.com/rowanz/neural-motifs), [kern](https://github.com/yuweihao/KERN) and [vctree](https://github.com/KaihuaTang/VCTree-Scene-Graph-Generation). We thank for their releasing nice codes.

## Citation

If the paper significantly inspires you, please cite our work.
```
@inproceedings{yang2021probabilistic,
  title={Probabilistic Modeling of Semantic Ambiguity for Scene Graph Generation},
  author={Yang, Gengcong and Zhang, Jingyi and Zhang, Yong and Wu, Baoyuan and Yang, Yujiu},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```
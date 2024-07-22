## Low-Rank Knowledge Decomposition for Medical Foundation Models


This repository is an official PyTorch implementation of "Low-Rank Knowledge Decomposition for Medical Foundation Models"

### Download Dataset

- RadImageNet  [here](https://github.com/BMEII-AI/RadImageNet).
- MedMNIST  [here](https://github.com/MedMNIST/MedMNIST).


### Run


For example, you can run the following command for LoRKD training.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_lora_shufflenet_medmnist.py
```


## Citation

If you find ``LoRKD`` useful for your research or development, please cite the following:

```latex
@inproceedings{zhou2024low,
  title={Low-Rank Knowledge Decomposition for Medical Foundation Models},
  author={Zhou, Yuhang and Li, Haolin and Du, Siyuan and Yao, Jiangchao and Zhang, Ya and Wang, Yanfeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11611--11620},
  year={2024}
}
```

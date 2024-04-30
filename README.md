# Lipreading

A reporitory for lipreading on the [lipreading in the wild (LRW) dataset](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) using the SpotFast Networks (ICONIP 2020). 
The SpotFast Networks utilize temporal window, two input pathways with lateral connections and two memory-augmented transformers to recognize word-level lip movements. The test accuracy is 84.4%. For comparisons with other methods, please consider [PapersWithCode](https://paperswithcode.com/sota/lipreading-on-lip-reading-in-the-wild). The models (most in this task) are sensitive to random seeds as far as I feel. Mixing random seeds between epoch cycles improves to results surprisingly. Since the dataset is mostly large scale and there is an affiliation wall on this task, I personally do not see anyone doing any insightful example/model mining on this issue. It takes lots of computational powers also even if we may filter or minimize the dataset.

Wiriyathammabhum, Peratham. "SpotFast Networks with Memory Augmented Lateral Transformers for Lipreading." 
International Conference on Neural Information Processing. Springer, Cham, 2020.

## Citation
A link to the [paper](https://link.springer.com/chapter/10.1007/978-3-030-63820-7_63) and its [ArXiv](https://arxiv.org/abs/2005.10903).

```bixtex
@inproceedings{wiriyathammabhum2020spotfast,
  title={SpotFast Networks with Memory Augmented Lateral Transformers for Lipreading},
  author={Wiriyathammabhum, Peratham},
  booktitle={International Conference on Neural Information Processing},
  pages={554--561},
  year={2020},
  organization={Springer}
}
```

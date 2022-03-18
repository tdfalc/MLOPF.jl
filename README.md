# Machine Learning Assisted OPF
MLOPF.jl is a Julia package used as a basis for research concerning machine learning assisted optimal power flow. 
## Outstanding Tasks
- [x] Implement concurrent sampler
- [x] Implement regression setting
- [x] Implement classification setting
- [x] Implement fully-connected neural network
- [ ] Implement convolutional neural network
- [ ] Implement graph neural networks
- [ ] Add locality analysis
- [ ] Add prediction time analysis
- [ ] Add convergence plots
- [ ] Add dataframe -> latex tables script
- [x] Add .gitignore to bypass cache files

## Citation
If you find MLOPF.jl useful in your work, we kindly request that you cite the following [publication](https://arxiv.org/abs/2110.00306):
```
@misc{falconer2022leveraging,
      title={Leveraging power grid topology in machine learning assisted optimal power flow}, 
      author={Thomas Falconer and Letif Mones},
      year={2022},
      eprint={2110.00306},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
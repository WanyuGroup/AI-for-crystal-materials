# AI-for-crystal-materials： models and benchmarks
Here we have collected papers with the theme of "AI for crystalline materials" that have appeared at top machine learning conferences and journals (ICML, ICLR, NeurIPS, AAAI, NPJ, NC, etc.) in recent years.

## Crystalline Material Physicochemical Property Prediction

|Method         |           Paper            |
|----------------|-------------------------------|
|SchNet|    Schnet: A continuous-filter convolutional neural network for modeling quantum interactions (NeurIPS2017) [[**Paper**](https://proceedings.neurips.cc/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html)][[**Code**](https://github.com/atomistic-machine-learning/schnetpack)]       |         
|CGCNN          |    Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties (Physical Review Letters, 2018) [[**Paper**](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301)][[**Code**](https://github.com/txie-93/cgcnn)]        |     
|MEGNET          | Graph networks as a universal machine learning framework for molecules and crystals (Chemistry of Materials, 2019) [[**Paper**](https://pubs.acs.org/doi/10.1021/acs.chemmater.9b01294)][[**Code**](https://github.com/materialsvirtuallab/megnet)]     | 
|GATGNN          | Graph convolutional neural networks with global attention for improved materials property prediction (Physical Chemistry Chemical Physics, 2020) [[**Paper**](https://pubs.rsc.org/en/content/articlelanding/2020/cp/d0cp01474e/unauth)][[**Code**](https://github.com/superlouis/GATGNN)]     | 
|ALIGNN          | Atomistic line graph neural network for improved materials property predictions (npj Computational Materials, 2021) [[**Paper**](https://www.nature.com/articles/s41524-021-00650-1)][[**Code**](https://github.com/usnistgov/alignn)]     | 
|ECN          | Equivariant networks for crystal structures (NeurIPS2022) [[**Paper**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1abed6ee581b9ceb4e2ddf37822c7fcb-Abstract-Conference.html)][[**Code**](https://github.com/oumarkaba/equivariant_crystal_networks)]     | 
|PotNet          | Efficient Approximations of Complete Interatomic Potentials for Crystal Property Prediction (ICML2023) [[**Paper**](https://proceedings.mlr.press/v202/lin23m.html)][[**Code**](https://github.com/divelab/AIRS/tree/main/OpenMat/PotNet)]     | 
|CrysGNN          | Crysgnn: Distilling pre-trained knowledge to enhance property prediction for crystalline materials (AAAI2023) [[**Paper**](https://ojs.aaai.org/index.php/AAAI/article/view/25892)][[**Code**](https://github.com/kdmsit/crysgnn)]     | 
|ETGNN          | A general tensor prediction framework based on graph neural networks (The Journal of Physical Chemistry Letters, 2023) [[**Paper**](https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.3c01200)]     | 
|GMTNet          |  A Space Group Symmetry Informed Network for O(3) Equivariant Crystal Tensor Prediction (IMCL2024) [[**Paper**](https://openreview.net/forum?id=BOFjRnJ9mX)][[**Code**](https://github.com/divelab/AIRS/tree/main/OpenMat/GMTNet)]     | 
|CEGANN          | CEGANN: Crystal Edge Graph Attention Neural Network for multiscale classification of materials environment (npj Computational Materials, 2023) [[**Paper**](https://www.nature.com/articles/s41524-023-00975-z)][[**Code**](https://github.com/sbanik2/CEGANN)]     | 
|ComFormer          | Complete and Efficient Graph Transformers for Crystal Material Property Prediction (ICLR2024) [[**Paper**](https://openreview.net/forum?id=BnQY9XiRAS)][[**Code**](https://github.com/divelab/AIRS/tree/main/OpenMat/ComFormer)]     | 
|Crystalformer          |Crystalformer: infinitely connected attention for periodic structure encoding (ICLR2024) [[**Paper**](https://openreview.net/pdf?id=BnQY9XiRAS)][[**Code**](https://github.com/omron-sinicx/crystalformer)]     | 
|Crystalformer          | Conformal Crystal Graph Transformer with Robust Encoding of Periodic Invariance (AAAI2024) [[**Paper**](https://ojs.aaai.org/index.php/AAAI/article/view/27781)]   | 
|E(3)NN          | Direct prediction of phonon density of states with Euclidean neural networks (Advanced Science, 2021) [[**Paper**](https://onlinelibrary.wiley.com/doi/full/10.1002/advs.202004214)][[**Code**](https://github.com/zhantaochen/phonondos_e3nn)]     | 
|DOSTransformer          | Density of States Prediction of Crystalline Materials via Prompt-guided Multi-Modal Transformer (NeurIPS2023) [[**Paper**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c23fdcb9f8e28af705a87de1375a705c-Abstract-Conference.html)][[**Code**](https://github.com/HeewoongNoh/DOSTransformer)]     | 
|Matformer          | Periodic Graph Transformers for Crystal Material Property Prediction (NeurIPS2022) [[**Paper**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6145c70a4a4bf353a31ac5496a72a72d-Abstract-Conference.html)][[**Code**](https://github.com/YKQ98/Matformer)]     | 
|CrysDiff          | A Diffusion-Based Pre-training Framework for Crystal Property Prediction (AAAI2024) [[**Paper**](https://ojs.aaai.org/index.php/AAAI/article/view/28748)]     | 
|MOFTransformer        | A multi-modal pre-training transformer for universal transfer learning in metal-organic frameworks (Nature Machine Intelligence, 2023) [[**Paper**](https://www.nature.com/articles/s42256-023-00628-2)][[**Code**](https://github.com/hspark1212/MOFTransformer)]     | 
|Uni-MOF          | A comprehensive transformer-based approach for high-accuracy gas adsorption predictions in metal-organic frameworks (Nature Communications, 2024) [[**Paper**](https://www.nature.com/articles/s41467-024-46276-x)][[**Code**](https://github.com/dptech-corp/Uni-MOF)]     | 



## Crystalline Material Synthesis

|Method         |           Paper            |
|----------------|-------------------------------|
|G-SchNet         | Symmetry-adapted generation of 3d point sets for the targeted discovery of molecules (NeurIPS2019) [[**Paper**](https://proceedings.neurips.cc/paper/2019/hash/a4d8e2a7e0d0c102339f97716d2fdfb6-Abstract.html)][[**Code**](https://github.com/atomistic-machine-learning/G-SchNet)]     | 
|CDVAE          | Crystal Diffusion Variational Autoencoder for Periodic Material Generation (ICLR2022) [[**Paper**](https://openreview.net/forum?id=03RLpj-tc_)][[**Code**](https://github.com/txie-93/cdvae)]     | 
|Con-CDVAE          | Con-CDVAE: A method for the conditional generation of crystal structures (Computational Materials Today, 2024) [[**Paper**](https://www.sciencedirect.com/science/article/pii/S2950463524000036)][[**Code**](https://github.com/cyye001/Con-CDVAE)]     | 
|Cond-CDVAE         | Deep learning generative model for crystal structure prediction (Arxiv, 2024) [[**Paper**](https://arxiv.org/abs/2403.10846)][[**Code**](https://github.com/ixsluo/cond-cdvae)]     | 
|LCOMs          | Latent Conservative Objective Models for Data-Driven Crystal Structure Prediction (NeurIPS2023 Workshop) [[**Paper**](https://openreview.net/forum?id=BTeWafMOyt)]     | 
|DiffCSP          | Crystal structure prediction by joint equivariant diffusion on lattices and fractional coordinates (NeurIPS2023) [[**Paper**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/38b787fc530d0b31825827e2cc306656-Abstract-Conference.html)][[**Code**](https://github.com/jiaor17/DiffCSP)]     | 
|EquiCSP         | Equivariant Diffusion for Crystal Structure Prediction (ICML2024) [[**Paper**](https://openreview.net/forum?id=VRv8KjJNuj)][[**Code**](https://github.com/EmperorJia/EquiCSP)]     | 
|GemsDiff         | a (Nature Communications, 2024) [[**Paper**]()][[**Code**]()]     | 
|Uni-MOF          | a (Nature Communications, 2024) [[**Paper**]()][[**Code**]()]     | 
|Uni-MOF          | a (Nature Communications, 2024) [[**Paper**]()][[**Code**]()]     | 
|Uni-MOF          | a (Nature Communications, 2024) [[**Paper**]()][[**Code**]()]     | 
|Uni-MOF          | a (Nature Communications, 2024) [[**Paper**]()][[**Code**]()]     | 


## Crystal Representation
- Resolving the data ambiguity for periodic crystals (NeurIPS2022) [[**Paper**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c256fa1965318b7fcb9ed104c265540-Abstract-Conference.html)]

- Neural Structure Fields with Application to Crystal Structure Autoencoders (NeurIPS2022 workshop) [[**Paper**](https://openreview.net/pdf?id=qLKFSAvMka4)]

- Symmetry-Informed Geometric Representation for Molecules, Proteins, and Crystalline Materials (NeurIPS2023) [[**Paper**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d07379f3acf3af51dfc8598862cadfa0-Abstract-Datasets_and_Benchmarks.html)]
            [[**Code**](https://github.com/chao1224/Geom3D)]

- Stoichiometry Representation Learning with Polymorphic Crystal Structures (NeurIPS2023 workshop) [[**Paper**](https://openreview.net/pdf?id=DBiWSzlaGz)]
            [[**Code**](https://github.com/Namkyeong/PolySRL_AI4Science)]

- Connectivity Optimized Nested Line Graph Networks for Crystal Structures (NeurIPS2023) [[**Paper**](https://openreview.net/pdf?id=l3K28QS6R6)]
            [[**Code**](https://github.com/matbench-submission-coGN/CrystalGNNs)]

## Other

- Discovering Symmetry Breaking in Physical Systems with Relaxed Group Convolution (ICML2024) [[**Paper**](https://openreview.net/forum?id=59oXyDTLJv)]
    [[**Code**](https://github.com/atomicarchitects/Symmetry-Breaking-Discovery)]

- A language-based recommendation system for material discovery (ICML2023 workshop) [[**Paper**](https://openreview.net/pdf?id=eR6HlKQDvt)]
     
- Rigid Body Flows for Sampling Molecular Crystal Structures (ICML2023) [[**Paper**](https://proceedings.mlr.press/v202/kohler23a.html)]
            [[**Code**](https://github.com/noegroup/rigid-flows)]

- Automated Diffraction Pattern Analysis for Identifying Crystal Systems Using Multiview Opinion Fusion Machine Learning (NeurIPS2023 workshop) [[**Paper**](https://openreview.net/pdf?id=L6AJmCkfNe)]
            [[**Code**](https://github.com/YKQ98/Matformer)]




- Mitigating Bias in Scientific Data: A Materials Science Case Study (NeurIPS2023 workshop) [[**Paper**](https://openreview.net/pdf?id=PfpbWuC0Yk)]
            [[**Code**](https://github.com/Henrium/ET-AL)]
  



- EGraFFBench: Evaluation of Equivariant Graph Neural Network Force Fields for Atomistic Simulations (NeurIPS2023 workshop) [[**Paper**](https://openreview.net/pdf?id=SeXGn7MeUr)]
                        [[**Code**](https://github.com/M3RG-IITD/MDBENCHGNN)]


## Nature

- An invertible, invariant crystal representation for inverse design of solid-state materials using generative deep learning (NC 2023) [[**Paper**](https://www.nature.com/articles/s41467-023-42870-7)]
            [[**Code**](https://github.com/xiaohang007/SLICES)]

- A comprehensive transformer-based approach for high-accuracy gas adsorption predictions in metal-organic frameworks (NC 2024) [[**Paper**](https://www.nature.com/articles/s41467-024-46276-x)]

- Explainable machine learning in materials science (npj Computational Materials 2022) [[**Paper**](https://www.nature.com/articles/s41524-022-00884-7)]
 
- A universal graph deep learning interatomic potential for the periodic table (Nature Computational Science 2022) [[**Paper**](https://www.nature.com/articles/s43588-022-00349-3)]
 
- Physics guided deep learning for generative design of crystal materials with symmetry constraints (npj Computational Materials 2023) [[**Paper**](https://doi.org/10.1038/s41524-023-00987-9)]
 

- MatterGen: a generative model for inorganic materials design (arXiv 2023) [[**Paper**](https://arxiv.org/abs/2312.03687)]
 
- Space Group Informed Transformer for Crystalline Materials Generation (arXiv 2024) [[**Paper**](https://arxiv.org/abs/2403.15734)]

- CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling (Nature Machine Intelligence 2023) [[**Paper**](https://www.nature.com/articles/s42256-023-00716-3)]

- MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields (arXiv 2023) [[**Paper**](https://arxiv.org/abs/2206.07697)]

- Scaling deep learning for materials discovery (Nature 2023) [[**Paper**](https://www.nature.com/articles/s41586-023-06735-9)]

  

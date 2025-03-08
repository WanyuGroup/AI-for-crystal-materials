# AI for Crystal Materials： models and benchmarks
Here we have collected papers with the theme of "AI for crystalline materials" that have appeared at top machine learning conferences and journals (ICML, ICLR, NeurIPS, AAAI, NPJ, NC, etc.) in recent years. See https://arxiv.org/abs/2408.08044 for details. We will keep this page updated.

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
|DTNet          | Dielectric tensor prediction for inorganic materials using latent information from preferred potential (npj Computational Materials, 2024) [[**Paper**](https://www.nature.com/articles/s41524-024-01450-z)][[**Code**](https://github.com/pfnet-research/dielectric-pred)]     | 
|GMTNet          |  A Space Group Symmetry Informed Network for O(3) Equivariant Crystal Tensor Prediction (ICML2024) [[**Paper**](https://openreview.net/forum?id=BOFjRnJ9mX)][[**Code**](https://github.com/divelab/AIRS/tree/main/OpenMat/GMTNet)]     | 
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
|SODNet          | Learning Superconductivity from Ordered and Disordered Material Structures (NeurIPS2024) [[**Paper**](https://openreview.net/forum?id=iNYrB3ip9F#discussion)][[**Code**](https://github.com/pincher-chen/SODNet)]     |
|ChargE3Net          | Higher-order equivariant neural networks for charge density prediction in materials (npj Computational Materials, 2024) [[**Paper**](https://www.nature.com/articles/s41524-024-01343-1?fromPaywallRec=false)][[**Code**](https://github.com/AIforGreatGood/charge3net)]     |
|ECSG          | Predicting thermodynamic stability of inorganic compounds using ensemble machine learning based on electron configuration (Nature Communications, 2025) [[**Paper**](https://www.nature.com/articles/s41467-024-55525-y)][[**Code**](https://github.com/Haozou-csu/ECSG)]     | 
|ECD         |ECD: A Machine Learning Benchmark for Predicting Enhanced-Precision Electronic Charge Density in Crystalline Inorganic Materials (ICLR2025) [[**Paper**](https://openreview.net/forum?id=SBCMNc3Mq3)]    | 
|CrystalFramer         |Rethinking the role of frames for SE(3)-invariant crystal structure modeling (ICLR2025) [[**Paper**](https://openreview.net/forum?id=gzxDjnvBDa)]    | 
|SimXRD         |SimXRD-4M: Big Simulated X-ray Diffraction Data and Crystalline Symmetry Classification Benchmark (ICLR2025) [[**Paper**](https://openreview.net/forum?id=mkuB677eMM)]  [[**Code**](https://github.com/Bin-Cao/SimXRD)]   | 
|ct-UAE         |Transformer-generated atomic embeddings to enhance prediction accuracy of crystal properties with machine learning (Nature Communications, 2025) [[**Paper**](https://www.nature.com/articles/s41467-025-56481-x)]  [[**Code**](https://github.com/fduabinitio/ct-UAE)]   | 
|-         |Cross-scale covariance for material property prediction (npj Computational Materials, 2025) [[**Paper**](https://www.nature.com/articles/s41524-024-01453-w)]  [[**Code**](https://github.com/bjasperson/strength_covariance)]   | 

## Crystalline Material Synthesis

|Method         |           Paper            |
|----------------|-------------------------------|
|G-SchNet         | Symmetry-adapted generation of 3d point sets for the targeted discovery of molecules (NeurIPS2019) [[**Paper**](https://proceedings.neurips.cc/paper/2019/hash/a4d8e2a7e0d0c102339f97716d2fdfb6-Abstract.html)][[**Code**](https://github.com/atomistic-machine-learning/G-SchNet)]     | 
|CDVAE          | Crystal Diffusion Variational Autoencoder for Periodic Material Generation (ICLR2022) [[**Paper**](https://openreview.net/forum?id=03RLpj-tc_)][[**Code**](https://github.com/txie-93/cdvae)]     | 
|Con-CDVAE          | Con-CDVAE: A method for the conditional generation of crystal structures (Computational Materials Today, 2024) [[**Paper**](https://www.sciencedirect.com/science/article/pii/S2950463524000036)][[**Code**](https://github.com/cyye001/Con-CDVAE)]     | 
|Cond-CDVAE         | Deep learning generative model for crystal structure prediction (npj Computational Materials, 2024) [[**Paper**](https://www.nature.com/articles/s41524-024-01443-y?fromPaywallRec=false)][[**Code**](https://github.com/ixsluo/cond-cdvae)]     | 
|LCOMs          | Latent Conservative Objective Models for Data-Driven Crystal Structure Prediction (NeurIPS2023 Workshop) [[**Paper**](https://openreview.net/forum?id=BTeWafMOyt)]     | 
|DiffCSP          | Crystal structure prediction by joint equivariant diffusion on lattices and fractional coordinates (NeurIPS2023) [[**Paper**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/38b787fc530d0b31825827e2cc306656-Abstract-Conference.html)][[**Code**](https://github.com/jiaor17/DiffCSP)]     | 
|DiffCSP-SC         | Learning Superconductivity from Ordered and Disordered Material Structures (NeurIPS2024) [[**Paper**](https://openreview.net/forum?id=iNYrB3ip9F#discussion)][[**Code**](https://github.com/pincher-chen/DiffCSP-SC)]     | 
|EquiCSP         | Equivariant Diffusion for Crystal Structure Prediction (ICML2024) [[**Paper**](https://openreview.net/forum?id=VRv8KjJNuj)][[**Code**](https://github.com/EmperorJia/EquiCSP)]     | 
|GemsDiff         | Vector Field Oriented Diffusion Model for Crystal Material Generation (AAAI2024) [[**Paper**](https://ojs.aaai.org/index.php/AAAI/article/view/30224)][[**Code**](https://github.com/aklipf/gemsdiff)]     | 
|SyMat          | Towards symmetry-aware generation of periodic materials (NeurIPS2023) [[**Paper**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a73474c359ed523e6cd3174ed29a4d56-Abstract-Conference.html)][[**Code**](https://github.com/divelab/AIRS/tree/main/OpenMat/SyMat)]     | 
|EMPNN         | Equivariant Message Passing Neural Network for Crystal Material Discovery (AAAI2023) [[**Paper**](https://ojs.aaai.org/index.php/AAAI/article/view/26673)][[**Code**](https://github.com/aklipf/pegnn)]     | 
|UniMat         | Scalable Diffusion for Materials Generation (ICLR2024) [[**Paper**](https://openreview.net/forum?id=wm4WlHoXpC)][[**Code**](https://unified-materials.github.io/unimat/)]  | 
|PGCGM          | Physics guided deep learning for generative design of crystal materials with symmetry constraints (npj Computational Materials, 2023) [[**Paper**](https://www.nature.com/articles/s41524-023-00987-9)][[**Code**](https://github.com/MilesZhao/PGCGM)]     | 
|CubicGAN         | High-throughput discovery of novel cubic crystal materials using deep generative neural networks (Advanced Science, 2021) [[**Paper**](https://onlinelibrary.wiley.com/doi/full/10.1002/advs.202100566)][[**Code**](https://github.com/MilesZhao/CubicGAN)]     | 
|PCVAE          | PCVAE: A Physics-informed Neural Network for Determining the Symmetry and Geometry of Crystals (IJCNN2023) [[**Paper**](https://ieeexplore.ieee.org/abstract/document/10191051)][[**Code**](https://github.com/zjuKeLiu/PCVAE)]     | 
|DiffCSP++          | Space Group Constrained Crystal Generation (ICLR2024) [[**Paper**](https://openreview.net/forum?id=jkvZ7v4OmP)][[**Code**](https://github.com/jiaor17/DiffCSP-PP)]     | 
|FlowMM         | FlowMM: Generating Materials with Riemannian Flow Matching (ICML2024) [[**Paper**](https://openreview.net/forum?id=W4pB7VbzZI)][[**Code**](https://github.com/facebookresearch/flowmm)]     | 
|Govindarajan         | Behavioral Cloning for Crystal Design (ICLR2023 Workshop) [[**Paper**](https://openreview.net/forum?id=qxuIaeDlemv)][[**Code**]()]     | 
|CHGFlowNet         | Hierarchical GFlownet for Crystal Structure Generation (NeurIPS2023 Workshop) [[**Paper**](https://openreview.net/forum?id=dJuDv4MKLE)]    | 
|LM-CM,LM-AC          | Language models can generate molecules, materials, and protein binding sites directly in three dimensions as xyz, cif, and pdb files (Arxiv, 2023) [[**Paper**](https://arxiv.org/abs/2305.05708)][[**Code**](https://github.com/danielflamshep/xyztransformer)]     | 
|CrystaLLM          | Crystal structure generation with autoregressive large language modeling (Nature Communications, 2024) [[**Paper**](https://www.nature.com/articles/s41467-024-54639-7)][[**Code**](https://github.com/lantunes/CrystaLLM)]     | 
|CrystalFormer          | Space Group Informed Transformer for Crystalline Materials Generation (Arxiv, 2024) [[**Paper**](https://arxiv.org/abs/2403.15734)][[**Code**](https://github.com/deepmodeling/CrystalFormer)]     | 
|SLI2Cry         | An invertible, invariant crystal representation for inverse design of solid-state materials using generative deep learning (Nature Communications, 2023) [[**Paper**](https://www.nature.com/articles/s41467-023-42870-7)][[**Code**](https://github.com/xiaohang007/SLICES/tree/main)]     | 
|Gruver         | Fine-Tuned Language Models Generate Stable Inorganic Materials as Text (ICLR2024) [[**Paper**](https://openreview.net/forum?id=vN9fpfqoP1)][[**Code**](https://github.com/facebookresearch/crystal-text-llm)]     | 
|FlowLLM         | FlowLLM: Flow Matching for Material Generation with Large Language Models as Base Distributions (NeurIPS2024) [[**Paper**](https://openreview.net/forum?id=0bFXbEMz8e)][[**Code**](https://github.com/facebookresearch/flowmm)]     | 
|Mat2Seq         | Invariant Tokenization of Crystalline Materials for Language Model Enabled Generation (NeurIPS2024) [[**Paper**](https://openreview.net/forum?id=18FGRNd0wZ&noteId=Tmq6A9Gswe)]   | 
|GenMS         | Generative Hierarchical Materials Search (NeurIPS2024) [[**Paper**](https://openreview.net/forum?id=PsPR4NOiRC&noteId=NkcymWeWnc)]   | 
|ChemReasoner         | CHEMREASONER: Heuristic Search over a Large Language Model’s Knowledge Space using Quantum-Chemical Feedback (ICML2024) [[**Paper**](https://openreview.net/forum?id=3tJDnEszco)]  [[**Code**](https://github.com/pnnl/chemreasoner)]  | 
|a²c         | Predicting emergence of crystals from amorphous precursors with deep learning potentials (Nature Computational Science, 2024) [[**Paper**](https://www.nature.com/articles/s43588-024-00752-y)][[**Code**](https://github.com/jax-md/jax-md/tree/main/jax_md/a2c)]     | 
|-          |Rapid prediction of molecular crystal structures using simple topological and physical descriptors (Nature Communications, 2024) [[**Paper**](https://www.nature.com/articles/s41467-024-53596-5?fromPaywallRec=false)]    | 
|MatterGen         | A generative model for inorganic materials design (Nature, 2025) [[**Paper**](https://www.nature.com/articles/s41586-025-08628-5)][[**Code**](https://github.com/microsoft/mattergen)]     | 
|SymmCD         |SymmCD: Symmetry-Preserving Crystal Generation with Diffusion Models (ICLR2025) [[**Paper**](https://openreview.net/forum?id=xnssGv9rpW)][[**Code**](https://github.com/sibasmarak/SymmCD)]     | 
|MatExpert         |MatExpert: Decomposing Materials Discovery By Mimicking Human Experts (ICLR2025) [[**Paper**](https://openreview.net/forum?id=AUBvo4sxVL)]    | 
|-         |Designing Mechanical Meta-Materials by Learning Equivariant Flows (ICLR2025) [[**Paper**](https://openreview.net/forum?id=VMurwgAFWP)]    | 
|MOFFlow         |MOFFlow: Flow Matching for Structure Prediction of Metal-Organic Frameworks (ICLR2025) [[**Paper**](https://openreview.net/forum?id=dNT3abOsLo)]   | 
|TGDMat          |Periodic Materials Generation using Text-Guided Joint Diffusion Model (ICLR2025) [[**Paper**](https://openreview.net/forum?id=AkBrb7yQ0G)]    | 
|CrysBFN          |A Periodic Bayesian Flow for Material Generation (ICLR2025) [[**Paper**](https://openreview.net/forum?id=Lz0XW99tE0)]  [[**Code**](https://github.com/wu-han-lin/CrysBFN)]   | 
|OSDAs         |OSDA Agent: Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agents (ICLR2025) [[**Paper**](https://openreview.net/forum?id=9YNyiCJE3k)]    | 
|MAGUS        | Efficient crystal structure prediction based on the symmetry principle (Nature Computational Science, 2025) [[**Paper**](https://www.nature.com/articles/s43588-025-00775-z)]     | 
|Target XXXI        | A robust crystal structure prediction method to support small molecule drug development with large scale validation and blind study (Nature Communications, 2025) [[**Paper**](https://www.nature.com/articles/s41467-025-57479-1)]    |


## Aiding Characterization
|Method         |           Paper            |
|----------------|-------------------------------|
|    -      | Insightful classification of crystal structures using deep learning  (Nature Communications, 2018) [[**Paper**](https://www.nature.com/articles/s41467-018-05169-6)]     | 
|-          | Advanced steel microstructural classification by deep learning methods  (Scientific Reports, 2018) [[**Paper**](https://www.nature.com/articles/s41598-018-20037-5)]    | 
|    -      | Neural network for nanoscience scanning electron microscope image recognition (Scientific Reports, 2017) [[**Paper**](https://www.nature.com/articles/s41598-017-13565-z)]     | 
|-         | Deep Learning-Assisted Quantification of Atomic Dopants and Defects in 2D Materials (Advanced Science, 2021) [[**Paper**](https://onlinelibrary.wiley.com/doi/full/10.1002/advs.202101099)]   | 
|-        | Classification of crystal structure using a convolutional neural network  (IUCrJ,2017) [[**Paper**](https://journals.iucr.org/m/issues/2017/04/00/fc5018/index.html)]    | 
|-          | Synthesis, optical imaging, and absorption spectroscopy data for 179072 metal oxides  (Scientific Data, 2019) [[**Paper**](https://www.nature.com/articles/s41597-019-0019-4)]    | 
|-          | Adaptively driven X-ray diffraction guided by machine learning for autonomous phase identification  (npj Computational Materials, 2023) [[**Paper**](https://www.nature.com/articles/s41524-023-00984-y?fromPaywallRec=false)] [[**Code**](https://github.com/njszym/AdaptiveXRD)]     | 
|-          | Automated classification of big X-ray diffraction data using deep learning models  (npj Computational Materials, 2023) [[**Paper**](https://www.nature.com/articles/s41524-023-01164-8?fromPaywallRec=false)] [[**Code**](https://github.com/AGI-init/XRDs)]     | 
|XRD-AutoAnalyzer         | Integrated analysis of X-ray diffraction patterns and pair distribution functions for machine-learned phase identification  (npj Computational Materials, 2024) [[**Paper**](https://www.nature.com/articles/s41524-024-01230-9?fromPaywallRec=false)] [[**Code**](https://github.com/njszym/XRD-AutoAnalyzer)]     | 
|CrystalNet          | Towards end-to-end structure determination from x-ray diffraction data using deep learning  (npj Computational Materials, 2024) [[**Paper**](https://www.nature.com/articles/s41524-024-01401-8)] [[**Code**](https://github.com/gabeguo/deep-crystallography-public)]     | 
|-          | Construction and Application of Materials Knowledge Graph in Multidisciplinary Materials Science via Large Language Model  (NeurIPS2024) [[**Paper**](https://openreview.net/forum?id=GB5a0RRYuv)] [[**Code**](https://github.com/MasterAI-EAM/Material-Knowledge-Graph)]     | 




## Accelerating Theoretical Computation
|Method         |           Paper            |
|----------------|-------------------------------|
|BPNN         | Generalized neural-network representation of high-dimensional potential-energy surfaces  (Physical Review Letters, 2007) [[**Paper**](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401)]    | 
|-          | Gaussian approximation potentials: The accuracy of quantum mechanics, without the electrons  (Physical Review Letters, 2010) [[**Paper**](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.104.136403)]    | 
|NequIP           | E (3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials  (Nature Communications, 2022) [[**Paper**](https://www.nature.com/articles/s41467-022-29939-5)][[**Code**](https://github.com/mir-group/nequip)]     | 
|CHGNet          | CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling  (Nature Machine Intelligence, 2023) [[**Paper**](https://www.nature.com/articles/s42256-023-00716-3)][[**Code**](https://github.com/CederGroupHub/chgnet)]     | 
|Cormorant          | Cormorant: Covariant molecular neural networks  (NeurIPS2019) [[**Paper**](https://proceedings.neurips.cc/paper/2019/hash/03573b32b2746e6e8ca98b9123f2249b-Abstract.html)][[**Code**](https://github.com/risilab/cormorant)]     | 
|MACE         | MACE: Higher order equivariant message passing neural networks for fast and accurate force fields  (NeurIPS2022) [[**Paper**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4a36c3c51af11ed9f34615b81edb5bbc-Abstract-Conference.html)][[**Code**](https://github.com/ACEsuit/mace)]     | 
|DimeNet          | Directional Message Passing for Molecular Graphs  (ICLR2020) [[**Paper**](https://openreview.net/forum?id=B1eWbxStPH)][[**Code**](https://github.com/gasteigerjo/dimenet)]     | 
|M3GNet        | A universal graph deep learning interatomic potential for the periodic table  (Nature Computational Science, 2022) [[**Paper**](https://www.nature.com/articles/s43588-022-00349-3)][[**Code**](https://github.com/materialsvirtuallab/m3gnet)]     | 
|-          | Injecting domain knowledge from empirical interatomic potentials to neural networks for predicting material properties  (NeurIPS2022) [[**Paper**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5ef1df239d6640a27dd6ed9a59f518c9-Abstract-Conference.html)][[**Code**](https://github.com/shuix007/EIP4NNPotentials)]     | 
|CHGNet        | CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling  (Nature Machine Intelligence, 2023) [[**Paper**](https://www.nature.com/articles/s42256-023-00716-3)][[**Code**](https://github.com/CederGroupHub/chgnet)]     | 
|-          | Forces are not Enough: Benchmark and Critical Evaluation for Machine Learning Force Fields with Molecular Simulations  (Transactions on Machine Learning Research, 2023) [[**Paper**](https://openreview.net/forum?id=A8pqQipwkt)]     | 
|DeepH-E3        |General framework for E (3)-equivariant neural network representation of density functional theory Hamiltonian (Nature Communications, 2023) [[**Paper**](https://www.nature.com/articles/s41467-023-38468-8)]  [[**Code**](https://github.com/Xiaoxun-Gong/DeepH-E3/tree/main)]   | 
|AdsorbDiff     |AdsorbDiff: Adsorbate Placement via Conditional Denoising Diffusion  (ICML2024) [[**Paper**](https://openreview.net/forum?id=ZMgpE58PMj)]   [[**Code**](https://github.com/AdeeshKolluru/AdsorbDiff)]  | 
|DeepRelax     |Scalable crystal structure relaxation using an iteration-free deep generative model with uncertainty quantification  (Nature Communications, 2024) [[**Paper**](https://www.nature.com/articles/s41467-024-52378-3)]   [[**Code**](https://github.com/Shen-Group/DeepRelax)]  | 
|AssembleFlow     |AssembleFlow: Rigid Flow Matching with Inertial Frames for Molecular Assembly  (ICLR2025) [[**Paper**](https://openreview.net/forum?id=jckKNzYYA6)]  |  

## Common Dataset and Platform

| Dataset | Description | URL |
|---------|-------------|-----|
| Materials Project | Materials Project encompasses over 120,000 materials, each accompanied by a comprehensive specification of its crystal structure and important physical properties. | [Materials Project](https://next-gen.materialsproject.org/) |
| JARVIS-DFT | JARVIS-DFT encompasses data for approximately 40,000 materials and includes around one million calculated properties. | [JARVIS-DFT](https://jarvis.nist.gov/) |
| OQMD | OQMD is a repository of thermodynamic and structural properties of inorganic materials, derived from high-throughput DFT calculations. | [OQMD](http://oqmd.org) |
| Perov-5 | Perov-5 is a specialized dataset of perovskite crystal materials, containing 18,928 different perovskite materials. | [Perov-5](https://github.com/txie-93/cdvae/tree/main/data/perov_5) |
| Carbon-24 | Carbon-24 is a specialized dataset of carbon materials, containing over 10,000 different carbon structures. | [Carbon-24](https://figshare.com/articles/dataset/Carbon24/22705192) |
| Crystallography Open Database | Crystallography Open Database is a crystallography database that specializes in collecting and storing crystal structure information for inorganic compounds, small organic molecules, metal-organic compounds, and minerals. | [Crystallography Open Database](https://www.crystallography.net/) |
| Raman Open Database | Raman Open Database is an open database that specializes in collecting and storing Raman spectroscopy data. | [Raman Open Database](https://solsa.crystallography.net/rod/index.php) |
| Inorganic Crystal Structure Database | Inorganic Crystal Structure Database is the world's largest database for completely identified inorganic crystal structures. | [Inorganic Crystal Structure Database](https://icsd.products.fiz-karlsruhe.de/en) |
| Open Catalyst Project | The goal of Open Catalyst Project is to utilize artificial intelligence to simulate and discover new catalysts for renewable energy storage. | [Open Catalyst Project](https://opencatalystproject.org/) |
| Python Materials Genomics | Python Materials Genomics is a robust, open-source Python library for materials analysis, offering a range of modules for handling crystal structures, band structures, phase diagrams, and material properties. | [Python Materials Genomics](https://pymatgen.org/) |
| MatBench | MatBench is a benchmark suite in the field of materials science, designed to evaluate and compare the performance of various ML models. | [MatBench](https://github.com/materialsproject/matbench) |
| M² Hub | M² Hub is a machine learning toolkit for materials discovery research that covers the entire workflow. | [M² Hub](https://github.com/yuanqidu/M2Hub) |
| Phonon DOS Dataset | Phonon DOS Dataset contains approximately 1,500 crystalline materials whose phonon DOS is calculated from DFPT. | [Phonon DOS Dataset](https://doi.org/10.6084/m9.figshare.c.3938023) |
| Carolina Materials Database | CMD primarily consists of ternary and quaternary materials generated by some AI methods. | [Carolina Materials Database](http://www.carolinamatdb.org/) |
| Alexandria Database | Alexandria Database includes a large quantity of hypothetical crystal structures generated by ML methods or other algorithmic methodologies. | [Alexandria Database](https://alexandria.icams.rub.de/) |
| Materials Project Trajectory Dataset | MPtrj contains 1,580,395 atomic configurations, corresponding energies, 7,944,833 magnetic moments, 49,295,660 forces, and 14,223,555 stress values. | [Materials Project Trajectory Dataset](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842?file=41619375) |
| Quantum MOF | QMOF is a dataset of over 20K metal-organic frameworks and coordination polymers derived from DFT. | [Quantum MOF](https://github.com/Andrew-S-Rosen/QMOF) |
| Open Materials 2024 | OMat24 contains over 110 million DFT calculations focused on structural and compositional diversity. | [Open Materials 2024](https://huggingface.co/datasets/fairchem/OMAT24) |
| SuperCon3D | SuperCon3D contains 1,578 superconductor materials (includes 83 distinct elements), each with both Tc and crystal structure data. | [SuperCon3D](https://github.com/pincher-chen/SODNet/tree/main/datasets/SuperCon) |
| Atomly | The Atomly database provides an extensive collection of material data generated through high-throughput first-principles calculations. This includes 320,000 inorganic crystal structures, 310,000 bandgap and density of states profiles, 12,000 dielectric constant tensors, and 16,000 mechanical tensors. | [Atomly](https://atomly.net) |

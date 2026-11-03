---
title: 'GRICS-torch: A PyTorch implementation of joint motion-corrected MRI reconstruction'
tags:
  - Python
  - PyTorch
  - MRI
  - MRI reconstruction
  - MRI motion correction
authors:
  - name: Karyna Isaieva
    affiliation: '1'
  - name: Henri Weber
    affiliation: '2'
  - name: Marc Fauvel
    affiliation: '3'
  - name: Freddy Odille
    affiliation: '1, 3'
affiliations:
  - index: 1
    name: 'Université de Lorraine, Inserm, IADI, F-54000 Nancy, France'
  - index: 2
    name: 'Institute of Medical Informatics, University of Lübeck, D-23562 Lübeck, Germany'
  - index: 3
    name: 'CHRU-Nancy, INSERM, Université de Lorraine, CIC, Innovation Technologique, F-54000 Nancy, France'
date: 02 October 2026
bibliography: paper.bib
---

# Summary

`GRICS-torch` is a package for motion-corrected MRI reconstruction. Its core is the previously validated GRICS algorithm [@odille2008grics], implemented in PyTorch [@paszke2019pytorch]. It supports 2D and 3D rigid and non-rigid MRI motion correction. The input data can be loaded from common data formats or generated using integrated sampling and motion simulators. After reconstruction, `GRICS-torch` outputs a motion-corrected MRI image, optionally in the DICOM format, and the estimated motion model.

# Statement of need

MRI acquisitions can require relatively long measurement times, making the acquired data sensitive to subject motion. The most common sources of motion include physiological non-rigid motion (e.g., respiratory and cardiac motion) and head motion. Motion occurring during an acquisition introduces inconsistencies between acquired samples and may result in artifacts such as blurring, ghosting, and loss of anatomical detail [@zaitsev2015motion].

To address this problem, numerous motion-corrected MRI reconstruction algorithms have been developed. However, many currently available non-rigid methods rely on the reconstruction of motion-resolved images. In cases of relatively small motion and in the absence of oversampling, this approach may fail. In contrast, the Generalized Reconstruction by Inversion of Coupled Systems (GRICS) algorithm [@odille2008grics] jointly estimates both the motion-corrected image and the motion model. Despite being extensively used internally for correcting non-rigid [@odille2010cine; @vuissoz2012cine; @fayad2015petmr; @odille2018cine3d; @isaieva2023; @isaieva2025] and rigid motion [@zeroual2024performance], the historical C++ implementation of GRICS remained in-house and required additional preprocessing scripts. Its recent application in studies conducted in collaboration with other laboratories [@nohava2024breast; @gnanago2025torso; @witte2026grics] has demonstrated interest in the GRICS algorithm beyond its original laboratory. Therefore, there was a need for an open, self-contained, and maintainable implementation of this algorithm, and `GRICS-torch` fills this gap.

# State of the field

A large number of retrospective and prospective motion-correction strategies have been proposed, several of which are available as open-source implementations.

For applications such as brain MRI, motion can often be approximated as rigid. Three-dimensional rigid motion is described by only six degrees of freedom, which makes it possible to estimate motion parameters jointly with the image reconstruction. Examples of openly available rigid motion-correction algorithms are TAMER [@haskell2018tamer] and DISORDER [@corderogrande2020disorder].

Non-rigid physiological motion is more challenging since the deformation field contains many more degrees of freedom. A common strategy exploits the quasi-periodic nature of physiological motion, which allows for motion binning. If the input data allow, a set of motion-resolved images (i.e., images corresponding to each of the motion bins) can be reconstructed. The motion model can then be estimated from these images using optical flow or other methods, and motion-compensated images can therefore be obtained by incorporating the estimated motion field into the forward MRI encoding operator. An open-source implementation of this approach is included in the BART MRI reconstruction toolbox [@uecker2015bart]. Other openly available implementations include XD-GRASP [@feng2016xdgrasp], iMoCo [@zhu2020imoco], MostMoCo [@ding2022mostmoco], and MoCoLoR [@tan2023mocolor].

However, in many cases, the reconstruction of good-quality motion-resolved images can itself be extremely challenging. An alternative strategy is thus to estimate the motion model directly from the acquired data. MR-MOTUS [@huttinga2020mrmotus] estimates non-rigid motion directly from k-space data but relies on a previously reconstructed reference image. Another strategy is to estimate the motion model jointly with the motion-corrected image by alternating optimization problems, as in GRICS. However, no open-source implementations of this strategy were previously available.

# Software design

The code was developed using PyTorch with CUDA [@nickolls2008cuda] support. `GRICS-torch` separates data preparation, physical operators, numerical optimization, and experiment configuration. Configuration is organized into TOML files covering runtime settings, reconstruction parameters, simulation, and other options. The project also provides image export utilities and a Docker-based environment. Additionally, it provides four demonstration notebooks and ready-to-use pipelines used in the authors' applications. Its full documentation is available at [`GRICS-torch` documentation](https://github.com/IADI-Nancy/GRICS-torch/tree/main/docs).

## The GRICS algorithm and the related code

GRICS is a multiresolution optimization algorithm that performs several Gauss-Newton (GN) iterations at each resolution level. Each GN iteration alternates between image reconstruction and motion-model estimation steps, solved using the conjugate gradient (CG) method (\autoref{fig:GRICS_algo}). The method requires MRI k-space data and synchronized physiological surrogate data. A detailed description of the algorithm can be found in the original article [@odille2008grics] or in the `GRICS-torch` documentation: [GRICS-torch documentation](https://github.com/IADI-Nancy/GRICS-torch/blob/main/docs/algorithm.md).
 

![Schematic representation of the GRICS algorithm. CG denotes conjugate gradient. α in this illustration corresponds to a non-rigid motion model but can also represent rigid motion parameters.](figures/GRICS_algo.png){#fig:GRICS_algo}

The GRICS algorithm is implemented in the `reconstruction/JointReconstructor` class. We made an effort to separate the core of the algorithm from the logging/plotting part to make the algorithm more readable. The forward and adjoint MRI encoding operators, motion perturbation operators, and motion operator are implemented in dedicated classes. The conjugate gradient solver was reimplemented to operate directly on PyTorch tensors with matrix-free representations of the encoding and motion perturbation operators.

At the time of publication, it supports only Cartesian sampling trajectories.

## Data preparation

Data preparation is managed by the `DataPreparer` class and includes loading the input (real or synthetic) data, estimation of coil sensitivity maps, optional motion simulation, and binning into different motion states.

### Input data load

The data are loaded by the `InputDataLoader` class (\autoref{fig:data_loader}). It supports both synthetic and real MRI data. Currently supported synthetic data types include images and the Shepp–Logan phantom [@shepp1974fourier], both of which require sampling simulation (in k-space). This sampling is done by `SamplingSimulator` and is called from the `SyntheticDataGenerator` class.
 

![Data flow of the InputDataLoader class.](figures/data_loader.png){#fig:data_loader}

Regarding real MRI data, `GRICS-torch` currently supports the vendor-agnostic ISMRMRD (MRD) [@inati2017ismrmrd] format, as well as the Siemens raw data format (which is subsequently converted to ISMRMRD using the dedicated converter [@siemens_ismrmrd_converter]). It also supports several physiological data formats, including SAEC [@isaieva2022saec] and the Polaris Vega (Northern Digital, Waterloo, Ontario, Canada) infrared tracker format. These files are loaded by dedicated classes and then synchronized in the `RawDataPreparer` class. There is also the possibility of saving already synchronized raw data in a preprocessed HDF5 format, which can be directly loaded by `InputDataLoader`.

### Coil sensitivity calculation

Coil sensitivities can be calculated using the ESPIRiT method [@uecker2014espirit] or the unpublished “Odille-spline” method described in the documentation: [documentation](https://github.com/IADI-Nancy/GRICS-torch/blob/main/docs/implementation/coil_sensitivity_calculator.md). The latter performs better in the authors' applications.

### Motion simulation

Motion simulation is performed by applying transformations in image space and sampling subsets of k-space lines for each motion state. For rigid motion, the number and duration of motion events can be specified, with realistic translation and rotation ranges [@topolnjak2024motion]. For non-rigid motion, respiration is simulated using a sinusoidal signal, and the displacement field is obtained by modulating a respiratory-like deformation model.

### Motion binning

Binning can be performed either using k-means clustering [@macqueen1967kmeans] or using a more advanced "k-space energy" method. The latter takes into account the fact that lines from the k-space center correspond to low spatial frequencies and affect the resulting image quality more than peripheral lines. A more detailed description of the method can be found in the documentation: [documentation](https://github.com/IADI-Nancy/GRICS-torch/blob/main/docs/implementation/motion_binner.md).

# Research impact statement

The original GRICS implementation was extensively validated for cardiac MRI [@odille2010cine; @vuissoz2012cine; @odille2015joint; @odille2018cine3d]. It can also be applied to lung or liver MRI, or pediatric head MRI. However, its current main application is supine breast MRI.

Supine breast MRI suffers from respiratory motion of small amplitude, which, however, can significantly degrade image quality [@isaieva2023]. In this context, the GRICS algorithm is currently actively used by the main authors' team (University of Lorraine, France) [@isaieva2023; @isaieva2025; @meullenet2023breast; @megel2025breast], as well as by the Medical University of Vienna, Austria [@nohava2024breast; @gnanago2025torso]. There are ongoing projects in these teams that will use `GRICS-torch` implementation, including a novel hybrid reconstruction algorithm inspired by NLINV-Net [@blumenthal2024nlinvnet], and studies devoted to the use of unconventional motion sensors and application clinical populations currently conducted at the Medical University of Vienna.

In order to ensure that `GRICS-torch` is not inferior to the previously validated GRICS++ CPU implementation, a small validation study was performed. Two free-breathing supine breast MRI datasets with respiratory bellow signals [@isaieva2025] were used: 20 T2-weighted 2D volumes (288 × 475, 60–68 slices) and 12 T1-weighted 3D volumes (160 × 218 × 56). `GRICS-torch` performance was compared with GRICS++ on a workstation with two AMD EPYC 7F72 24-Core CPUs (96 logical threads), 512 GB RAM, and an NVIDIA A100 PCIe 40 GB GPU. The results were compared in terms of a no-reference sharpness index [@leclaire2015sharpness], as well as structural similarity (SSIM) and normalized root mean square error (NRMSE). Additionally, the computation time was evaluated.
 

![Comparison of `GRICS-torch` performance with the previously validated GRICS++ version in terms of reconstruction quality and time. The insets present the similarity metrics between the images reconstructed using GRICS++ and GRICS-torch.](figures/boxplots.png){#fig:boxplots}


The quantitative results of these comparisons are presented in \autoref{fig:boxplots}, and examples of motion-corrected reconstruction are provided in \autoref{fig:examples}. The corrected images were almost identical for the 2D case (SSIM 0.98, NRMSE 0.05) and somewhat different for the 3D case (SSIM 0.92, NRMSE 0.15). However, the sharpness index and visual evaluation supported the non-inferiority of `GRICS-torch` reconstructions. These discrepancies can be explained by minor implementation differences. Finally, the PyTorch implementation was substantially slower for 2D multislice CPU multithread reconstruction but significantly faster for 3D reconstruction.
 

![Examples of reconstruction using the GRICS++ and `GRICS-torch` algorithms compared with the uncorrected images. The contrast is deliberately exaggerated to highlight the artifacts. The dominant artifact type in the 2D images is blurring, while in the 3D images it is ghosting. Visible ghosting artifacts are indicated by the red arrows. While both versions correct motion artifacts, `GRICS-torch` performed better in this example.](figures/examples.png){#fig:examples}


# AI usage disclosure

We used generative AI tools for code generation, refactoring, and documentation. The main tool was Codex (OpenAI), and the exact version varied depending on the task and date. The level of AI assistance varied across tasks: the algorithmic part of the code was mostly human-written, while configuration parsing was entirely AI-assisted. Some parts of the code were automatically translated from other programming languages. All architectural decisions were made by a human. The code was entirely reviewed by the authors, and numerous tests (present of the [test branch](https://github.com/IADI-Nancy/GRICS-torch/tree/test/tests) were performed to ensure its reliability. Additionally, ChatGPT (OpenAI) was used for formatting and grammar checking of the manuscript text.

# Acknowledgements

The authors thank Ziad Al-Haj Hemidi for useful discussions at early stages of the project and Nora Vogt for providing the sharpness index code.

# References

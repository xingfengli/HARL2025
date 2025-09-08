# HARL2025
# Human Auditory Representation Learning for Cross-Dialect Bird Species Recognition
# Overview
Bird species recognition (BSR) is a critical tool for biodiversity monitoring and ecological health assessment. This study proposes human auditory representation learning, hereafter HARL, a novel approach that integrates gammatone- and Mel-spectrogram features with deep learning architectures, including ResNet50 and multi-head attention (MHA) mechanisms, to address these challenges. Experiments demonstrate that HARL significantly outperforms baseline methods. The combination of gammatone- and Mel-spectrogram features proves particularly effective, with MHA further enhancing generalization across regions. These results highlight the potential of HARL for ecological monitoring and conservation, offering a scalable and accurate solution for automated BSR in diverse geographic contexts. Our work bridges human auditory science and machine learning, providing a foundation for future research in bioacoustics and biodiversity conservation.

![HARL Blockdiagram](https://github.com/xingfengli/HARL2025/blob/main/models/blockdiagram.png)

# Requirements

# Datasets
The DB3V dataset was used for all the experiments in this work. 
Jing, X., Zhang, L., Xie, J., Gebhard, A., Baird, A., & Schuller, B. (2024). DB3V: A Dialect Dominated Dataset of Bird Vocalisation for Cross-corpus Bird Species Recognition, INTERSPEECH 2024, pp. 127-131, Kos, Greece. 
Zenodo. https://doi.org/10.5281/zenodo.11544734

# Instructions to run codes in features (Matlab 2024b)
```markdown
## Setup and Visualization Instructions

- Run `addpath('your_own_specified_path/features/gammatonegram')` to set up required files.  
- Run `visualization_demo.m` to visualize the spectrograms along with their delta variants.  
- Run `listFilesAndFolders.m` to extract, get, and store the Mel- and Gamma-Spectrograms.  
- **Reminder**: Do NOT forget to revise the folder paths to suit your devices.

- Note: Zipped extracted features could also be found as follows: https://1drv.ms/f/c/f1ce98298ad945ca/EspF2YopmM4ggPHrcAUAAAAB7gAUF3LLA8F9aL1zETtmFQ?e=r1FLv7

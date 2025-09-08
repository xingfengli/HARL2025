# HARL2025
# Human Auditory Representation Learning for Cross-Dialect Bird Species Recognition
# Overview
Bird species recognition (BSR) is a critical tool for biodiversity monitoring and ecological health assessment. This study proposes human auditory representation learning, hereafter HARL, a novel approach that integrates gammatone- and Mel-spectrogram features with deep learning architectures, including ResNet50 and multi-head attention (MHA) mechanisms, to address these challenges. Experiments demonstrate that HARL significantly outperforms baseline methods. The combination of gammatone- and Mel-spectrogram features proves particularly effective, with MHA further enhancing generalization across regions. These results highlight the potential of HARL for ecological monitoring and conservation, offering a scalable and accurate solution for automated BSR in diverse geographic contexts. Our work bridges human auditory science and machine learning, providing a foundation for future research in bioacoustics and biodiversity conservation.

![HARL Blockdiagram](https://github.com/xingfengli/HARL2025/blob/main/models/blockdiagram.png)

# Datasets
The DB3V dataset was used for all the experiments in this work. 
Jing, X., Zhang, L., Xie, J., Gebhard, A., Baird, A., & Schuller, B. (2024). DB3V: A Dialect Dominated Dataset of Bird Vocalisation for Cross-corpus Bird Species Recognition, INTERSPEECH 2024, pp. 127-131, Kos, Greece. 
Zenodo. https://doi.org/10.5281/zenodo.11544734

# Instructions to Run Codes in Features (MATLAB 2024b)

## Setup and Visualization Instructions

To set up and visualize Mel and Gamma spectrograms, follow these steps:

1. **Set up the environment**:
   - Run `addpath('your_own_specified_path/features/gammatonegram')` in MATLAB 2024b to include the required files.
   - Replace `your_own_specified_path` with the actual path to the `features/gammatonegram` folder on your device.

2. **Visualize spectrograms**:
   - Run `visualization_demo.m` to generate and display Mel and Gamma spectrograms along with their delta variants.

3. **Extract and store spectrograms**:
   - Run `listFilesAndFolders.m` to extract, process, and store Mel and Gamma spectrograms.

4. **Access pre-extracted features**:
   - Download pre-extracted spectrogram features in `.mat` format from: [Zipped Features (OneDrive)](https://1drv.ms/f/c/f1ce98298ad945ca/EspF2YopmM4ggPHrcAUAAAAB7gAUF3LLA8F9aL1zETtmFQ?e=r1FLv7).
   - Unzip the files to `your_own_specified_path/features/` before running `listFilesAndFolders.m`.

**Resources**:
- [MATLAB 2024b Documentation](https://www.mathworks.com/help/releases/R2024b/matlab/) for environment setup and scripting.
- [Download MATLAB 2024b](https://www.mathworks.com/products/matlab.html) to install the required software.
- [GitHub Markdown Guide](https://docs.github.com/en/get-started/writing-on-github) for formatting this README.

**Reminder**:
- Ensure all folder paths (e.g., `your_own_specified_path`) are updated to match your local setup.
- Verify that the OneDrive link is accessible; contact the repository owner if access is restricted.

## Python Model Training and Visualization (Python 3.8+)

1. **Set up the environment**:
   - numpy==1.24.4
   - scipy==1.10.1
   - torch==2.0.1
   - torchaudio==2.0.2
   - torchvision==0.15.2
   - scikit-learn==1.3.2
   - matplotlib==3.7.2
   - umap-learn==0.5.3.
   - Update folder paths in all scripts to match your local setup.

2. **Train Mel spectrogram models**:
   - Run `D1D2_mel_wo_atten.py` for training with D1 and testing with D2, without attention.
   - Run `D1D2_mel_wi_atten.py` for training with D1 and testing with D2, with attention.

3. **Train Gammatone spectrogram models**:
   - Run `D1D2_gamma_wo_atten.py` for training with D1 and testing with D2, without attention.
   - Run `D1D2_gamma_wi_atten.py` for training with D1 and testing with D2, with attention.

4. **Train combined Mel and Gammatone models**:
   - Run `D1D2_mel_plus_gamma_wo_atten.py` for combined training with D1 and testing with D2, without attention.
   - Run `D1D2_mel_plus_gamma_wi_atten.py` for combined training with D1 and testing with D2, with attention.

5. **Visualize results**:
   - Run `ROC.py` and `UMAPs.py` to generate ROC curves and UMAP visualizations.

## Resources

- [Python 3 Documentation](https://docs.python.org/3/) for Python environment setup.

## Reminders

- Update all folder paths (e.g., `your_own_specified_path`) to match your local setup for MATLAB and Python.
- For Python scripts, adjust paths for other `DmDn` cases (e.g., D1D3) as needed.
- Install Python dependencies (e.g., `pip install numpy matplotlib tensorflow umap-learn scipy`).


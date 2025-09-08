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

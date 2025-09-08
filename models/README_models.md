## Python Model Training and Visualization (Python 3.x)

1. **Set up the environment**:
   - Ensure Python 3.x is installed with required libraries (e.g., NumPy, Matplotlib, TensorFlow/PyTorch, UMAP).
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

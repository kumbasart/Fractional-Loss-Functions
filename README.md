# Fractional Loss Functions
Adaptive and robust loss functions for regression tasks.

![Fractional Loss Functions](Main_figure.png)

This repository provides the official MATLAB and Python implementation of the following paper:

> **When fractional calculus meets robust learning: Adaptive robust lossfunctions**
>
> Mert Can Kurucu, Müjde Güzelkaya, İbrahim Eksin, Tufan Kumbasar
>
> AI and Intelligent Systems Laboratory, Istanbul Technical University, Istanbul, 34469, Turkiye
> 
> Istanbul Technical University, Istanbul, Turkey
> 
> **Abstract:**  *In deep learning, robust loss functions are crucial for addressing challenges like outliers and noise. This paper introduces a novel family of adaptive robust loss functions, Fractional Loss Functions (FLFs), generated by deploying the fractional derivative operator into conventional ones. We demonstrate that adjusting the fractional derivative order $\alpha$ allows generating a diverse spectrum of FLFs while preserving the essential properties necessary for gradient-based learning. We show that tuning $\alpha$ gives the unique property to morph the loss landscape to reduce the influence of large residuals. Thus, $\alpha$ serves as an interpretable hyperparameter defining the robustness level of FLFs. However, determining $\alpha$ prior to training requires a manual exploration to pinpoint an FLF that aligns with the learning tasks. To overcome this issue, we reveal that FLFs can balance robustness against outliers while increasing penalization of inliers by tuning $\alpha$. This inherent feature allows transforming $\alpha$ to an adaptive parameter as a trade-off that ensures balanced learning of $\alpha$ is feasible. Thus, FLFs can dynamically adapt their loss landscape, facilitating error minimization while providing robustness during training. We performed experiments across diverse tasks and showed that FLFs significantly enhanced performance.*

Please contact kurucum@itu.edu.tr for inquiries. Thanks.

## Repository Contents

- **MATLAB Implementation:**  
  - `ExperimentForToyDataset.m` – Main script to run experiments.
  - Individual loss function files (e.g., L2, log-cosh, Cauchy, Geman-McClure, Welsch).

- **Python Implementation:**  
  - `example.ipynb` – A Jupyter Notebook demonstrating the PyTorch version.

## Dependencies

### MATLAB

- MATLAB 2023b or later
- MATLAB Deep Learning Toolbox

### Python / PyTorch

- Python 3.8 or later ([Download Python](https://www.python.org/downloads/release/python-380/))
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [PyTorch](https://pytorch.org/)
  
## Runing the code

Run the ExperimentForToyDataset.m script to execute the MATLAB code. Each loss function (L2, log-cosh, Cauchy, Geman-McClure, and Welsch) is defined in a separate file, and you can choose the desired loss function at the beginning of the script.

For the PyTorch version, open the example.ipynb Jupyter Notebook and run the cells in order.

## Please cite the following paper when using this code
```latex
@article{kurucu2025,
 title = {When fractional calculus meets robust learning: Adaptive robust loss functions},
 journal = {Knowledge-Based Systems},
 pages = {1-13},
 year = {2025},
 issn = {.},
 doi = {.},
 url = {.},
 author = {Mert Can Kurucu, M\"ujde G\"uzelkaya, Ibrahim Eksin, Tufan Kumbasar},
}
```

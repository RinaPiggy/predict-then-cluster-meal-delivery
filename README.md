# ReadMe

This repository supports the research paper titled **A Short-Term Predict-Then-Cluster Framework for Meal Delivery Services**.  
**Authors**: Jingyi Cheng and Shadi Sharif Azadeh*.  
**Institute**: Transport and Planning, Delft University of Technology.  
The manuscript is under review at _Transportation Research Part A: Policy and Practice_.

## Abstract:

Micro-delivery services offer promising solutions for on-demand city logistics, but their success relies on efficient real-time delivery operations and fleet management. 
On-demand meal delivery platforms seek to optimize real-time operations based on anticipatory insights into citywide demand distributions.    
To address these needs, this study proposes a short-term predict-then-cluster framework for on-demand meal delivery services. The framework utilizes ensemble-learning methods for point and distributional forecasting with multivariate features, including lagged-dependent inputs to capture demand dynamics. 
We introduce Constrained K-Means Clustering (CKMC) and Contiguity Constrained Hierarchical Clustering with Iterative Constraint Enforcement (CCHC-ICE) to generate dynamic clusters based on predicted demand and geographical proximity, tailored to user-defined operational constraints.
Evaluations of European and Taiwanese case studies demonstrate that the proposed methods outperform traditional time series approaches in both accuracy and computational efficiency. Clustering results demonstrate that the incorporation of distributional predictions effectively addresses demand uncertainties, improving the quality of operational insights. Additionally, a simulation study demonstrates the practical value of short-term demand predictions for proactive strategies, such as idle fleet rebalancing, significantly enhancing delivery efficiency.
By addressing demand uncertainties and operational constraints, our predict-then-cluster framework provides actionable insights for optimizing real-time operations. The approach is adaptable to other on-demand platform-based city logistics and passenger mobility services, promoting sustainable and efficient urban operations.

## Research Description
![image](https://github.com/user-attachments/assets/202ae13a-be9c-49b1-bd57-065bcd0cd440)

This repository contains the implementation of the **Short-Term Predict-Then-Cluster Framework**, which includes:
1. **Demand Prediction**: Utilizes ensemble-learning methods for accurate point and distributional forecasting of short-term demand using multivariate features and lagged-dependent inputs.
2. **Dynamic Clustering**:
   - **Constrained K-Means Clustering (CKMC)**: Generates clusters based on predicted demand and user-defined constraints.
   - **CCHC-ICE**: Applies iterative constraint enforcement to form geographically contiguous clusters tailored to operational needs.
3. **Simulation Study**: Demonstrates how short-term demand predictions can enhance real-time operations, such as idle fleet rebalancing, improving delivery efficiency and sustainability.



## Instructions

### Prerequisites

- Python (>=3.9) installed with dependencies listed in `requirements.txt`.
- A GPU is recommended for computational efficiency.

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/predict-then-cluster.git
   cd predict-then-cluster
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure the settings in the `config/` directory as needed (e.g., `settings.yaml`).

### Running the Code

1. Train the demand prediction model:
   ```bash
   python src/train_forecasting.py --config config/settings.yaml
   ```
2. Perform dynamic clustering:
   ```bash
   python src/clustering.py --config config/settings.yaml
   ```
3. Run the simulation study:
   ```bash
   python src/simulation.py --config config/settings.yaml
   ```
4. Visualize results:
   ```bash
   python scripts/visualize_results.py
   ```

## Repository Structure

```
project-root/
├── README.md               # Project description and setup instructions
├── src/                    # Source code for forecasting, clustering, and simulation
│   ├── train_forecasting.py
│   ├── clustering.py
│   ├── simulation.py
│   └── model/
├── tests/                  # Test scripts
│   └── test_clustering.py
├── config/                 # Configuration files
│   └── settings.yaml
├── docs/                   # Documentation
│   └── framework_overview.md
├── data/                   # Sample datasets
│   └── european_case_study.csv
├── images/                 # Images for documentation
│   └── methodology.png
├── scripts/                # Utility scripts
│   └── visualize_results.py
├── .gitignore              # Files to ignore
├── papers/                 # Relevant publications
├── presentations/          # Slides and other presentations
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

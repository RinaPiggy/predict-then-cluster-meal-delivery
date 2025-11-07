# ReadMe

This repository supports the research paper titled **A Short-Term Predict-Then-Cluster Framework for Meal Delivery Services**.  
**Authors**: Jingyi Cheng and Shadi Sharif Azadeh.  
**Institute**: Transport and Planning, Delft University of Technology.  
This work has been accepted by Data Science for Transportation ([manuscript](https://link.springer.com/article/10.1007/s42421-025-00140-6)). This repository will be updated as soon as possible.

## Abstract:

Micro-delivery services offer promising solutions for on-demand city logistics, but their success relies on efficient real-time delivery operations and fleet management. On-demand meal delivery platforms seek to optimize real-time operations based on anticipatory insights into city-wide demand distributions. To address these needs, this study proposes a short-term predict-then-cluster framework for on-demand meal delivery services. In the forecasting stage, point and distributional predictions are generated using multivariate features, including temporal, contextual, and lagged-dependent features to capture complex demand dynamics. In the clustering stage, we propose two methods: Constrained K-Means Clustering (CKMC) and Contiguity Constrained Hierarchical Clustering with Iterative Constraint Enforcement (CCHC-ICE). These approaches form dynamic, geographically coherent clusters based on predicted demand, while accommodating user-defined operational constraints. Case studies on European and Taiwanese datasets demonstrate that lagged-dependent ensemble learning models perform robustly under sparse, zero-inflated demand conditions, whereas deep learning models such as LSTM excel in denser data regimes. Furthermore, results from the European case study highlight that incorporating distributional forecasts effectively captures demand uncertainty, thereby enhancing the quality of clustering outcomes and operational decision-making. By integrating demand uncertainty and operational constraints, the proposed framework delivers forward-looking, actionable insights for optimizing real-time meal delivery operations. The approach is adaptable to other on-demand platform-based city logistics and passenger mobility services, contributing to more sustainable and efficient urban operations.

## Research Description
![image](https://github.com/user-attachments/assets/202ae13a-be9c-49b1-bd57-065bcd0cd440)

This repository contains the implementation of the **Short-Term Predict-Then-Cluster Framework**, which includes:
1. **Demand Prediction**: Utilizes ensemble-learning methods for accurate point and distributional forecasting of short-term demand using multivariate features and lagged-dependent inputs.
2. **Dynamic Clustering**:
   - **Constrained K-Means Clustering (CKMC)**: Generates clusters based on predicted demand and user-defined constraints.
   - **CCHC-ICE**: Applies iterative constraint enforcement to form geographically contiguous clusters tailored to operational needs.
3. **Simulation Study**: Demonstrates how short-term demand predictions can enhance real-time operations, such as idle fleet rebalancing, improving delivery efficiency and sustainability.



## Instructions

To be updated..

## License

This project is licensed under the MIT License - see the LICENSE file for details.

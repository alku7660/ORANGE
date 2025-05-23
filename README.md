# ORANGE
## Information to consider:

### Datasets
Most of the datasets can be found in the UCI Machine Learning Repository (please access https://archive.ics.uci.edu/).
The synthetic datasets can be found in the Datasets folder.
The Compas dataset (called "compass" in the repository) is the one studied in the Propublica analysis (please check https://www.propublica.org/datastore/dataset/compas-recidivismrisk-score-data-and-analysis).  
The data preparation for all the datasets follows the preprocessing steps proposed in: A.-H. Karimi, G. Barthe, B. Balle, and I. Valera, “Model-agnostic counterfactual explanations for consequential decisions,” in International Conference on Artificial Intelligence and Statistics. PMLR, 2020, pp. 895–905. Part of the code is reproduced here in the data_preparation.py file. Please, refer to their documentation for further details.

## Information to run the algorithm:
### Running ORANGE
The ORANGE algorithm requires the information on possible feature values, feature values directionality and feature mutability, for each of the features in each dataset of interest.  
In the data_constructor.py file, you may find the definitions for each of these properties for each of the tested datasets in the paper. Should you require adding one dataset that is not part of the ones already set, please complete that dataset information in this file.  

#### Machine
The experiments were run on a machine with the following main specifications:  

memory      256GiB System memory  
processor   AMD Ryzen Threadripper 3990X 64-Core Processor  
storage     Seagate FireCuda 520 SSD ZP2000GM30002 (NVMe)  
display     TU102 TITAN RTX

Once the dataset properties are set, the Gurobi package and license are installed, and you have verified the program requirements: run the main.py file with the inclusion of the dataset address in your local machine.  

These are examples notebooks and python scripts for the paper Mosso et al. (2025).
The input data is the postprocessed data obtained using the code published on https://github.com/mgh011/windpy on raw 20Hz data.
Variables preparation notebooks are run first, with different codes for the different models used in the analysis. 
The ML models are trained on a slurm cluster, with the code included in the Cluster_code folder.
Results are plotted and analyzed using the Forest_result notebooks. 
A comparison of feature selection methods is done in the Feature_selection_comparison notebook.

# Imaging_analysis

Example of Imaging analysis of neuronal activity.
Related to Fernandes et. al 2019

# Question: Can we find identify functionally distinct neurons by using clustering and linear regression?

Brief description of the approach to answer the question:

Inspired by Miri et al., 2011, a regressor-based ROI analysis of the imaging data was performed.

Regressors are generated with time series that are set to zero for all time points except the time points of stimulation, which are set to one (visual stimuli in this case are Prey-like, Looming and Dimming). 
The regressors are then convolved with a kernel describing the GCaMP response function.

A linear regression approach (using Python scikit-learn) was used to select neurons, removing neurons with activity not locked to stimulus presentation (spontaneously active).

Extracted neurons were clustered using hierarchical clustering (agglomerative approach with Python scipy.cluster.hierarchy.linkage) for visualization of response types.

The maximum score of either the prey-like stimuli (nasalward and temporalward), looming or dimming stimuli was used to assign ROIs to specific response types.

References:
Miri, A., Daie, K., Burdine, R.D., Aksay, E., and Tank, D.W. (2011). Regression-based identification of behavior-encoding neurons during large-scale optical imaging of neural activity at cellular resolution. J. Neurophysiol. 105, 964–980.

António M. Fernandes, Johannes Larsch, Joseph C. Donovan, Thomas O. Helmbrecht, Duncan Mearns, Yvonne Kölsch, Marco Dal Maschio, Herwig Baier
bioRxiv 598383; doi: https://doi.org/10.1101/598383

Some of the helper functions were written with the help of Joe Donovan (https://github.com/joe311), Vilim Štih(https://github.com/vilim) and Thomas Helmbrecht.

# Answer: Yes. This approach reveals functionally distinct neuronal classes. See Notebook for analysis

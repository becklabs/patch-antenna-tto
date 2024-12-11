# Improving Generative Inverse Design of Rectangular Patch Antennas with Test Time Optimization

## Abstract
We propose a two-stage deep learning framework for the inverse design of rectangular patch antennas. Our approach leverages generative modeling to learn a latent representation of antenna scattering parameters and conditions a subsequent generative model on these parameters to produce feasible antenna geometries. 
We further demonstrate that leveraging search and optimization techniques at test-time improves the accuracy of the generated designs and enables consideration of auxilliary objectives such as manufacturability. Our approach generalizes naturally to different design criteria, and can be easily adapted to more complex geometric design spaces. 

## Experiments

#### Test Time Compute Scaling
<div style="display: flex; justify-content: space-between;">
    <img src="figs/paper/curve_scaling.png" alt="Test Time Compute Image 1" width="45%">
    <img src="figs/paper/design_scaling.png" alt="Test Time Compute Image 2" width="45%">
</div>

#### Target vs Simulated Responses
<img src="figs/paper/s11_comparison.png" alt="Target vs Simulated Response Plot" width="100%">


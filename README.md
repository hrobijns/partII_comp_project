This repository contains an implementation of Barnes-Hut and FMM algorithms for N-body simulation, for the Part II Physics Computing Project coursework. The structure of the repo is detailed below:

- report.ipynb: the main report, a write-up of the project.

- data: a folder containing complexity scaling data collected from simulations that is plotted for the report.

- figures: the pre-generated figures used in the report.

- source: the relevant code used to implement the algorithms and carry out analysis for the report.

  - naive:
    - simulation.py: the main body of the code which carries out the pairwise summation (both non-vectorised, as displayed in the report, and a faster, vectorised version using NumPy array broadcasting).
    - animation.py: a script which produces animations using the vectorised naive simulation approach.
    - softening.py: the script which produced the softening GIF, Figure 3.
    - integration.py: the script which produced the integration GIF, Figure 4.
    - complexity.py: a script which helps analyse timing and memory complexity.
   
  - BH:
    - quadtree.py: the script which constructs the quadtree and carries out force calculations as by the BH criterion (presented in the report).
    - quadtree_plot.py: a script which visualises the quadtree construction (Figure 1).
    - simulation.py: a script which uses quadtree.py to simulate an N-body system.
    - energy.py: a script which plots energy conservation as a function of $\theta$ (Figure 6).
    - timingtheta.py: a script which plots computation time as a function of $\theta$ (Figure 6).
    - complexity.py: a script which helps analyse timing and memory complexity.
   
  - FMM:
    - kernels.py: the script which implements the kernel functions as by their mathematical definition.
    - kernels_test.py: a test script to verify whether the kernels are behaving as expected.
    - kernels_vectorised.py: a vectorised implementation of kernels.py.
    - quadtree.py: the simplified FMM quadtree, which also calculates interaction lists.
    - quadtree_plot.py: a script which visualises the interaction list (Figure 2).
    - fmm.py: the body of the algorithm, as presented in the report.
    - error.py: a script which shows relative error compared to the pairwise brute force approach as a function of expansion order.
    - p_timing.py: a script which shows computational time as a function of expansion order.
    - complexity.py: a script which helps analyse timing and memory complexity.
 

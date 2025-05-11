# The Naïve, Barnes-Hut and FMM Algorithms for N-Body Simulation
**Part II Physics Computing Project**

---

## 📂 Repository Structure

```text
.
├── report.ipynb      # the main report, a write-up of the project
├── data/      # simulation data for complexity scaling comparison
├── figures/      # pre-generated figures used in the report
└── source/      # source code for algorithms and analysis
    ├── naive/           
    │   ├── simulation.py   # python loop and NumPy array broadcasting approaches
    │   ├── animation.py   # animation using simulation.py
    │   ├── softening.py   # visualisation of softening parameter (Figure 3)
    │   ├── integration.py   # comparison of integration techniques (Figure 4)
    │   └── complexity.py   # time/memory complexity analysis
    ├── BH/              
    │   ├── quadtree.py   # quadtree construction and force calculation
    │   ├── quadtree_plot.py   # visualisation of quadtree (Figure 1)
    │   ├── simulation.py   # simulation using logic in quadtree.py 
    │   ├── energy.py   # energy conservation as a function of $\theta$
    │   ├── timingtheta.py   # computation time as a function of $\theta$
    │   └── complexity.py   # time/memory complexity analysis
    └── FMM/             
        ├── kernels.py   # expansions and translations for FMM
        ├── kernels_test.py   # test to verify implementation of kernels
        ├── kernels_vectorised.py   # vectorised version of kernels.py
        ├── quadtree.py   # symmetric quadtree with interaction list
        ├── quadtree_plot.py   # interaction list visualisation (Figure 2)
        ├── fmm.py   # main body of algorithm, presented in report
        ├── error.py   # error w.r.t naive approach as a function of $p$
        ├── p_timing.py   # computation time as a function of $p$
        └── complexity.py   # time/memory complexity analysis

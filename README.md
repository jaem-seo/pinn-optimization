# PINN for real-world optimization tasks
- To solve real-world optimization tasks, A new type of PINN incorporating "Goal loss" is introduced.
- In this repository, some showcases of optimization tasks (inverting a pendulum, finding the fastest path, and spacecraft swingby) are available, which can be seen in the [paper](https://www.nature.com/articles/s41598-023-49977-3).

# Note
- This is a proof-of-concept work, and there still are issues of convergence and seed-dependency. We recommend training on different seeds and choosing the one with the smallest loss, which is still practically reasonable.
- The saved weights and plots are generated with RTX-3080Ti GPU.

# References
- J. Seo, "Solving real-world optimization tasks using physics-informed neural computing." Scientific Reports 14 (2024) 202. (URL: [https://www.nature.com/articles/s41598-023-49977-3](https://www.nature.com/articles/s41598-023-49977-3))

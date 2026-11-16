# GIN-DPPO-E: code of the DRL framework and raw experimental results

This repository accompanies the manuscript *Online Pre-Scheduling for Shared Manufacturing Platforms with Distributed
Manufacturers: An End-to-End Dual-Actor Deep Reinforcement Learning Framework* (under review). It contains the code of the
deep reinforcement learning (DRL) framework and the raw results of all experiments reported in the manuscript.

The scheduling environment and the case data are not part of this repository, so the scripts cannot be run as they are.
The result files below can be used to check every table and figure, and `plot_paper_figs.py` redraws Figs. 6, 9 and 10
from them (`python plot_paper_figs.py --out figs`; requires numpy and matplotlib).

## Code

| File | Content |
|---|---|
| `smp/nets.py` | GIN encoder with edge features (Eq. (31)), mean / sum graph readout, GCN layer and MLP embedding used in the ablation study, networks of the compared DRL methods |
| `smp/agents.py` | task actor, manufacturer actor and shared critic of GIN-DPPO-E and its variants (including the single-actor variant), PBMP, R-PPO and QIHGRL |
| `smp/ppo.py` | dual-actor PPO with clipped surrogate loss and adaptive entropy coefficient (Algorithm 1) |
| `smp/ddqn.py`, `smp/dq.py`, `smp/meta.py` | compared methods MPDDQN, D-Q, GA and KTPOA |
| `smp/methods.py` | configuration of all methods and variants (hyperparameters of Table 5) |
| `train.py`, `run_all.py` | training of the learning-based methods and the complete experimental pipeline |
| `exp_static.py` | static instances (Tables 6–8) |
| `exp_estimation.py` | accuracy and computation time of the three evaluation methods (Fig. 6) |
| `exp_time.py` | computation time versus number of orders (Fig. 8) and computational cost of the GIN encoder (Section 3.2.2) |
| `exp_dynamic.py`, `fig_order7.py` | continuous order arrivals (Tables 9 and 10, Figs. 9 and 10) |
| `exp_validation.py` | model validation (Section 2.3.2) |
| `plot_curves.py` | training curves (Fig. 7) |
| `plot_paper_figs.py` | Figs. 6, 9 and 10 drawn from the result files |

## Results

Order completion times and transportation times are in hours, computation times in seconds unless stated otherwise.

| File | Manuscript |
|---|---|
| `results/table6.csv` | Table 6, static instances (ablation study) |
| `results_ablation_dynamic/` | Table 6, columns D-30, D-50 and D-100 (ablation variants under continuous order arrivals; same file structure as below) |
| `results/table7.csv`, `results/static_time.csv` | Table 7 and the computation times of GA and KTPOA in Section 4.2 |
| `results/table8.csv` | Table 8 |
| `results/table9.csv`, `results/table10.csv` | Tables 9 and 10, Orders 1–10 |
| `results/dynamic_summary.csv` | Tables 9 and 10, rows D-30, D-50 and D-100 (mean ± standard deviation over 10 arrival sequences), decision time per order |
| `results/dynamic_detail.csv` | every order of every arrival sequence in the continuous-arrival scenarios |
| `results/dynamic_tests.csv` | paired comparison over the 10 arrival sequences (Wilcoxon signed-rank test) |
| `results/table9_realized.csv`, `results/static_realized.csv` | completion times realized when the schedules are executed by the insertion-based simulation |
| `results/fig6.csv`, `results/fig6_summary.csv` | Fig. 6 |
| `runs/<method>/train_log.csv`, `runs/<method>/train_time.txt` | training logs of all learning-based methods (Fig. 7: `runs/GIN-DPPO-E/train_log.csv`) |
| `results/checkpoints.csv` | Section 4.2, best-validation checkpoint versus last checkpoint |
| `results/fig8.csv` | Fig. 8 |
| `results/fig9_info.txt`, `results/order7_schedule.csv` | Figs. 9 and 10 |
| `results/gin_cost.csv` | Section 3.2.2, computational cost of the GIN encoder |
| `results/validation_shop.csv`, `results/validation_order.csv`, `results/validation_summary.csv` | Section 2.3.2, model validation |
| `results/fig6.png` … `results/fig10.png` | Figs. 6–10 |
| `versions.txt` | software and hardware used for the experiments |

## License

Copyright (c) 2026 the authors. All rights reserved.

This repository is provided solely for the peer review of the above manuscript. No permission is granted to use, copy,
modify or distribute the code or the data.

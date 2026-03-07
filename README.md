This README describes a **complete, end-to-end** workflow to reproduce the experiments:
1) Download the **two benchmark datasets** released by **Luo et al.** (BTD-Combo and HBTD)  
2) Generate **ProtT5** (ProtTrans) embedding features (`t5` vectors)  
3) Generate **PSSM** features (via PSI-BLAST) and convert them to fixed-length vectors  
4) Generate **NMBAC** features  
5) Run training + evaluation (`train_eval.py`) to output the **paper-consistent metrics**:  

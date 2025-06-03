# TurnTakingPD

Part of the project Responsible AI for Voice Diagnostics (RAIVD) with file number NGF.1607.22.013 of the research program NGF AiNed Fellowship Grants, which is financed by the Dutch Research Council (NWO). This work was conducted in close collaboration with the project “Turn-taking in Dialogue in Populations with Communicative Impairment” (https://www.ru.nl/en/research/research-projects/turntakingin-dialogue-in-populations-with-communicative-impairment). This work used the Dutch national e-infrastructure with the support of the SURF Cooperative using grant no. EINF-10519.

---

## Abstract

Speech-based Parkinson’s disease (PD) detection has gained attention for its automated, cost-effective, and non-intrusive nature. As research studies usually rely on data from diagnostic-oriented speech tasks, this work explores the feasibility of diagnosing PD on the basis of speech data not originally intended for diagnostic purposes, using the Turn-Taking (TT) dataset. Our findings indicate that TT can be as useful as diagnostic-oriented PD datasets like PC-GITA. We also investigate which specific dataset characteristics impact PD classification performance. The results show that concatenating audio recordings and balancing participants’ gender and status distributions can be beneficial. Cross-dataset evaluation reveals that models trained on PC-GITA generalize poorly to TT, whereas models trained on TT perform better on PC-GITA. Furthermore, we provide insights into the high variability across folds, which is mainly due to large differences in individual speaker performance.

---

## Citation

If you use this dataset or methods from this project in academic work, please cite:

```bibtex
@misc{zhong2025evaluatingusefulnessnondiagnosticspeech,
  title        = {Evaluating the Usefulness of Non-Diagnostic Speech Data for Developing Parkinson's Disease Classifiers},
  author       = {Terry Yi Zhong and Esther Janse and Cristian Tejedor-Garcia and Louis ten Bosch and Martha Larson},
  year         = {2025},
  eprint       = {2505.18722},
  archivePrefix= {arXiv},
  primaryClass = {eess.AS},
  url          = {https://arxiv.org/abs/2505.18722},
}

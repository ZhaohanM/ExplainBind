<table>
  <tr>
    <td width="120" valign="middle">
      <img src="./assets/ExplainBind_logo.png"
           alt="ExplainBind Logo"
           height="250">
    </td>
    <td valign="middle">
      <h1>
        Explainable Physicochemical Determinants of Protein–Ligand Binding via Non-Covalent Interactions
      </h1>
    </td>
  </tr>
</table>


<div align="center">

<!-- ───── ExplainBind Two-line Rotating Typing Animation (Blue Q, Orange A) ───── -->
<div align="center" style="display:flex; flex-direction:column; align-items:center; gap:8px;">

  <!-- Question line (Blue) -->
  <a href="https://readme-typing-svg.vercel.app">
    <img
      src="https://readme-typing-svg.vercel.app?font=Fira+Code&weight=500&size=22&duration=5000&pause=2800&color=3B82F6&center=true&vCenter=true&width=900&lines=How+to+advance+PLI+prediction%3F;Why+treat+training+as+a+black-box%3F;Why+do+models+fail+out-of-domain%3F;Do+predicted+attention+maps+align+with+biology%3F&v=8"
      alt="ExplainBind Question"
      style="display:block;"
    />
  </a>

  <!-- Answer line (Orange) -->
  <a href="https://readme-typing-svg.vercel.app">
    <img
      src="https://readme-typing-svg.vercel.app?font=Fira+Code&weight=500&size=22&duration=5000&pause=2800&color=F77D67&center=true&vCenter=true&width=900&lines=Provide+nine+new+benchmarks+with+true+interaction+maps.;Break+it+-+Supervise+with+real+interaction+attention+maps.;Protein+sequence+similarity+is+the+key+factor.;Top+k25+of+attentions+hit+protein+binding+pockets.&v=8"
      alt="ExplainBind Answer"
      style="display:block; margin-top:6px;"
    />
  </a>

</div>

<!-- ───── Project Badges ───── -->
[![Project Page](https://img.shields.io/badge/Project-Page-4285F4?style=for-the-badge&logo=googlelens&logoColor=4285F4)](https://zhaohanm.github.io/ExplainBind/)
[![Gradio UI](https://img.shields.io/badge/Gradio-Online_Demo-FFCC00?style=for-the-badge&logo=gradio&logoColor=yellow&labelColor=grey)](https://huggingface.co/spaces/Zhaohan-Meng/ExplainBind)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](https://github.com/ZhaohanM/ExplainBind/blob/main/LICENSE)
[![Visitors](https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fgithub.com%2FZhaohanM%2FExplainBind&label=Views&countColor=%23f36f43&style=for-the-badge)](https://visitorbadge.io/status?path=https%3A%2F%2Fgithub.com%2FZhaohanM%2FExplainBind)

</div>

## 🔥 News
<!-- - **[Feb 2026]** 🧩 Introduce **InteractBind** with residue–atom ground-truth interaction maps. -->
- **[Feb 2026]** 🧠 Preprint will available soon!
- **[Feb 2026]** 🚀 ExplainBind demo UI is now live on [Hugging Face Spaces](https://huggingface.co/spaces/Zhaohan-Meng/ExplainBind)!
<!-- [ArXiv](https://arxiv.org/abs/2509.XXXXX). -->
## 🧩 Overview

**ExplainBind** is an interaction-aware framework for **protein–ligand binding (PLB)** prediction.  
It supervises token-level cross-attention using **non-covalent interaction maps** (e.g. hydrogen bonds, salt bridges, hydrophobic contacts, van der Waals, π–π, and cation–π interactions) derived from curated **PDB** protein–ligand complexes in **InteractBind**.  
By aligning model attention with these physically grounded signals, ExplainBind transforms PLB prediction from a black-box reasoning into an **chemistry-grounded** process suitable for large-scale screening.

<details open>
<summary>ExplainBind Framework</summary>

![framework](./assets/ExplainBind_main.png)

</details>

## 📖 Contents
- [⚙️ Installation](#️-installation)
- [⚡ Quick Start](#-quick-start)
- [🔬 Foundation Models](#-foundation-models)
- [🧫 Dataset](#️-dataset)
- [📝 Citation](#-citation)
- [🧰 Intended Use](#-intended-use)

## ⚙️ Installation
> [!TIP] 
> Clone this Github repo and set up a new conda environment.

```
# create a new conda environment
$ conda create --name ExplainBind python=3.9
$ conda activate ExplainBind

# install requried python dependencies
$ pip install -r requirements.txt

# clone the source code of ExplainBind
$ git https://github.com/ZhaohanM/ExplainBind.git
$ cd ExplainBind
```
> Requires: Python ≥ 3.9 and a CUDA-compatible GPU.

## ⚡ Quick Start

### Command-Line Inference
```bash
bash run.sh
```

## 🔬 Foundation Models

### 🧬 Protein Foundation Models

| Model Name | HuggingFace Link | Input Type |
|------------|------------------|-------------|
| ESM2 | [facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D) | Amino Acid Sequence |
| SaProt | [westlake-repl/SaProt_650M_AF2](https://huggingface.co/westlake-repl/SaProt_650M_AF2) | Structure-aware Sequence |
| SaProt | [westlake-repl/SaProt_650M_PDB](https://huggingface.co/westlake-repl/SaProt_650M_PDB) | Structure-aware sequence |

### 💊 Molecular Foundation Models

| Model Name | HuggingFace Link | Input Type |
|------------|------------------|-------------|
| MoLFormer-XL | [ibm-research/MoLFormer-XL-both-10pct](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct) | SMILES |
| SELFormer | [HUBioDataLab/SELFormer](https://huggingface.co/HUBioDataLab/SELFormer) | SELFIES |
| SELFIES-TED | [ibm-research/materials.selfies-ted](https://huggingface.co/ibm-research/materials.selfies-ted) | SELFIES |

> [!NOTE]  
> All foundation models remain frozen. ExplainBind trains the Fusion Module using structure-derived attention map supervision and the Classifier.

## 🧫 Dataset

We provide **9 benchmarks** with true residue–level interaction maps for PLI prediction evaluation. It will release soon!

| Dataset | Type | Example Use |
|----------|------|--------------|
| InteractBind (affinity) | Affinity score splits | Evaluate in-domain |
| InteractBind-P-25%/28%/31%/33% | Protein similarity splits | Evaluate sequence-level generalisation |
| InteractBind-L-06%/43%/62%/88% | Ligand similarity splits | Evaluate sequence-level generalisation |


<!-- ## 📝 Citation
```bibtex
@inproceedings{meng2026ExplainBind,
  title={ExplainBind: Explainable Protein–Ligand Binding via Non-Covalent Interaction Supervision},
  author={Meng, Zhaohan and Bai, Zhen and William, Oldham and Ounis, Iadh and Yuan, Ke and Meng, Zaiqiao and Xu, Hao and Joseph, Loscalzo},
  booktitle={BioArxiv},
  year={2026}
}
``` -->

## 📚 Acknowledgments

This work was supported in part by National Institutes of Health grants HL155107 and HL166137, and by American Heart Association MERIT award AHA1185447 to JL.
K.Y. acknowledges support from Cancer Research UK (EDDPGM-Nov21/100001, DRCMDP-Nov23/100010 and core funding to the CRUK Scotland Institute (A31287)), BBSRC BB/V016067/1, Prostate Cancer UK MA-TIA22-001 and EU Horizon 2020 grant ID: 101016851.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🧰 Intended Use

**ExplainBind** is designed to assist **computational biologists**, **AI researchers**, and **drug-discovery scientists** in analysing and explaining molecular interactions.

### Applications

- 🔬 **Drug Discovery** — Identify explainable binding fingerprints between novel compounds and proteins.  
- 🧠 **Model Explainability** — Quantify token-level biological grounding via attention-map supervision.  
- 🧪 **Cross-Domain Generalisation** — Diagnose prediction drop-offs across protein similarity strata.  

> [!IMPORTANT]  
> This framework is intended **solely for research purposes** and should **not** be used for clinical decision-making.


# 🧬 InteractBind  

> A physically grounded, large-scale protein–ligand interaction dataset  
> for interpretable and interaction-aware binding prediction  

---

## 🔬 Motivation  

Most existing protein–ligand binding datasets provide only coarse-grained supervision, such as binary labels or scalar affinity values. While effective for prediction, these signals compress complex molecular interaction processes into a single outcome, limiting interpretability and mechanistic understanding.

**InteractBind** addresses this limitation by explicitly modelling *non-covalent interaction patterns* derived from experimentally resolved protein–ligand complexes.  

It enables **token-level supervision**, bridging sequence-based representations with physically meaningful interaction structures.

---

## 📊 Dataset Overview  

InteractBind is constructed from high-quality experimentally resolved complexes and includes:

- 🧬 Protein sequences (FASTA and structure-aware sequence)
- 💊 Ligand molecular representations (SMILES and SELFIES)
- 📈 Binding labels and affinity annotations
- 🗺️ Token-level non-covalent interaction maps  

The dataset is designed to support both **prediction accuracy** and **mechanistic interpretability**.

---
## 🧫 Dataset

We provide **9 benchmarks** with true residue–level interaction maps for PLI prediction evaluation.

| Dataset | Type | Example Use |
|----------|------|--------------|
| InteractBind (affinity) | Affinity score splits | Evaluate in-domain |
| InteractBind-P-25%/28%/31%/33% | Protein similarity splits | Evaluate sequence-level generalisation |
| InteractBind-L-08%/35%/40%/59% | Ligand similarity splits | Evaluate sequence-level generalisation |

## 🧪 Supported Interaction Types  

Structured annotations are provided for major non-covalent interaction categories:

- Hydrogen bonds  
- Hydrophobic interactions  
- Salt bridges  
- π–π stacking  
- π–cation interactions  
- Van der Waals contacts  

Each interaction channel can be used independently or combined for multi-channel supervision.

---

## 🧠 Key Features  

- **Physically grounded supervision**  
  Derived from experimentally resolved complexes rather than heuristic attention signals.  

- **Token-level interaction maps**  
  Enables fine-grained modelling of residue–atom interactions.  

- **Model-agnostic integration**  
  Compatible with sequence-based encoders (e.g., ESM, SELFormer, and other protein–ligand models).  

- **Interpretability support**  
  Facilitates binding residue identification and interaction pattern analysis.  

- **Scalable design**  
  Allows large-scale training without requiring full structural modelling during inference.  

---

## 🚀 Research Applications  

InteractBind supports a broad range of research directions:

- Protein–ligand binding prediction  
- Binding pocket localisation  
- Interaction-aware representation learning  
- Mechanistic hypothesis generation  
- Drug discovery and virtual screening  
- Explainable AI for molecular modelling  

---

## 📦 Availability  

The full dataset, preprocessing scripts, and documentation are currently being finalised to ensure reproducibility and ease of use.

---

## ⏳ Coming Soon  

🚧 **InteractBind will be publicly released very soon.**  

Stay tuned for the official announcement.

---

If you are interested in early access or collaboration, please feel free to get in touch.

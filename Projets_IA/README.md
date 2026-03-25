# 🔍 Détection d'anomalies dans les écritures comptables
### Projet IA — Option Contrôle, Audit & Conseil | ENCG Settat
> **Module : Intelligence Artificielle · Encadré par: Dr. A. Larhlimi**

---

## 👥 Équipe Projet

<table>
  <tr>
    <td align="center" width="33.33%">
      <br/><br/>
      <br/><br/>
      <b><img src="Données/HOUTATE_Saïd_24010355.png" style="height:200px;margin-right:200px; float:left; border-radius:10px;"/></b><br/><br/>
      <b>Nom complet :</b> <code>[HOUTATE Saïd]</code><br/>
      <b>Apogée :</b> <code>[**24010355**]</code><br/>
      <br/><br/>
    </td>
    <td align="center" width="33.33%">
      <br/><br/>
      <br/><br/>
      <b><img src="Données/JAMAL_Yassine_22007655.jpg" style="height:200px;margin-right:200px; float:left; border-radius:10px;"/></b><br/><br/>
      <b>Nom complet :</b> <code>[JAMAL Yassine]</code><br/>
      <b>Apogée :</b> <code>[**22007655**]</code><br/>
      <br/><br/>
    </td>
     <td align="center" width="33.33%">
      <br/>
      <br/><br/>
      <b><img src="Données/Projet IA.png" style="height:200px;margin-right:200px; float:left; border-radius:10px;"/></b><br/><br/>
       <b>Google Collab :</b> <code>[QR Code]</code><br/>
      <br/><br/><br/>
    </td>
  </tr>
</table>

---

## Vidéo explicative : https://drive.google.com/drive/folders/1NfeadfPAaXAuXKnRgnJFMQeFnLivCJ8D?usp=sharing

---

## 📋 Table des matières

1. [Vue d'ensemble du projet](#1--vue-densemble-du-projet)
2. [Contexte & Problématique](#2--contexte--problématique)
3. [Sources de données vérifiées](#3--sources-de-données-vérifiées)
4. [Structure du dépôt](#4--structure-du-dépôt)
5. [Méthodologie](#5--méthodologie)
6. [Résultats & Performances](#6--résultats--performances)
7. [Normes IFAC / IAASB utilisées](#7--normes-ifac--iaasb-utilisées)
8. [Références bibliographiques](#8--références-bibliographiques)

---

## 1. 🗂 Vue d'ensemble du projet

| Champ | Détail |
|-------|--------|
| **Thème** | Audit financier & Détection d'anomalies |
| **Titre** | Détection d'anomalies dans les écritures comptables |
| **Type de tâche** | Classification supervisée |
| **Variable cible** | Écriture suspecte : `0 = normale` / `1 = anormale` |
| **Outils** | Python · Scikit-learn · XGBoost · LightGBM · Google Colab · GitHub · Claude AI |

### 🎯 Problématique

> **Comment un modèle d'apprentissage automatique peut-il identifier automatiquement les écritures comptables anormales à partir d'un journal général, en s'appuyant sur des indicateurs statistiques, comportementaux et temporels ?**

### ✨ Intérêt pédagogique & métier

Ce projet est au **cœur de l'audit des systèmes d'information**. Il mobilise :
- La **règle de Benford** — standard reconnu dans les missions d'audit légal
- Des **algorithmes Random Forest / XGBoost / LightGBM** sur des patterns comptables atypiques
- Un cadre directement lié aux **missions d'audit légal au Maroc** (ONEC, normes ISA/IFAC)

---

## 2. 🏦 Contexte & Problématique

L'audit financier est confronté à un défi structurel : les volumes de transactions comptables dans les grandes entreprises atteignent des millions d'écritures par exercice. L'auditeur humain ne peut prétendre à un examen exhaustif — les techniques d'échantillonnage traditionnel (ISA 530) ne couvrent que 5 à 10 % des écritures.

Face à cette réalité, ce projet propose d'automatiser la détection des écritures suspectes via le **Machine Learning**, en apprenant à distinguer les patterns normaux des anomalies révélatrices de fraudes ou d'erreurs matérielles, conformément aux exigences de l'**ISA 240** (Responsabilités de l'auditeur en matière de fraude).

---

## 3. 🗄 Sources de données vérifiées

> ⚠️ **Transparence des sources** : Il n'existe pas de dataset générique "Journal Entries" sur Kaggle. Ce projet s'appuie sur deux jeux de données publics **réels et vérifiés**, complétés par une augmentation synthétique reproduisant la structure d'un journal général SAP FICO.

### Datasets Kaggle (liens vérifiés ✅)

| Dataset | Auteur | Contenu | Lien |
|---------|--------|---------|------|
| **Audit Risk Dataset** | Siddharth Saxena (`sid321axn`) | 776 enregistrements d'audit · 26 variables de risque · label fraude binaire | [🔗 kaggle.com/datasets/sid321axn/audit-data](https://www.kaggle.com/datasets/sid321axn/audit-data) |
| **PaySim — Synthetic Financial Fraud** | Edgar Lopez-Rojas (`ealaxi`) | 6,3M transactions financières synthétiques · label fraude (~0,13%) | [🔗 kaggle.com/datasets/ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1) |

### Stratégie de construction du dataset

Le dataset final de **20 000 écritures** (95 % normales / 5 % suspectes) est construit en :
1. **Reproduisant la structure** du journal général SAP FICO (tables BKPF + BSEG)
2. **S'inspirant des variables de risque** de l'Audit Risk Dataset
3. **Calibrant les patterns de fraude** sur PaySim (montants, temporalité, utilisateurs)
4. **Appliquant les critères ISA 240** pour labelliser les anomalies

---

## 4. 📁 Structure du dépôt

```
📦 AI / Projets_IA /
│
├── 📄 README.md                  ← Ce fichier (présentation du projet)
├── 📄 compte_rendu_skill.pdf     ← Rapport complet structuré (compte rendu)
├── 💻 Notebook.ipynb             ← Notebook Google Colab commenté et reproductible
├── 𝄜  audit_data.csv      
├── 📹Vidéo_explicative.md
│
└── 📂 Données/                   ← Données du projet
    ├── README_data.md            ← Instructions de téléchargement (Kaggle)
    ├── ENCG-S.png                ← Logo ENCG SETTAT
    ├── Projet IA.png             ← QR Code vers le projet sur Google Collab
    ├── HOUTATE_Saïd_24010355.png
    └── JAMAL_Yassine_22007655.jpg         
```

---

## 5. 📄 Méthodologie

### Pipeline complet

```
Données brutes (Kaggle)
        │
        ▼
┌─────────────────────┐
│ Feature Engineering │  ← Loi de Benford · Variables temporelles · Comportement utilisateur
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│    Prétraitement    │  ← StandardScaler · LabelEncoder · SMOTE (déséquilibre 95/5)
└─────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│            Modélisation (4 modèles)          │
│ Baseline LR → Random Forest → XGBoost → LGBM │
└──────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────┐
│     Évaluation      │  ← F1-Score · AUC-ROC · PR-AUC · Matrice de confusion
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Explicabilité SHAP │  ← Importance globale · Force Plot · Top 20 alertes
└─────────────────────┘
```

### Features construites

| Famille | Features | Référence ISA |
|---------|----------|---------------|
| **Montant** | `log_amount`, `amount_zscore`, `round_amount_flag` | ISA 240, §A1 |
| **Loi de Benford** | `benford_deviation`, `benford_expected_freq` | ISA 240, §A10 |
| **Temporelles** | `is_weekend`, `is_late_night`, `near_closing` | ISA 240, §A3 |
| **Utilisateur** | `user_activity_zscore`, `user_entry_count` | ISA 240, §A2 |
| **Comptes** | `account_pair_rarity`, `account_debit_enc` | ISA 315 |

---

## 6. 📊 Résultats & Performances

### Tableau comparatif

| Modèle | Precision | Recall | F1-Score | AUC-ROC | PR-AUC |
|--------|:---------:|:------:|:--------:|:-------:|:------:|
| Logistic Regression *(baseline)* | 0.61 | 0.68 | 0.64 | 0.81 | 0.54 |
| Random Forest | 0.79 | 0.74 | 0.76 | 0.91 | 0.72 |
| **XGBoost** | **0.83** | 0.78 | **0.80** | **0.94** | **0.79** |
| **LightGBM** | 0.82 | **0.80** | **0.81** | 0.93 | 0.78 |

> 🏆 **LightGBM** — meilleur Recall (0.80) : recommandé en audit pour minimiser les anomalies non détectées  
> 🥇 **XGBoost** — meilleur équilibre global (F1 = 0.80, AUC-ROC = 0.94)

### Top features (SHAP)

```
1. amount_zscore          ████████████████████  (montant anormalement élevé/bas)
2. benford_deviation      ████████████████      (écart à la loi de Benford)
3. is_late_night          ████████████          (saisie nocturne 21h–6h)
4. user_activity_zscore   ██████████            (comportement inhabituel utilisateur)
5. round_amount_flag      ████████              (montant rond >= 10 000)
6. near_closing           ██████                (5 derniers jours du mois)
7. account_pair_rarity    █████                 (paire de comptes rare)
```

### Gain vs audit traditionnel

```
AVANT (échantillonnage ISA 530) : ~5–10% des écritures examinées manuellement
APRÈS (scoring ML)              : 100% des écritures analysées, alertes priorisées par score
```

---


## 7. 📜 Normes IFAC / IAASB utilisées

| Norme | Description | Application dans le projet | Lien officiel |
|-------|-------------|---------------------------|---------------|
| **ISA 240** *(révisée 2025)* | Responsabilités de l'auditeur en matière de fraude | Variable cible · Types d'anomalies labellisées | [🔗 iaasb.org](https://www.iaasb.org/publications/isa-240-revised-auditor-s-responsibilities-relating-fraud-audit-financial-statements) |
| **ISA 240 PDF** *(2013/2020)* | Handbook complet ISA 240 | Mise en œuvre des critères §A1–A10 | [🔗 ifac.org — PDF](https://www.ifac.org/_flysystem/azure-private/publications/files/A012%202013%20IAASB%20Handbook%20ISA%20240.pdf) |
| **ISA 315** *(2019)* | Identification & évaluation des risques d'anomalies significatives | Features comportementales utilisateurs | [🔗 ifac.org — PDF](https://www.ifac.org/_flysystem/azure-private/publications/files/ISA-315-Full-Standard-and-Conforming-Amendments-2019-.pdf) |
| **ISA 530** | Sondages en audit | Remplacé par scoring ML exhaustif | [🔗 ifac.org](https://www.ifac.org/knowledge-gateway/audit-assurance) |

---

## 8. 📚 Références bibliographiques

### Bases de données
- Saxena, S. (2019). *Audit Risk Dataset*. Kaggle. https://www.kaggle.com/datasets/sid321axn/audit-data
- Lopez-Rojas, E. (2019). *PaySim: Synthetic Financial Fraud Detection Dataset*. Kaggle. https://www.kaggle.com/datasets/ealaxi/paysim1

### Algorithmes de Machine Learning
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
- Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 2017*.
- Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321–357.

### Explicabilité & Audit Data Analytics
- Lundberg, S., & Lee, S. (2017). A Unified Approach to Interpreting Model Predictions (SHAP). *NeurIPS 2017*.
- Nigrini, M. (2012). *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection*. Wiley.

### Cadre institutionnel
- ONEC — Ordre National des Experts Comptables du Maroc. [www.oec.ma](https://oec.ma/)
- Loi marocaine 17-95 relative aux Sociétés Anonymes.
- Loi marocaine 5-96 relative aux autres formes de sociétés.

---

## 🏷️ Badges du projet

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0-brightgreen)
![SHAP](https://img.shields.io/badge/SHAP-Explicabilité-blueviolet)
![IFAC](https://img.shields.io/badge/Norme-ISA%20240%20%7C%20ISA%20315-blue)
![License](https://img.shields.io/badge/Licence-Académique-lightgrey)

---

<div align="center">

*Projet réalisé dans le cadre du module Intelligence Artificielle*  

</div>

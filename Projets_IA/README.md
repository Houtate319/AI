# 🔍 Détection d'anomalies dans les écritures comptables
### Projet IA — Option Contrôle, Audit & Conseil | ENCG Settat
> **Module : Intelligence Artificielle · Mars 2026 · Encadrant : A. Larhlimi**

---

## 👥 Équipe Projet

<table>
  <tr>
    <td align="center" width="50%">
      <br/>
      <!-- 📸 Remplacer le bloc ci-dessous par la photo de l'étudiant·e 1 -->
      <!-- <img src="assets/photo_etudiant1.jpg" width="130" height="130" style="border-radius:50%; object-fit:cover;" alt="Photo Étudiant 1"> -->
      <br/><br/>
      <b>🎓 Étudiant·e 1</b><br/><br/>
      <b>Nom complet :</b> <code>[ Prénom NOM ]</code><br/>
      <b>Filière :</b> <code>[ Option Contrôle, Audit & Conseil ]</code><br/>
      <b>Année :</b> <code>[ 2025–2026 ]</code><br/>
      <b>Email :</b> <code>[ email@encg-settat.ma ]</code><br/>
      <b>GitHub :</b> <a href="https://github.com/username1">@username1</a><br/>
      <b>LinkedIn :</b> <a href="https://linkedin.com/in/profil1">linkedin.com/in/profil1</a>
      <br/><br/>
    </td>
    <td align="center" width="50%">
      <br/>
      <!-- 📸 Remplacer le bloc ci-dessous par la photo de l'étudiant·e 2 -->
      <!-- <img src="assets/photo_etudiant2.jpg" width="130" height="130" style="border-radius:50%; object-fit:cover;" alt="Photo Étudiant 2"> -->
      <br/><br/>
      <b>🎓 Étudiant·e 2</b><br/><br/>
      <b>Nom complet :</b> <code>[ Prénom NOM ]</code><br/>
      <b>Filière :</b> <code>[ Option Contrôle, Audit & Conseil ]</code><br/>
      <b>Année :</b> <code>[ 2025–2026 ]</code><br/>
      <b>Email :</b> <code>[ email@encg-settat.ma ]</code><br/>
      <b>GitHub :</b> <a href="https://github.com/username2">@username2</a><br/>
      <b>LinkedIn :</b> <a href="https://linkedin.com/in/profil2">linkedin.com/in/profil2</a>
      <br/><br/>
    </td>
  </tr>
</table>

> 💡 **Comment ajouter vos photos :**
> 1. Placez vos photos dans le dossier `assets/` du dépôt
> 2. Nommez-les `photo_etudiant1.jpg` et `photo_etudiant2.jpg`
> 3. Décommentez les lignes `<img ...>` ci-dessus en retirant les `<!--` et `-->`

---

## 📋 Table des matières

1. [Vue d'ensemble du projet](#1--vue-densemble-du-projet)
2. [Contexte & Problématique](#2--contexte--problématique)
3. [Livrables](#3--livrables)
4. [Sources de données vérifiées](#4--sources-de-données-vérifiées)
5. [Structure du dépôt](#5--structure-du-dépôt)
6. [Méthodologie](#6--méthodologie)
7. [Résultats & Performances](#7--résultats--performances)
8. [Installation & Exécution](#8--installation--exécution)
9. [Normes IFAC / IAASB utilisées](#9--normes-ifac--iaasb-utilisées)
10. [Références bibliographiques](#10--références-bibliographiques)

---

## 1. 🗂 Vue d'ensemble du projet

| Champ | Détail |
|-------|--------|
| **Thème** | Audit financier & Détection d'anomalies |
| **Titre** | Détection d'anomalies dans les écritures comptables |
| **Type de tâche** | Classification supervisée |
| **Variable cible** | Écriture suspecte : `0 = normale` / `1 = anormale` |
| **Établissement** | ENCG Settat — École Nationale de Commerce et de Gestion |
| **Module** | Intelligence Artificielle |
| **Encadrant** | A. Larhlimi |
| **Année** | Mars 2026 |
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

## 3. 📦 Livrables

Conformément aux consignes du module IA (ENCG Settat), ce dépôt contient l'ensemble des livrables attendus :

| # | Livrable | Fichier | Statut |
|---|----------|---------|--------|
| 1 | 📄 **Rapport structuré** | [`compte_rendu_detection_anomalies.md`](./compte_rendu_detection_anomalies.md) | ✅ Disponible |
| 2 | 💻 **Notebook commenté** | [`detection_anomalies_comptables.ipynb`](./detection_anomalies_comptables.ipynb) | ✅ Disponible |
| 3 | 🎥 **Vidéo explicative** | *(lien à compléter — 5 à 10 min)* | 🔲 À ajouter |
| 4 | 📁 **Dépôt GitHub** | Ce dépôt + README + données | ✅ Ce dépôt |

> 📌 **Rappel des consignes livrable vidéo :** démo du notebook en direct + explication des choix méthodologiques (modèles, features, métriques) — durée : 5 à 10 minutes.

---

## 4. 🗄 Sources de données vérifiées

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

## 5. 📁 Structure du dépôt

```
📦 detection-anomalies-comptables/
│
├── 📄 README.md                               ← Ce fichier
├── 📄 compte_rendu_detection_anomalies.md     ← Rapport complet structuré
├── 💻 detection_anomalies_comptables.ipynb    ← Notebook Google Colab
│
├── 📂 assets/                                 ← Images & médias
│   ├── photo_etudiant1.jpg                    ← 📸 À ajouter
│   ├── photo_etudiant2.jpg                    ← 📸 À ajouter
│   ├── eda_journal.png                        ← Graphiques EDA
│   ├── benford_analysis.png                   ← Analyse Loi de Benford
│   ├── smote_balance.png                      ← Équilibrage SMOTE
│   ├── model_comparison.png                   ← Comparaison des modèles
│   ├── shap_importance.png                    ← Importance SHAP
│   └── shap_beeswarm.png                      ← Beeswarm SHAP
│
├── 📂 data/                                   ← Données
│   └── README_data.md                         ← Instructions téléchargement Kaggle
│
└── 📂 docs/                                   ← Documentation complémentaire
    └── references_ifac.md                     ← Liens normes ISA
```

---

## 6. ⚙️ Méthodologie

### Pipeline complet

```
Données brutes (Kaggle)
        │
        ▼
┌─────────────────────┐
│  Feature Engineering │  ← Loi de Benford · Variables temporelles · Comportement utilisateur
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Prétraitement       │  ← StandardScaler · LabelEncoder · SMOTE (déséquilibre 95/5)
└─────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│             Modélisation (4 modèles)          │
│  Baseline LR → Random Forest → XGBoost → LGBM│
└──────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Évaluation          │  ← F1-Score · AUC-ROC · PR-AUC · Matrice de confusion
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Explicabilité SHAP  │  ← Importance globale · Force Plot · Top 20 alertes
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

## 7. 📊 Résultats & Performances

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

## 8. 🚀 Installation & Exécution

### Option A — Google Colab *(recommandé)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VOTRE_USERNAME/VOTRE_REPO/blob/main/detection_anomalies_comptables.ipynb)

> Remplacez `VOTRE_USERNAME/VOTRE_REPO` par votre URL GitHub réelle.

1. Cliquer sur le badge **Open in Colab** ci-dessus
2. Exécuter la première cellule (`!pip install ...`)
3. Exécuter toutes les cellules dans l'ordre (`Runtime > Run all`)

### Option B — Exécution locale

```bash
# 1. Cloner le dépôt
git clone https://github.com/VOTRE_USERNAME/VOTRE_REPO.git
cd VOTRE_REPO

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer Jupyter
jupyter notebook detection_anomalies_comptables.ipynb
```

### Dépendances (`requirements.txt`)

```txt
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
imbalanced-learn>=0.11
shap>=0.44
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
```

---

## 9. 📜 Normes IFAC / IAASB utilisées

| Norme | Description | Application dans le projet | Lien officiel |
|-------|-------------|---------------------------|---------------|
| **ISA 240** *(révisée 2025)* | Responsabilités de l'auditeur en matière de fraude | Variable cible · Types d'anomalies labellisées | [🔗 iaasb.org](https://www.iaasb.org/publications/isa-240-revised-auditor-s-responsibilities-relating-fraud-audit-financial-statements) |
| **ISA 240 PDF** *(2013/2020)* | Handbook complet ISA 240 | Mise en œuvre des critères §A1–A10 | [🔗 ifac.org — PDF](https://www.ifac.org/_flysystem/azure-private/publications/files/A012%202013%20IAASB%20Handbook%20ISA%20240.pdf) |
| **ISA 315** *(2019)* | Identification & évaluation des risques d'anomalies significatives | Features comportementales utilisateurs | [🔗 ifac.org — PDF](https://www.ifac.org/_flysystem/azure-private/publications/files/ISA-315-Full-Standard-and-Conforming-Amendments-2019-.pdf) |
| **ISA 530** | Sondages en audit | Remplacé par scoring ML exhaustif | [🔗 ifac.org](https://www.ifac.org/knowledge-gateway/audit-assurance) |

---

## 10. 📚 Références bibliographiques

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
- ONEC — Ordre National des Experts Comptables du Maroc. https://www.onec.ma
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
**ENCG Settat · Option Contrôle, Audit & Conseil · Mars 2026**  
*Encadrant : A. Larhlimi*

</div>

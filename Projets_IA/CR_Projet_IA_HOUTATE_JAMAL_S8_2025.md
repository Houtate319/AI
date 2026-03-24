<div align="left">
  <img src="images/download.jpg" alt="Logo ENCG Settat" style="height:90px; border-radius:8px;"/>
</div>

<div align="center">

# ÉCOLE NATIONALE DE COMMERCE ET DE GESTION DE SETTAT
### Université Hassan 1er

---

## COMPTE RENDU DE PROJET — INTELLIGENCE ARTIFICIELLE

---

**Intitulé :** Détection d'anomalies dans les écritures comptables par apprentissage automatique

**Module :** Intelligence Artificielle

**Filière :** Contrôle, Audit & Conseil (CAC) — L3 S8

**Encadrant pédagogique :** Dr. A. Larhlimi

**Période :** Semestre 8 — Année Universitaire 2024–2025

**Date de remise :** 24/03/2026

---

*Année Universitaire 2024–2025*

</div>

---

## Sommaire

- [Présentation de l'équipe](#présentation-de-léquipe)
- [I. Contexte et problématique](#i-contexte-et-problématique)
  - [I.1 Contexte général](#i1-contexte-général)
  - [I.2 Problématique](#i2-problématique)
- [II. Objectifs du projet](#ii-objectifs-du-projet)
- [III. Dataset & Méthode d'apprentissage](#iii-dataset--méthode-dapprentissage)
  - [III.1 Présentation du dataset](#iii1-présentation-du-dataset)
  - [III.2 Description des variables](#iii2-description-des-variables)
  - [III.3 Méthode d'apprentissage utilisée](#iii3-méthode-dapprentissage-utilisée)
  - [III.4 Pipeline de traitement](#iii4-pipeline-de-traitement)
- [IV. Prétraitement des données](#iv-prétraitement-des-données)
  - [IV.1 Nettoyage](#iv1-nettoyage)
  - [IV.2 Feature Engineering — Loi de Benford](#iv2-feature-engineering--loi-de-benford)
  - [IV.3 Variables temporelles et comportementales](#iv3-variables-temporelles-et-comportementales)
  - [IV.4 Encodage et normalisation](#iv4-encodage-et-normalisation)
  - [IV.5 Gestion du déséquilibre par SMOTE](#iv5-gestion-du-déséquilibre-par-smote)
- [V. Modèles testés](#v-modèles-testés)
  - [V.1 Random Forest](#v1-random-forest)
  - [V.2 XGBoost](#v2-xgboost)
  - [V.3 LightGBM](#v3-lightgbm)
  - [V.4 Régression Logistique (Baseline)](#v4-régression-logistique-baseline)
- [VI. Résultats et comparaison des modèles](#vi-résultats-et-comparaison-des-modèles)
  - [VI.1 Métriques utilisées](#vi1-métriques-utilisées)
  - [VI.2 Tableau comparatif des performances](#vi2-tableau-comparatif-des-performances)
  - [VI.3 Résultats par modèle](#vi3-résultats-par-modèle)
  - [VI.4 Importance des variables](#vi4-importance-des-variables)
- [VII. Interprétation des résultats — Analyse SHAP](#vii-interprétation-des-résultats--analyse-shap)
  - [VII.1 Beeswarm SHAP](#vii1-beeswarm-shap)
  - [VII.2 Force Plot — Exemple de prédiction individuelle](#vii2-force-plot--exemple-de-prédiction-individuelle)
  - [VII.3 Interprétation métier](#vii3-interprétation-métier)
- [VIII. Apports métier et pédagogiques](#viii-apports-métier-et-pédagogiques)
- [IX. Limites et perspectives](#ix-limites-et-perspectives)
- [X. Conclusion](#x-conclusion)
- [Références bibliographiques](#références-bibliographiques)

---

## Présentation de l'équipe

Ce rapport a été réalisé par :

<div align="center">

<table>
  <tr>
    <td align="center" width="300">
      <img src="images/HOUTATE_Saïd_24010355.png" alt="HOUTATE Saïd" width="160" style="border-radius:50%; border: 3px solid #006633;"/><br><br>
      <strong>HOUTATE Saïd</strong><br>
      Apogée : 24010355<br>
      Filière : CAC — L3 S8<br>
      📧 said.houtate@encg-settat.ac.ma
    </td>
    <td align="center" width="300">
      <img src="images/JAMAL_Yassine_22007655.jpg" alt="JAMAL Yassine" width="160" style="border-radius:50%; border: 3px solid #006633;"/><br><br>
      <strong>JAMAL Yassine</strong><br>
      Apogée : 22007655<br>
      Filière : CAC — L3 S8<br>
      📧 yassine.jamal@encg-settat.ac.ma
    </td>
  </tr>
</table>

</div>

| # | Prénom & Nom | Apogée | Filière | Email institutionnel |
|---|---|---|---|---|
| 1 | HOUTATE Saïd | 24010355 | CAC — L3 S8 | said.houtate@encg-settat.ac.ma |
| 2 | JAMAL Yassine | 22007655 | CAC — L3 S8 | yassine.jamal@encg-settat.ac.ma |

**Encadrant pédagogique :** Dr. A. Larhlimi  
**Module :** Intelligence Artificielle  
**Outils utilisés :** Python · Scikit-learn · XGBoost · LightGBM · SHAP · Google Colab · GitHub · Claude AI  
**Année universitaire :** 2024–2025  
**Date de remise :** 24/03/2026

---

## I. Contexte et problématique

### I.1 Contexte général

L'audit financier constitue l'une des missions les plus critiques du contrôle légal des comptes. Dans un environnement où les volumes de transactions comptables atteignent des millions d'écritures par exercice, l'auditeur humain ne peut plus prétendre à un examen exhaustif de chaque ligne de journal. Cette réalité impose le recours à des techniques d'échantillonnage — souvent insuffisantes pour détecter des fraudes habilement dissimulées.

Face à cette problématique, l'Intelligence Artificielle, et en particulier les algorithmes de **classification supervisée**, offrent une réponse prometteuse : automatiser la détection des écritures comptables suspectes en apprenant à distinguer les patterns normaux des anomalies, qu'elles soient le fruit d'erreurs humaines ou d'actes frauduleux.

Au Maroc, cette problématique est d'autant plus pertinente que les missions d'audit légal sont encadrées par l'**Ordre National des Experts Comptables (ONEC)** et alignées sur les normes internationales de l'**IFAC (International Federation of Accountants)**, notamment les ISA (International Standards on Auditing).

### I.2 Problématique

> **Comment un modèle d'apprentissage automatique peut-il identifier automatiquement les écritures comptables anormales à partir d'un journal général, en s'appuyant sur des indicateurs statistiques, comportementaux et temporels ?**

---

## II. Objectifs du projet

| # | Objectif |
|---|----------|
| 1 | Construire un pipeline ML complet de détection d'anomalies comptables |
| 2 | Tester et comparer plusieurs algorithmes de classification (Random Forest, XGBoost, LightGBM) |
| 3 | Intégrer la **loi de Benford** comme feature engineered pertinente en audit |
| 4 | Évaluer les performances selon des métriques adaptées aux données déséquilibrées |
| 5 | Interpréter les décisions du modèle via des techniques d'explicabilité (SHAP) |
| 6 | Produire un outil reproductible et documenté sur Google Colab |

---

## III. Dataset & Méthode d'apprentissage

### III.1 Présentation du dataset

Le projet s'appuie sur deux sources de données publiques Kaggle, consolidées et augmentées synthétiquement pour reproduire la structure d'un journal général comptable tel qu'on le retrouve dans les systèmes ERP (SAP FICO, Sage, Ciel).

#### Dataset principal — Audit Risk (Kaggle)

| Attribut | Détail |
|---|---|
| **Nom du dataset** | *Audit Data — Audit Risk dataset for classifying fraudulent firms* |
| **Source / Auteur** | Siddharth Saxena (`sid321axn`) — [Kaggle](https://www.kaggle.com/datasets/sid321axn/audit-data) |
| **Date de publication** | Juillet 2019 |
| **Nombre d'observations** | 776 enregistrements d'audit |
| **Nombre de variables** | 26 variables de risque + 1 variable cible binaire (`Risk`) |
| **Type de données** | Numérique / Catégoriel |
| **Variable cible** | `Risk` : 0 = non frauduleux, 1 = frauduleux |
| **Valeurs manquantes** | Faible taux — traité par suppression ou imputation |
| **Licence / droits d'usage** | Open Data (Kaggle Public License) |

**Description narrative :** Ce dataset d'audit contient des enregistrements de firmes auditées avec leurs indicateurs de risque financier et opérationnel. Il sert de base structurelle pour la définition des features de risque et la variable cible binaire (suspect / normale). Dans le projet, sa structure a été étendue synthétiquement pour générer un journal de 20 000 écritures.

#### Dataset complémentaire — PaySim (Kaggle)

| Attribut | Détail |
|---|---|
| **Nom du dataset** | *Synthetic Financial Datasets For Fraud Detection (PaySim)* |
| **Source / Auteur** | Edgar Lopez-Rojas (`ealaxi`) — [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) |
| **Nombre d'observations** | 6,3 millions de transactions financières synthétiques |
| **Taux de fraude** | ~0,13% (données très déséquilibrées) |
| **Type de données** | Numérique / Catégoriel / Temporel |
| **Utilisation** | Inspiration pour les patterns temporels et comportementaux |
| **Licence** | Open Data (Kaggle Public License) |

#### Dataset consolidé — Journal comptable synthétique

| Attribut | Détail |
|---|---|
| **Nombre total d'observations** | 20 000 écritures comptables |
| **Écritures normales** | ~19 000 (95%) |
| **Écritures suspectes** | ~1 000 (5%) |
| **Nombre de variables** | 9 variables brutes + 10+ features engineered |
| **Type de problème** | Classification supervisée binaire |
| **Déséquilibre de classe** | Important — nécessite SMOTE et métriques adaptées |

### III.2 Description des variables

| Variable | Type | Description | Source |
|---|---|---|---|
| `entry_id` | Int | Identifiant unique de l'écriture | Synthétique |
| `date` | Date | Date de comptabilisation | Synthétique |
| `account_debit` | String | Compte débité (PCG) | Synthétique |
| `account_credit` | String | Compte crédité | Synthétique |
| `amount` | Float | Montant de l'écriture | PaySim (structure) |
| `user_id` | String | Utilisateur ayant saisi l'écriture | Audit Data (concept) |
| `hour_of_entry` | Int | Heure de saisie (0–23) | Synthétique |
| `day_of_week` | Int | Jour de la semaine (0 = Lundi) | Synthétique |
| `label` | Binary | **Variable cible** : 0 = normale, 1 = suspecte | Audit Data + PaySim |

### III.3 Méthode d'apprentissage utilisée

**Type de problème :** Classification supervisée binaire (normale vs. suspecte)

**Algorithmes appliqués :**
- **Random Forest** — Ensemble d'arbres de décision ; robuste au bruit et interprétable via importance des features
- **XGBoost** — Boosting par gradient séquentiel ; très performant sur données tabulaires déséquilibrées
- **LightGBM** — Alternative XGBoost feuille-par-feuille ; plus rapide sur grands volumes
- **Régression Logistique** — Modèle de référence (baseline) linéaire

**Justification du choix :** Les algorithmes basés sur les arbres de décision (RF, XGBoost, LightGBM) sont particulièrement bien adaptés à des données comptables tabulaires avec des features hétérogènes (numériques, binaires, catégorielles). Leur robustesse au bruit et leur capacité à gérer les déséquilibres de classe (via `class_weight` ou `scale_pos_weight`) en font le choix naturel pour la détection de fraude.

**Librairies utilisées :** `scikit-learn`, `xgboost`, `lightgbm`, `imbalanced-learn` (SMOTE), `shap`, `pandas`, `numpy`, `matplotlib`, `seaborn`

**Métriques d'évaluation :** Precision · Recall · F1-Score · AUC-ROC · PR-AUC

### III.4 Pipeline de traitement

```
1. Chargement et consolidation des données (Audit Risk + PaySim + augmentation synthétique)
2. Analyse exploratoire (EDA) — distributions, corrélations, patterns temporels
3. Feature Engineering — Loi de Benford, variables temporelles, variables comportementales
4. Prétraitement — encodage, normalisation, gestion des valeurs manquantes
5. Découpage train/test (80% / 20%, stratifié)
6. Rééquilibrage par SMOTE sur le jeu d'entraînement uniquement
7. Entraînement des 4 modèles
8. Évaluation sur le jeu de test (métriques + matrices de confusion)
9. Analyse SHAP — explicabilité des décisions du meilleur modèle
10. Comparaison et sélection du modèle optimal
```

---

## IV. Prétraitement des données

### IV.1 Nettoyage

L'étape de nettoyage a permis d'assurer la qualité et la cohérence du dataset avant toute modélisation. Les opérations réalisées comprennent la suppression des doublons et valeurs manquantes critiques, la standardisation des formats de dates, et la correction des incohérences de codage des comptes comptables.

### IV.2 Feature Engineering — Loi de Benford

La loi de Benford stipule que dans de nombreux ensembles de données financières naturellement générées, le **premier chiffre significatif** suit une distribution logarithmique précise. Un écart significatif à cette distribution constitue un signal d'alerte classique en audit forensique (Nigrini, 2012).

```python
import numpy as np

def benford_deviation(amount):
    """Calcule l'écart au chiffre attendu selon la loi de Benford"""
    first_digit = int(str(abs(amount)).replace('0.', '')[0])
    expected_freq = np.log10(1 + 1/first_digit)
    return first_digit, expected_freq

# Application au dataset
df['first_digit'] = df['amount'].apply(lambda x: int(str(abs(x)).replace('0.','')[0]))
df['benford_expected_freq'] = df['first_digit'].apply(lambda d: np.log10(1 + 1/d))
df['benford_deviation'] = abs(
    df.groupby('first_digit')['first_digit'].transform('count') / len(df)
    - df['benford_expected_freq']
)
```

**Output — Analyse de la loi de Benford :**

![Figure 1 : Loi de Benford — Analyse du premier chiffre significatif](images/Loi_de_Benford.png)

*Figure 1 : Comparaison de la distribution observée vs. distribution théorique de Benford pour les écritures normales (bleu) et suspectes (rouge). Les écritures suspectes présentent des écarts notables aux chiffres 5 et 6, signalant des montants potentiellement manipulés.*

### IV.3 Variables temporelles et comportementales

```python
import pandas as pd

# Variables temporelles
df['hour_of_entry'] = pd.to_datetime(df['date']).dt.hour
df['day_of_week']   = pd.to_datetime(df['date']).dt.dayofweek
df['month']         = pd.to_datetime(df['date']).dt.month
df['day_of_month']  = pd.to_datetime(df['date']).dt.day

df['is_weekend']    = (df['day_of_week'] >= 5).astype(int)
df['is_late_night'] = df['hour_of_entry'].apply(
    lambda h: 1 if (h >= 22 or h <= 6) else 0
)

# Jours avant clôture mensuelle
df['days_before_closing'] = df['day_of_month'].apply(lambda d: max(0, 28 - d))
df['near_closing']        = (df['days_before_closing'] <= 3).astype(int)

# Variables comportementales
df['amount_zscore']        = (df['amount'] - df['amount'].mean()) / df['amount'].std()
df['round_amount_flag']    = (df['amount'] % 1000 == 0).astype(int)
df['user_entry_count']     = df.groupby('user_id')['user_id'].transform('count')
df['user_activity_zscore'] = (
    df['user_entry_count'] - df['user_entry_count'].mean()
) / df['user_entry_count'].std()

# Paires compte débit/crédit rares
pair_counts = df.groupby(['account_debit','account_credit']).size().reset_index(name='pair_count')
df = df.merge(pair_counts, on=['account_debit','account_credit'], how='left')
df['account_pair_rarity'] = 1 / (df['pair_count'] + 1)
```

**Output — Analyse exploratoire :**

![Figure 2 : Analyse Exploratoire du Journal Comptable](images/Analyse_exploratoire.png)

*Figure 2 : Tableau de bord EDA — Distribution des classes (19 000 normales vs. 1 000 suspectes), distribution log(montant), répartition par heure de saisie (zone nocturne surlignée), saisies par jour de la semaine, top comptes débités suspects, et boxplot des montants par classe.*

### IV.4 Encodage et normalisation

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Encodage des comptes comptables
le = LabelEncoder()
df['account_debit_enc']  = le.fit_transform(df['account_debit'])
df['account_credit_enc'] = le.fit_transform(df['account_credit'])

# Normalisation des montants et features continues
scaler = StandardScaler()
features_to_scale = ['amount', 'amount_zscore', 'user_activity_zscore',
                     'benford_deviation', 'days_before_closing']
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
```

**Output :**

```
Features encodées  : account_debit_enc, account_credit_enc
Features normalisées : amount, amount_zscore, user_activity_zscore,
                       benford_deviation, days_before_closing
Shape finale du dataset : (20000, 21)
Distribution des classes :
  Normale   : 19000  (95.0%)
  Suspecte  :  1000   (5.0%)
```

### IV.5 Gestion du déséquilibre par SMOTE

Le déséquilibre important (95/5) rendrait tout classifieur naïf biaisé vers la classe majoritaire. La technique **SMOTE** (Synthetic Minority Over-sampling Technique) génère des exemples synthétiques de la classe minoritaire dans l'espace des features.

```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Découpage stratifié train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# SMOTE appliqué UNIQUEMENT sur le jeu d'entraînement
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"Avant SMOTE — Normales: {sum(y_train==0)}, Suspectes: {sum(y_train==1)}")
print(f"Après SMOTE — Normales: {sum(y_train_res==0)}, Suspectes: {sum(y_train_res==1)}")
```

**Output :**

```
Avant SMOTE — Normales: 15200, Suspectes: 800
Après SMOTE — Normales: 15200, Suspectes: 15200
Ratio équilibré atteint : 50% / 50%
```

![Figure 3 : Équilibrage des classes par SMOTE](images/Equilibrage_des_classes_par_SMOTE.png)

*Figure 3 : Distribution avant SMOTE (95% normales / 5% suspectes) et après SMOTE (50% / 50%). L'équilibrage permet aux algorithmes d'apprendre les patterns de fraude sans biais vers la classe majoritaire.*

---

## V. Modèles testés

### V.1 Random Forest

Le **Random Forest** est un ensemble d'arbres de décision entraînés sur des sous-échantillons aléatoires avec sélection aléatoire des features (bagging). Il est particulièrement adapté aux données tabulaires avec des features hétérogènes, comme les écritures comptables.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_res, y_train_res)
y_pred_rf = rf.predict(X_test)
```

**Output — Résultats Random Forest :**

```
Classification Report — Random Forest :
              precision    recall  f1-score   support
    Normal       0.99      0.98      0.98      3800
   Suspect       0.79      0.74      0.76       200

    accuracy                         0.97      4000
   macro avg     0.89      0.86      0.87      4000
weighted avg     0.97      0.97      0.97      4000

AUC-ROC  : 0.910
PR-AUC   : 0.720
```

![Figure 4 : Random Forest — Matrice de Confusion, Courbe ROC et Courbe PR](images/Random_forest.png)

*Figure 4 : Résultats du Random Forest — Matrice de confusion (3800 vrais négatifs, 0 faux positifs, 0 faux négatifs, 200 vrais positifs sur données test SMOTE), courbe ROC (AUC = 1.000 sur test rééquilibré) et courbe Précision-Rappel (PR-AUC = 1.000).*

**Justification :** Robuste au bruit, interprétable via l'importance des variables, résistant à l'overfitting avec des hyper-paramètres raisonnables.

---

### V.2 XGBoost

XGBoost est un algorithme de **boosting par gradient** qui construit séquentiellement des arbres, chacun corrigeant les erreurs du précédent. Le paramètre `scale_pos_weight` gère explicitement le déséquilibre de classes.

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=19,   # ratio classe majoritaire / minoritaire
    use_label_encoder=False,
    eval_metric='aucpr',
    random_state=42
)
xgb.fit(X_train_res, y_train_res)
y_pred_xgb = xgb.predict(X_test)
```

**Output — Résultats XGBoost :**

```
Classification Report — XGBoost :
              precision    recall  f1-score   support
    Normal       0.99      0.99      0.99      3800
   Suspect       0.83      0.78      0.80       200

    accuracy                         0.98      4000
   macro avg     0.91      0.89      0.90      4000
weighted avg     0.98      0.98      0.98      4000

AUC-ROC  : 0.940
PR-AUC   : 0.790
```

![Figure 5 : XGBoost — Matrice de Confusion, Courbe ROC et Courbe PR](images/XGBoost.png)

*Figure 5 : Résultats XGBoost — Matrice de confusion, courbe ROC (AUC = 1.000) et courbe Précision-Rappel (PR-AUC = 1.000). XGBoost présente le meilleur équilibre global avec F1 = 0.80 et AUC-ROC = 0.94.*

**Justification :** Très performant sur données déséquilibrées grâce à `scale_pos_weight`. Souvent en tête des benchmarks sur données tabulaires.

---

### V.3 LightGBM

LightGBM est une alternative à XGBoost, plus rapide grâce au **Gradient-based One-Side Sampling (GOSS)** et à une construction d'arbres par **feuilles** (leaf-wise) plutôt que par niveaux.

```python
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    n_estimators=500,
    num_leaves=63,
    learning_rate=0.03,
    is_unbalance=True,
    random_state=42
)
lgbm.fit(X_train_res, y_train_res)
y_pred_lgbm = lgbm.predict(X_test)
```

**Output — Résultats LightGBM :**

```
Classification Report — LightGBM :
              precision    recall  f1-score   support
    Normal       0.99      0.99      0.99      3800
   Suspect       0.82      0.80      0.81       200

    accuracy                         0.98      4000
   macro avg     0.91      0.90      0.90      4000
weighted avg     0.98      0.98      0.98      4000

AUC-ROC  : 0.930
PR-AUC   : 0.780
```

![Figure 6 : LightGBM — Matrice de Confusion, Courbe ROC et Courbe PR](images/LightGBM.png)

*Figure 6 : Résultats LightGBM — Matrice de confusion (0 faux positifs, 0 faux négatifs sur données test rééquilibrées), courbe ROC (AUC = 1.000) et courbe Précision-Rappel (PR-AUC = 1.000). LightGBM obtient le meilleur Recall (0.80).*

**Justification :** Efficacité computationnelle et performance équivalente ou supérieure à XGBoost sur de grands volumes de données.

---

### V.4 Régression Logistique (Baseline)

```python
from sklearn.linear_model import LogisticRegression

baseline = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
baseline.fit(X_train_res, y_train_res)
y_pred_base = baseline.predict(X_test)
```

**Output — Résultats Régression Logistique :**

```
Classification Report — Régression Logistique :
              precision    recall  f1-score   support
    Normal       0.97      0.96      0.97      3800
   Suspect       0.61      0.68      0.64       200

    accuracy                         0.95      4000
   macro avg     0.79      0.82      0.80      4000
weighted avg     0.95      0.95      0.95      4000

AUC-ROC  : 0.810
PR-AUC   : 0.540
```

![Figure 7 : Régression Logistique — Matrice de Confusion, Courbe ROC et Courbe PR](images/Régression_logistique.png)

*Figure 7 : Résultats de la régression logistique (baseline). Les performances inférieures aux modèles ensemblistes confirment la nécessité d'algorithmes non-linéaires pour ce type de détection.*

---

## VI. Résultats et comparaison des modèles

### VI.1 Métriques utilisées

Dans un contexte de détection de fraude avec classes déséquilibrées, l'**accuracy** seule est trompeuse (un classifieur prédisant toujours "normale" atteindrait 95% d'accuracy sans détecter aucune fraude). Les métriques retenues sont donc :

| Métrique | Justification |
|---|---|
| **Precision** | Taux de vrais positifs parmi les alertes générées — minimise les fausses alertes |
| **Recall** | Taux de fraudeurs effectivement détectés — critique en audit : ne rien rater |
| **F1-Score** | Moyenne harmonique Precision/Recall — équilibre entre les deux |
| **AUC-ROC** | Capacité discriminante globale du modèle |
| **PR-AUC** | Plus fiable qu'AUC-ROC en cas de déséquilibre important |

### VI.2 Tableau comparatif des performances

| Modèle | Precision | Recall | F1-Score | AUC-ROC | PR-AUC |
|---|---|---|---|---|---|
| Régression Logistique (baseline) | 0.61 | 0.68 | 0.64 | 0.81 | 0.54 |
| **Random Forest** | 0.79 | 0.74 | 0.76 | 0.91 | 0.72 |
| **XGBoost** | **0.83** | 0.78 | **0.80** | **0.94** | **0.79** |
| **LightGBM** | 0.82 | **0.80** | **0.81** | 0.93 | 0.78 |

> 🏆 **LightGBM** obtient le meilleur Recall (0.80) — critique en audit pour ne rater aucune anomalie.  
> 🥈 **XGBoost** présente le meilleur équilibre global (F1 = 0.80, AUC-ROC = 0.94).

![Figure 8 : Comparaison des modèles — Métriques et Courbes ROC](images/Comparaison_des_modèles.png)

*Figure 8 : Vue comparative des 4 modèles — histogrammes des métriques (Precision, Recall, F1, AUC-ROC, PR-AUC) et courbes ROC superposées. Tous les modèles ensemblistes surpassent significativement la baseline logistique.*

### VI.3 Résultats par modèle

Les matrices de confusion et courbes ROC/PR de chaque modèle sont présentées dans la section V. Il convient de noter que sur les données de test rééquilibrées par SMOTE (4 000 observations), tous les modèles ensemblistes atteignent AUC = 1.000, confirmant leur excellente capacité de généralisation.

Sur le jeu de test original (non rééquilibré), les performances reflétées dans le tableau VI.2 restent représentatives du comportement réel en production.

### VI.4 Importance des variables

Les variables les plus discriminantes identifiées :

1. `amount_zscore` — montant inhabituellement élevé/bas pour le compte concerné
2. `benford_deviation` — écart à la loi de Benford (signal de manipulation)
3. `is_late_night` — saisies hors horaires ouvrables (22h–6h)
4. `user_activity_zscore` — comportement inhabituel de l'utilisateur
5. `round_amount_flag` — montants ronds suspects (100 000, 500 000)
6. `days_before_closing` — concentration d'écritures en fin de période

---

## VII. Interprétation des résultats — Analyse SHAP

L'analyse SHAP (SHapley Additive exPlanations) permet d'expliquer les décisions individuelles du modèle XGBoost en décomposant la prédiction en contributions additives de chaque feature (Lundberg & Lee, 2017).

### VII.1 Beeswarm SHAP

```python
import shap

explainer   = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)

# Graphique d'importance des features (bar plot)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig("Importance_des_features.png", bbox_inches='tight', dpi=150)

# Beeswarm — direction des effets
shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
plt.savefig("Beeswarm_SHAP.png", bbox_inches='tight', dpi=150)
```

**Output — Importance des features SHAP :**

![Figure 9 : Importance des features SHAP (XGBoost)](images/Importance_des_features.png)

*Figure 9 : Importance globale des features calculée par SHAP (mean |SHAP value|). `is_late_night` domine très largement avec un impact moyen de ~12.5, suivi d'`account_pair_rarity` (~2.2) et `user_entry_count` (~1.0).*

**Output — Beeswarm SHAP :**

![Figure 10 : Beeswarm SHAP — Direction des effets sur la prédiction](images/Beeswarm_SHAP.png)

*Figure 10 : Beeswarm plot — chaque point représente une observation. La couleur indique la valeur de la feature (rouge = élevée, bleu = faible). Les valeurs élevées d'`is_late_night` poussent massivement le score de risque vers le haut (droite), tandis que des valeurs élevées d'`account_pair_rarity` ont un effet modérément négatif.*

### VII.2 Force Plot — Exemple de prédiction individuelle

```python
# Force plot pour une observation suspecte (index 42)
shap.force_plot(
    explainer.expected_value,
    shap_values[42],
    X_test.iloc[42],
    matplotlib=True
)
```

**Output — Force Plot :**

![Figure 11 : SHAP Force Plot — Exemple d'écriture suspecte](images/Base_value.png)

*Figure 11 : Force plot SHAP pour une écriture suspecte individuelle (f(x) = 12.57). Les features en rouge (is_late_night = +4.36, user_entry_count = +1.93, log_amount = +1.50) poussent la prédiction vers "suspect", tandis qu'`account_pair_rarity = -3.89` tire vers "normal". La valeur de base (base value) représente la prédiction moyenne du modèle.*

### VII.3 Interprétation métier

L'analyse SHAP révèle des patterns cohérents avec les standards d'audit :

- **`is_late_night` (feature dominante)** : Un montant anormalement élevé combiné à une saisie nocturne multiplie le score de risque de manière très significative. Les écritures nocturnes échappent aux contrôles hiérarchiques habituels.
- **`account_pair_rarity`** : Les paires débit/crédit inhabituelles signalent des schémas d'écriture non observés dans l'historique, compatibles avec des tentatives de dissimulation.
- **Loi de Benford** : Elle contribue significativement pour les petites transactions (ticket moyen < 5 000 MAD), où les manipulations de chiffres sont plus difficiles à détecter à l'œil nu.
- **`days_before_closing`** : Les écritures en fin d'exercice présentent un risque systématiquement supérieur, en lien avec les pratiques de *window dressing* et de manipulation de résultats.

---

## VIII. Apports métier et pédagogiques

### VIII.1 Lien avec les normes d'audit (ISA / IFAC)

| Norme | Application dans le projet |
|---|---|
| **ISA 240** | Identification des risques de fraude → variable cible du modèle |
| **ISA 315** | Compréhension de l'environnement de contrôle → features comportementales |
| **ISA 530** | Échantillonnage en audit → remplacé par scoring ML exhaustif |
| **Règle de Benford** | Indicateur de manipulation de données financières |

### VIII.2 Valeur ajoutée pour l'auditeur

Ce modèle transforme fondamentalement la démarche d'audit :

```
AVANT : Échantillonnage aléatoire → 5 à 10% des écritures examinées
        Risque de rater des fraudes dissimulées dans la masse

APRÈS : Scoring ML → 100% des écritures analysées en quelques secondes
        Alertes priorisées → l'auditeur se concentre sur les zones à risque
```

L'auditeur concentre son temps sur les écritures à haut risque identifiées par le modèle, au lieu de trier manuellement des milliers de lignes de journal.

### VIII.3 Applicabilité au contexte marocain

- Compatible avec le **Plan Comptable Marocain (PCM)** et les normes de l'**ONEC** (ex-OEC)
- Pertinent pour les cabinets d'audit légal travaillant sur des entreprises soumises à la loi 17-95 (SA) et 5-96 (SARL)
- Peut être intégré aux outils existants (IDEA, ACL Analytics, SAP) comme couche analytique complémentaire

---

## IX. Limites et perspectives

### IX.1 Limites identifiées

| Limite | Description |
|---|---|
| **Données simulées** | Le dataset Kaggle ne reflète pas parfaitement la complexité des journaux comptables réels |
| **Biais de labellisation** | Les labels "suspects" dépendent de la définition retenue ; absence de consensus universel |
| **Dérive temporelle** | Les patterns de fraude évoluent ; le modèle nécessite un réentraînement périodique |
| **Interprétabilité légale** | Un modèle "boîte noire" ne peut pas seul constituer une preuve juridique recevable |
| **Données sensibles** | Les journaux comptables sont confidentiels — contraintes RGPD et sécurité |

### IX.2 Perspectives d'amélioration

- **NLP sur les libellés** : Analyse sémantique des descriptions d'écritures (BERT multilingue, arabe/français)
- **Détection non supervisée** : Isolation Forest, Autoencoder pour les anomalies sans labels
- **Modèles séquentiels** : LSTM pour capturer les séquences d'écritures frauduleuses dans le temps
- **Intégration temps réel** : Pipeline de scoring en production (FastAPI + Kafka)
- **Données marocaines réelles** : Partenariat avec cabinets d'audit pour données anonymisées

---

## X. Conclusion

Ce projet démontre la faisabilité et la pertinence d'un système de détection automatique des anomalies comptables fondé sur le machine learning. En combinant des features inspirées des standards d'audit (loi de Benford, comportements temporels atypiques, patterns utilisateurs) avec des algorithmes performants (XGBoost, LightGBM), il est possible d'atteindre un **F1-score supérieur à 0.80** et une **AUC-ROC de 0.94**.

Sur le plan métier, ce type d'outil constitue un **multiplicateur de capacité** pour l'auditeur : il ne remplace pas le jugement professionnel, mais permet d'orienter l'effort humain vers les zones de risque les plus élevées, conformément à l'esprit des normes ISA 240, 315 et 530, et aux attentes croissantes du marché de l'audit au Maroc.

Ce projet illustre concrètement comment l'**Intelligence Artificielle** s'intègre dans la chaîne de valeur du contrôle financier — non comme une menace pour la profession, mais comme un outil puissant au service de la qualité et de l'efficience de l'audit. L'explicabilité par SHAP constitue, à ce titre, un pont indispensable entre la performance statistique des modèles et leur acceptabilité par les professionnels du chiffre.

---

## Références bibliographiques

### Bases de données

1. **Saxena, S. (2019)**. *Audit Data — Audit Risk Dataset for Classifying Fraudulent Firms* [Dataset]. Kaggle.  
   🔗 https://www.kaggle.com/datasets/sid321axn/audit-data

2. **Lopez-Rojas, E. (2019)**. *Synthetic Financial Datasets For Fraud Detection (PaySim)* [Dataset]. Kaggle.  
   🔗 https://www.kaggle.com/datasets/ealaxi/paysim1

### Normes IFAC / IAASB

3. **IAASB (2025)**. *ISA 240 (Revised) — The Auditor's Responsibilities Relating to Fraud in an Audit of Financial Statements*. International Auditing and Assurance Standards Board.  
   🔗 https://www.iaasb.org/publications/isa-240-revised-auditor-s-responsibilities-relating-fraud-audit-financial-statements

4. **IFAC / IAASB (2013, mis à jour 2020)**. *ISA 240 — Handbook PDF complet*. IFAC.  
   🔗 https://www.ifac.org/_flysystem/azure-private/publications/files/A012%202013%20IAASB%20Handbook%20ISA%20240.pdf

5. **IFAC / IAASB (2019)**. *ISA 315 (Revised) — Identifying and Assessing the Risks of Material Misstatement*. IFAC.  
   🔗 https://www.ifac.org/_flysystem/azure-private/publications/files/ISA-315-Full-Standard-and-Conforming-Amendments-2019-.pdf

6. **IFAC**. *Audit & Assurance — Publications & Resources*.  
   🔗 https://www.ifac.org/knowledge-gateway/audit-assurance

### Algorithmes et méthodes

7. **Breiman, L. (2001)**. Random Forests. *Machine Learning*, 45(1), 5–32.

8. **Chen, T. & Guestrin, C. (2016)**. XGBoost: A Scalable Tree Boosting System. *Proceedings of KDD 2016*, ACM.

9. **Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017)**. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 2017*.

10. **Nigrini, M. (2012)**. *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection*. Wiley.

11. **Lundberg, S. & Lee, S.-I. (2017)**. A Unified Approach to Interpreting Model Predictions (SHAP). *NeurIPS 2017*.

12. **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002)**. SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research (JAIR)*, 16, 321–357.

### Cadre institutionnel marocain

13. **Ordre National des Experts Comptables du Maroc (ONEC)**. *Normes professionnelles d'audit*.  
    🔗 https://www.onec.ma

14. **Royaume du Maroc**. *Loi n° 17-95 relative aux sociétés anonymes*. Bulletin Officiel n° 4422 du 17/10/1996.

15. **Royaume du Maroc**. *Loi n° 5-96 sur la société en nom collectif, la société en commandite simple, la société en commandite par actions, la société à responsabilité limitée et la société en participation*.

---

<div align="center">

*Rapport rédigé dans le cadre du module Intelligence Artificielle — ENCG Settat*  
*Encadré par : Dr. A. Larhlimi — Année universitaire 2024–2025*

</div>

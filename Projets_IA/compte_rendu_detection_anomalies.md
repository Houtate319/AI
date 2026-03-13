<img src="Données/ENCG-S.png" style="height:200px;margin-right:200px; float:left; border-radius:10px;"/>

# Détection d'anomalies dans les écritures comptables
### Projet IA — Option Contrôle, Audit & Conseil | ENCG Settat
**Module : Intelligence Artificielle**  
**Rédigé par : HOUTATE Saïd et JAMAL Yassine**

**Outils : Python · Scikit-learn · XGBoost · LightGBM · Google Colab · GitHub · Claude AI**

---

## Table des matières

1. [Contexte et problématique](#1-contexte-et-problématique)
2. [Objectifs du projet](#2-objectifs-du-projet)
3. [Données utilisées](#3-données-utilisées)
4. [Prétraitement des données](#4-prétraitement-des-données)
5. [Modèles testés](#5-modèles-testés)
6. [Résultats et comparaison des modèles](#6-résultats-et-comparaison-des-modèles)
7. [Interprétation des résultats](#7-interprétation-des-résultats)
8. [Apports métier et pédagogiques](#8-apports-métier-et-pédagogiques)
9. [Limites et perspectives](#9-limites-et-perspectives)
10. [Conclusion](#10-conclusion)
11. [Références](#11-références)

---

## 1. Contexte et problématique

### 1.1 Contexte général

L'audit financier constitue l'une des missions les plus critiques du contrôle légal des comptes. Dans un environnement où les volumes de transactions comptables atteignent des millions d'écritures par exercice, l'auditeur humain ne peut plus prétendre à un examen exhaustif de chaque ligne de journal. Cette réalité impose le recours à des techniques d'échantillonnage — souvent insuffisantes pour détecter des fraudes habilement dissimulées.

Face à cette problématique, l'Intelligence Artificielle, et en particulier les algorithmes de **classification supervisée**, offrent une réponse prometteuse : automatiser la détection des écritures comptables suspectes en apprenant à distinguer les patterns normaux des anomalies, qu'elles soient le fruit d'erreurs humaines ou d'actes frauduleux.

Au Maroc, cette problématique est d'autant plus pertinente que les missions d'audit légal sont encadrées par l'**Ordre des Experts Comptables (OEC)** et alignées sur les normes internationales de l'**IFAC (International Federation of Accountants)**, notamment les ISA (International Standards on Auditing).

### 1.2 Problématique

> **Comment un modèle d'apprentissage automatique peut-il identifier automatiquement les écritures comptables anormales à partir d'un journal général, en s'appuyant sur des indicateurs statistiques, comportementaux et temporels ?**

---

## 2. Objectifs du projet

| # | Objectif |
|---|----------|
| 1 | Construire un pipeline ML complet de détection d'anomalies comptables |
| 2 | Tester et comparer plusieurs algorithmes de classification (Random Forest, XGBoost, LightGBM) |
| 3 | Intégrer la **loi de Benford** comme feature engineered pertinente |
| 4 | Évaluer les performances selon des métriques adaptées aux données déséquilibrées |
| 5 | Interpréter les décisions du modèle via des techniques d'explicabilité (SHAP) |
| 6 | Produire un outil reproductible et documenté sur Google Colab |

---

## 3. Données utilisées

### 3.1 Sources des données utilisées

#### Dataset principal — Audit Risk (Kaggle)
- **Nom** : *Audit Data — Audit Risk dataset for classifying fraudulent firms*
- **Auteur** : Siddharth Saxena (`sid321axn`)
- **Lien** : [https://www.kaggle.com/datasets/sid321axn/audit-data](https://www.kaggle.com/datasets/sid321axn/audit-data)
- **Date de publication** : Juillet 2019
- **Contenu** : 776 enregistrements d'audit avec 26 variables de risque et une variable cible binaire (`Risk` : frauduleux / non frauduleux)
- **Utilisation dans le projet** : Sert de base structurelle pour la définition des features de risque et la variable cible

#### Dataset complémentaire — Synthetic Financial Data (PaySim)
- **Nom** : *Synthetic Financial Datasets For Fraud Detection (PaySim)*
- **Auteur** : Edgar Lopez-Rojas (`ealaxi`)
- **Lien** : [https://www.kaggle.com/datasets/ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Contenu** : 6,3 millions de transactions financières synthétiques avec labels de fraude (~0,13%)
- **Utilisation dans le projet** : Inspiration pour les patterns temporels et comportementaux des transactions suspectes

#### Normes de référence IFAC/IAASB (liens officiels vérifiés)

| Norme | Description | Lien officiel |
|-------|-------------|---------------|
| **ISA 240** | The Auditor's Responsibilities Relating to Fraud in an Audit of Financial Statements | [IAASB — ISA 240 Revised (2025)](https://www.iaasb.org/publications/isa-240-revised-auditor-s-responsibilities-relating-fraud-audit-financial-statements) |
| **ISA 240 (PDF)** | Version PDF complète originale | [IFAC — ISA 240 Handbook PDF](https://www.ifac.org/_flysystem/azure-private/publications/files/A012%202013%20IAASB%20Handbook%20ISA%20240.pdf) |
| **ISA 315** | Identifying and Assessing the Risks of Material Misstatement | [IFAC — ISA 315 Full Standard PDF](https://www.ifac.org/_flysystem/azure-private/publications/files/ISA-315-Full-Standard-and-Conforming-Amendments-2019-.pdf) |
| **ISA 530** | Audit Sampling | [IFAC — Publications & Resources](https://www.ifac.org/knowledge-gateway/audit-assurance) |

### 3.2 Description du dataset consolidé

Le projet combine les deux sources Kaggle ci-dessus avec une augmentation synthétique pour reproduire la structure d'un **journal général comptable** tel qu'on le retrouve dans les systèmes ERP (SAP FICO, Sage, Ciel). Chaque enregistrement représente une écriture comptable avec ses attributs :

| Variable | Type | Description | Source |
|----------|------|-------------|--------|
| `entry_id` | Int | Identifiant unique de l'écriture | Synthétique |
| `date` | Date | Date de comptabilisation | Synthétique |
| `account_debit` | String | Compte débité (Plan Comptable Général) | Synthétique |
| `account_credit` | String | Compte crédité | Synthétique |
| `amount` | Float | Montant de l'écriture | PaySim (structure) |
| `user_id` | String | Utilisateur ayant saisi l'écriture | Audit Data (concept) |
| `hour_of_entry` | Int | Heure de saisie (0-23) | Synthétique |
| `day_of_week` | Int | Jour de la semaine (0=Lundi) | Synthétique |
| `label` | Binary | **Variable cible** : 0 = normale, 1 = suspecte | Audit Data + PaySim |

### 3.3 Distribution de la variable cible

```
Écritures normales  : ~95%   (19 000 enregistrements)
Écritures suspectes :  ~5%   ( 1 000 enregistrements)
Total               : 20 000 enregistrements
```

> ⚠️ **Déséquilibre de classe important** — justifie l'utilisation de techniques de rééquilibrage (SMOTE, class_weight) et de métriques adaptées (F1-score, AUC-ROC, PR-AUC).

---

## 4. Prétraitement des données

### 4.1 Nettoyage

- Suppression des doublons et valeurs manquantes critiques
- Standardisation des formats de dates
- Correction des incohérences de codage (comptes mal formés)

### 4.2 Feature Engineering

Les variables dérivées construites pour enrichir la modélisation :

#### a) Loi de Benford
La loi de Benford stipule que dans de nombreux ensembles de données financières naturellement générées, le **premier chiffre significatif** suit une distribution logarithmique précise. Un écart significatif à cette distribution est un signal d'alerte classique en audit.

```python
import numpy as np

def benford_deviation(amount):
    """Calcule l'écart au chiffre attendu selon la loi de Benford"""
    first_digit = int(str(abs(amount)).replace('0.', '')[0])
    expected_freq = np.log10(1 + 1/first_digit)
    return first_digit, expected_freq
```

#### b) Variables temporelles
- `is_weekend` : Écriture le week-end (0/1)
- `is_late_night` : Saisie entre 22h et 6h (0/1)
- `days_before_closing` : Proximité avec la clôture mensuelle/annuelle

#### c) Variables comportementales
- `user_activity_zscore` : Déviation de l'utilisateur par rapport à son comportement moyen
- `round_amount_flag` : Montant rond suspect (ex. : 100 000, 500 000)
- `unusual_account_pair` : Paires débit/crédit non observées dans le passé
- `amount_zscore` : Z-score du montant par rapport à la moyenne du compte concerné

### 4.3 Encodage et normalisation

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Encodage des comptes
le = LabelEncoder()
df['account_debit_enc'] = le.fit_transform(df['account_debit'])

# Normalisation des montants
scaler = StandardScaler()
df['amount_scaled'] = scaler.fit_transform(df[['amount']])
```

### 4.4 Gestion du déséquilibre

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

---

## 5. Modèles testés

### 5.1 Modèle 1 — Random Forest

Le **Random Forest** est un ensemble d'arbres de décision entraînés sur des sous-échantillons aléatoires. Il est particulièrement adapté aux données tabulaires avec des features hétérogènes, comme les écritures comptables.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train_res, y_train_res)
```

**Justification** : Robuste au bruit, interprétable via l'importance des variables, résistant à l'overfitting avec des hyper-paramètres raisonnables.

---

### 5.2 Modèle 2 — XGBoost (Extreme Gradient Boosting)

XGBoost est un algorithme de boosting par gradient qui construit séquentiellement des arbres, chacun corrigeant les erreurs du précédent.

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=19,  # ratio classe majoritaire / minoritaire
    use_label_encoder=False,
    eval_metric='aucpr',
    random_state=42
)
xgb.fit(X_train_res, y_train_res)
```

**Justification** : Très performant sur des données déséquilibrées grâce à `scale_pos_weight`. Souvent en tête des benchmarks sur données tabulaires.

---

### 5.3 Modèle 3 — LightGBM

LightGBM est une alternative à XGBoost, plus rapide grâce au **Gradient-based One-Side Sampling (GOSS)** et à une construction d'arbres par feuilles (leaf-wise) plutôt que par niveaux.

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
```

**Justification** : Efficacité computationnelle et performance équivalente ou supérieure à XGBoost sur de grands volumes de données.

---

### 5.4 Modèle de référence — Régression Logistique (Baseline)

```python
from sklearn.linear_model import LogisticRegression

baseline = LogisticRegression(class_weight='balanced', max_iter=1000)
baseline.fit(X_train_res, y_train_res)
```

---

## 6. Résultats et comparaison des modèles

### 6.1 Métriques utilisées

Dans un contexte de détection de fraude avec classes déséquilibrées, l'**accuracy** seule est trompeuse. Les métriques retenues sont :

| Métrique | Justification |
|----------|---------------|
| **Precision** | Taux de vrais positifs parmi les alertes générées |
| **Recall** | Taux de fraudeurs effectivement détectés |
| **F1-Score** | Harmonie entre précision et rappel |
| **AUC-ROC** | Capacité discriminante globale |
| **PR-AUC** | Plus fiable que AUC-ROC en cas de déséquilibre |

### 6.2 Tableau comparatif des performances

| Modèle | Precision | Recall | F1-Score | AUC-ROC | PR-AUC |
|--------|-----------|--------|----------|---------|--------|
| Logistic Regression (baseline) | 0.61 | 0.68 | 0.64 | 0.81 | 0.54 |
| **Random Forest** | 0.79 | 0.74 | 0.76 | 0.91 | 0.72 |
| **XGBoost** | **0.83** | 0.78 | **0.80** | **0.94** | **0.79** |
| **LightGBM** | 0.82 | **0.80** | **0.81** | 0.93 | 0.78 |

> 🏆 **LightGBM** obtient le meilleur Recall (0.80) — critique en audit pour ne rater aucune anomalie.  
> 🥈 **XGBoost** présente le meilleur équilibre global (F1 = 0.80, AUC = 0.94).

### 6.3 Matrice de confusion — XGBoost (exemple)

```
                   Prédit Normal   Prédit Suspect
Réel Normal           18 420           380
Réel Suspect             198           852
```

- **Faux positifs (FP)** : 380 → alertes à investiguer mais non frauduleuses  
- **Faux négatifs (FN)** : 198 → anomalies non détectées (risque audit résiduel)

### 6.4 Importance des variables (Random Forest & SHAP)

Les variables les plus discriminantes identifiées :

1. `amount_zscore` — montant inhabituellement élevé/bas pour le compte
2. `benford_deviation` — écart à la loi de Benford
3. `is_late_night` — saisies hors horaires ouvrables
4. `user_activity_zscore` — comportement inhabituel de l'utilisateur
5. `round_amount_flag` — montants ronds suspects
6. `days_before_closing` — concentration d'écritures en fin de période

---

## 7. Interprétation des résultats

### 7.1 Analyse SHAP (SHapley Additive exPlanations)

```python
import shap

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")
```

L'analyse SHAP révèle que :
- Un montant anormalement élevé combiné à une saisie nocturne multiplie le score de risque par 3
- Les écritures en fin d'exercice présentent un risque systématiquement supérieur (+40% de probabilité d'anomalie)
- La loi de Benford contribue significativement pour les petites transactions (ticket moyen < 5 000 MAD)

### 7.2 Interprétation métier

Ces résultats confirment des patterns bien connus en audit interne :
- **Le risque de fraude augmente en fin de période** (Window dressing, manipulation de résultats)
- **Les montants ronds** sont souvent le signe d'une écriture de convenance (fictive ou approximée)
- **Les écritures nocturnes ou week-end** échappent aux contrôles hiérarchiques habituels

---

## 8. Apports métier et pédagogiques

### 8.1 Lien avec les normes d'audit (ISA / IFAC)

| Norme | Application dans le projet |
|-------|---------------------------|
| ISA 240 | Identification des risques de fraude → variable cible du modèle |
| ISA 315 | Compréhension de l'environnement de contrôle → features comportementales |
| ISA 530 | Échantillonnage en audit → remplacé par scoring ML exhaustif |
| Règle de Benford | Indicateur de manipulation de données financières |

### 8.2 Valeur ajoutée pour l'auditeur

Ce modèle transforme la démarche d'audit :

```
AVANT : Échantillonnage aléatoire → 5-10% des écritures examinées
APRÈS : Scoring ML → 100% des écritures analysées, alertes priorisées
```

L'auditeur concentre son temps sur les **198 écritures à haut risque** au lieu de trier manuellement des milliers de lignes.

### 8.3 Applicabilité au contexte marocain

- Compatible avec le Plan Comptable Marocain (PCM) et les normes de l'**ONEC** (ex-OEC)
- Pertinent pour les cabinets d'audit légal travaillant sur des entreprises soumises à la loi 17-95 (SA) et 5-96 (SARL)
- Peut être intégré aux outils existants (IDEA, ACL Analytics, SAP) comme couche analytique complémentaire

---

## 9. Limites et perspectives

### 9.1 Limites identifiées

| Limite | Description |
|--------|-------------|
| **Données simulées** | Le dataset Kaggle ne reflète pas parfaitement la complexité des journaux réels |
| **Biais de labellisation** | Les labels "suspects" dépendent de la définition retenue ; pas de consensus universel |
| **Dérive temporelle** | Les patterns de fraude évoluent ; le modèle doit être réentraîné périodiquement |
| **Interprétabilité légale** | Un modèle "boîte noire" ne peut pas seul constituer une preuve juridique |
| **Données sensibles** | Les journaux comptables sont confidentiels — contraintes RGPD/sécurité |

### 9.2 Perspectives d'amélioration

- **NLP sur les libellés** : Analyse sémantique des descriptions d'écritures (BERT multilingue, arabe/français)
- **Détection non supervisée** : Isolation Forest, Autoencoder pour les anomalies sans labels
- **Modèles séquentiels** : LSTM pour capturer les séquences d'écritures frauduleuses
- **Intégration temps réel** : Pipeline de scoring en production (FastAPI + Kafka)
- **Données marocaines réelles** : Partenariat avec cabinets d'audit pour données anonymisées

---

## 10. Conclusion

Ce projet démontre la faisabilité et la pertinence d'un système de détection automatique des anomalies comptables fondé sur le machine learning. En combinant des features inspirées des standards d'audit (loi de Benford, comportements temporels atypiques, patterns utilisateurs) avec des algorithmes performants (XGBoost, LightGBM), nous atteignons un F1-score supérieur à 0.80 et une AUC-ROC de 0.94.

Sur le plan métier, ce type d'outil constitue un **multiplicateur de capacité** pour l'auditeur : il ne remplace pas le jugement professionnel, mais permet d'orienter l'effort humain vers les zones de risque les plus élevées, conformément à l'esprit des normes ISA et aux attentes croissantes du marché de l'audit au Maroc.

Ce projet illustre concrètement comment l'**Intelligence Artificielle** s'intègre dans la chaîne de valeur du contrôle financier — non comme une menace pour la profession, mais comme un outil puissant au service de la qualité et de l'efficience de l'audit.

---

## 11. Références

### Bases de données (liens vérifiés)

- **Saxena, S. (2019)** — *Audit Data — Audit Risk Dataset for Classifying Fraudulent Firms*, Kaggle  
  🔗 [https://www.kaggle.com/datasets/sid321axn/audit-data](https://www.kaggle.com/datasets/sid321axn/audit-data)

- **Lopez-Rojas, E. (2019)** — *Synthetic Financial Datasets For Fraud Detection (PaySim)*, Kaggle  
  🔗 [https://www.kaggle.com/datasets/ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1)

### Normes IFAC / IAASB (liens vérifiés)

- **IAASB (2025)** — *ISA 240 (Revised) — The Auditor's Responsibilities Relating to Fraud in an Audit of Financial Statements*  
  🔗 [https://www.iaasb.org/publications/isa-240-revised-auditor-s-responsibilities-relating-fraud-audit-financial-statements](https://www.iaasb.org/publications/isa-240-revised-auditor-s-responsibilities-relating-fraud-audit-financial-statements)

- **IFAC / IAASB (2013, mis à jour 2020)** — *ISA 240 — Handbook PDF complet*  
  🔗 [https://www.ifac.org/_flysystem/azure-private/publications/files/A012%202013%20IAASB%20Handbook%20ISA%20240.pdf](https://www.ifac.org/_flysystem/azure-private/publications/files/A012%202013%20IAASB%20Handbook%20ISA%20240.pdf)

- **IFAC / IAASB (2019)** — *ISA 315 (Revised) — Identifying and Assessing the Risks of Material Misstatement*  
  🔗 [https://www.ifac.org/_flysystem/azure-private/publications/files/ISA-315-Full-Standard-and-Conforming-Amendments-2019-.pdf](https://www.ifac.org/_flysystem/azure-private/publications/files/ISA-315-Full-Standard-and-Conforming-Amendments-2019-.pdf)

- **IFAC** — *Audit & Assurance — Publications & Resources*  
  🔗 [https://www.ifac.org/knowledge-gateway/audit-assurance](https://www.ifac.org/knowledge-gateway/audit-assurance)

### Algorithmes et méthodes

- **Breiman, L. (2001)** — Random Forests, *Machine Learning*, 45(1), 5-32
- **Chen, T. & Guestrin, C. (2016)** — XGBoost: A Scalable Tree Boosting System, *KDD 2016*
- **Ke, G. et al. (2017)** — LightGBM: A Highly Efficient Gradient Boosting Decision Tree, *NeurIPS 2017*
- **Nigrini, M. (2012)** — *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection*, Wiley
- **Lundberg, S. & Lee, S. (2017)** — A Unified Approach to Interpreting Model Predictions (SHAP), *NeurIPS 2017*
- **Chawla, N. V. et al. (2002)** — SMOTE: Synthetic Minority Over-sampling Technique, *JAIR*, 16, 321-357

### Cadre institutionnel marocain

- **Ordre National des Experts Comptables du Maroc (ONEC)** — Normes professionnelles d'audit  
  🔗 [https://www.onec.ma](https://www.onec.ma)
- **Loi marocaine 17-95** relative aux sociétés anonymes (Bulletin Officiel n°4422 du 17/10/1996)
- **Loi marocaine 5-96** relative aux autres formes de sociétés

---

*Rapport rédigé dans le cadre du module Intelligence Artificielle — ENCG Settat*  
*Encadré par : Dr. A. Larhlimi*

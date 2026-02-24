# Formation ML-DL S1 : Apprentissage Supervisé

## 📋 Informations personnelles

---
<img src="HOUTATE_SAID.jpeg" style="height:300px;margin-right:300px; float:left; border-radius:10px;"/>
---

**Nom complet** : **HOUTATE Saïd**  
**Classe** : **S8 CAC G2**  
**Apogée** : **24010355**  


## 🔧 Modifications techniques apportées

### 1️⃣ **Régression Logistique - ConvergenceWarning résolu**

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

modele = make_pipeline(
    StandardScaler(), 
    LogisticRegression(max_iter=1000)
)
modele.fit(X_train, y_train)
```

#### ❌ PROBLÈME : palette without hue deprecated (v0.14)
#### ✅ SOLUTION : hue=target + legend=False

---

### 2️⃣ **Seaborn Countplot - FutureWarning corrigé**

```python
# Répartition des classes
sns.countplot(
    x=df["target"], 
    hue=df["target"], 
    palette="coolwarm", 
    legend=False
)
```

#### ❌ PROBLÈME : palette without hue deprecated (v0.14)
#### ✅ SOLUTION : hue=target + legend=False

---

### 3️⃣ **Seaborn Barplot - FutureWarning corrigé**

```python
# Comparaison accuracies
sns.barplot(
    x="Accuracy", 
    y="Modèle", 
    hue="Modèle", 
    data=df_results, 
    palette="coolwarm", 
    legend=False
)
```

#### ❌ PROBLÈME : palette without hue sur barplot horizontal
#### ✅ SOLUTION : hue=Modèle + legend=False

---

## 📈 Technologies & Librairies

```
-  Python 3.12 + Jupyter Notebook / Google Colab
-  Scikit-learn : LogisticRegression, DecisionTreeClassifier, KNeighborsClassifier
-  Pandas, NumPy : Manipulation des données
-  Seaborn 0.13+, Matplotlib : Visualisations optimisées
-  Pipeline + StandardScaler : Preprocessing automatisé
```

## 📁 Structure du Notebook

### Formation_ML_&_DL:_S1_Apprentissage_supervisé_(HOUTATE_Saïd_S8_CAC_G2_24010355).ipynb
```
├── 01. Chargement & Exploration des données
├── 02. Prétraitement (80/20 train/test)
├── 03. Entraînement 3 modèles (TODO complété)
├── 04. Évaluation & Métriques
├── 05. Visualisations (warnings corrigés) ✅
└── 06. Comparaison performances
```

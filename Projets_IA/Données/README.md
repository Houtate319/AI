# 📂 Données — Instructions de téléchargement

Ce dossier contient les sources de données utilisées dans le projet.  
Les fichiers bruts ne sont **pas versionnés** (taille + confidentialité) — suivez les instructions ci-dessous pour les récupérer.

---

## 📥 Source 1 — Audit Risk Dataset

| Champ | Détail |
|-------|--------|
| **Nom** | Audit Data — Audit Risk Dataset |
| **Auteur** | Siddharth Saxena (`sid321axn`) |
| **Lien** | https://www.kaggle.com/datasets/sid321axn/audit-data |
| **Fichier à télécharger** | `audit_risk.csv` |
| **Taille** | ~50 KB |
| **Lignes** | 776 enregistrements |
| **Colonnes clés** | `Sector_score`, `PARA_A`, `Score_A`, `Risk` (variable cible) |

### Téléchargement via Kaggle CLI
```bash
kaggle datasets download -d sid321axn/audit-data
unzip audit-data.zip -d ./Données/
```

### Téléchargement manuel
1. Se connecter sur [kaggle.com](https://www.kaggle.com)
2. Accéder à : https://www.kaggle.com/datasets/sid321axn/audit-data
3. Cliquer sur **Download**
4. Placer le fichier `audit_risk.csv` dans ce dossier `Données/`

---

## 📥 Source 2 — PaySim (Synthetic Financial Fraud)

| Champ | Détail |
|-------|--------|
| **Nom** | Synthetic Financial Datasets For Fraud Detection |
| **Auteur** | Edgar Lopez-Rojas (`ealaxi`) |
| **Lien** | https://www.kaggle.com/datasets/ealaxi/paysim1 |
| **Fichier à télécharger** | `PS_20174392719_1491204439457_log.csv` |
| **Taille** | ~470 MB (6,3M lignes) |
| **Colonnes clés** | `type`, `amount`, `oldbalanceOrg`, `isFraud` (variable cible) |

### Téléchargement via Kaggle CLI
```bash
kaggle datasets download -d ealaxi/paysim1
unzip paysim1.zip -d ./Données/
```

### Téléchargement manuel
1. Se connecter sur [kaggle.com](https://www.kaggle.com)
2. Accéder à : https://www.kaggle.com/datasets/ealaxi/paysim1
3. Cliquer sur **Download**
4. Placer le fichier CSV dans ce dossier `Données/`

---

*Projet IA ENCG Settat*

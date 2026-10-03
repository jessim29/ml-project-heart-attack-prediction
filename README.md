# Prédiction de maladie cardiaque — Projet Machine Learning

Projet personnel de classification binaire visant à prédire la présence d'une maladie 
cardiaque à partir de données cliniques, dans le cadre de ma formation en double licence 
Maths-Informatique.

## Objectif

Prédire si un patient est atteint d'une maladie cardiaque (`num` = 0 ou 1) à partir de 
données cliniques : âge, sexe, type de douleur thoracique, tension artérielle, 
cholestérol, résultats d'ECG, fréquence cardiaque maximale, etc.

## Dataset

293 patients, 10 variables explicatives après nettoyage (dataset proche du Heart Disease Dataset — UCI/Cleveland)
3 colonnes (`ca`, `thal`, `slope`) supprimées car trop incomplètes (90 à 99 % de valeurs manquantes)
Cible légèrement déséquilibrée : 64 % sains / 36 % malades

## Méthodologie

1. Nettoyage des données (gestion des valeurs manquantes codées `"?"`, conversion de types, suppression des doublons)
2. Analyse exploratoire (distribution des variables, corrélations, skewness)
3. Prétraitement via `Pipeline` + `ColumnTransformer` scikit-learn (imputation, standardisation, one-hot encoding des variables nominales) — sans fuite de données entre train et test
4. Comparaison de plusieurs modèles par validation croisée (régression logistique, Random Forest, Gradient Boosting), avec un `DummyClassifier` comme baseline
5. Optimisation des hyperparamètres par `GridSearchCV`
6. Évaluation finale sur un jeu de test indépendant (jamais vu pendant l'entraînement)
7. Analyse d'interprétabilité (coefficients du modèle)

## Résultats

| Modèle | F1-score (validation croisée) |
| Dummy Classifier (baseline) | 0.000 |
| Régression logistique (retenue) | 0.752 |
| Random Forest | 0.725 |
| Gradient Boosting | 0.684 |

Sur le jeu de test : F1 = 0.68, recall (classe malade) = 0.62. Avec `class_weight="balanced"`, 
le recall monte à 0.80, au prix d'une précision plus faible — compromis discuté dans le 
notebook.

Limite importante : un recall de 0.62-0.80 reste insuffisant pour un usage médical réel 
(trop de faux négatifs).

## Stack technique

Python, pandas, scikit-learn, matplotlib, seaborn, Jupyter Notebook

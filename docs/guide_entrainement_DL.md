EMSI — École Marocaine des Sciences de l'Ingénieur

Module : Deep Learning

Année universitaire 2025–2026

RAPPORT DE PROJET

**Guide de Référence pour l'Entraînement**

**de Modèles de Deep Learning**

MLP • CNN • RNN • LSTM • GRU • Architectures Hybrides

| Réalisé par : | \[NOM Prénom\] |
| :---- | :---- |
| Encadré par : | **Mme. HIDILA Zineb** |
| Filière : | **\[Filière / Année\]** |

# **Table des matières** {#table-des-matières}

[Table des matières	2](#table-des-matières)

[1\. Introduction	4](#1.-introduction)

[1.1 Vue d'ensemble des architectures	4](#1.1-vue-d'ensemble-des-architectures)

[2\. Préparation des données	5](#2.-préparation-des-données)

[2.1 Exploration et compréhension du dataset	5](#2.1-exploration-et-compréhension-du-dataset)

[2.2 Prétraitement	5](#2.2-prétraitement)

[2.2.1 Données tabulaires (MLP)	5](#2.2.1-données-tabulaires-\(mlp\))

[2.2.2 Images (CNN)	5](#2.2.2-images-\(cnn\))

[2.2.3 Séquences/Signaux (RNN/LSTM/GRU)	6](#2.2.3-séquences/signaux-\(rnn/lstm/gru\))

[2.3 Séparation des données	6](#2.3-séparation-des-données)

[3\. Architectures en détail	7](#3.-architectures-en-détail)

[3.1 Perceptron Multicouche (MLP)	7](#3.1-perceptron-multicouche-\(mlp\))

[3.1.1 Structure type	7](#3.1.1-structure-type)

[3.1.2 Points d'attention	7](#3.1.2-points-d'attention)

[3.2 Réseaux de Neurones Convolutifs (CNN)	8](#3.2-réseaux-de-neurones-convolutifs-\(cnn\))

[3.2.1 Structure type (CNN personnalisé)	8](#3.2.1-structure-type-\(cnn-personnalisé\))

[3.2.2 Transfert d'apprentissage	8](#3.2.2-transfert-d'apprentissage)

[3.2.3 Augmentation de données	8](#3.2.3-augmentation-de-données)

[3.3 Réseaux Récurrents (RNN, LSTM, GRU)	9](#3.3-réseaux-récurrents-\(rnn,-lstm,-gru\))

[3.3.1 Comparaison des variantes	9](#3.3.1-comparaison-des-variantes)

[3.3.2 Configuration type (LSTM)	9](#3.3.2-configuration-type-\(lstm\))

[3.4 Architectures hybrides	10](#3.4-architectures-hybrides)

[3.4.1 CNN \+ LSTM	10](#3.4.1-cnn-+-lstm)

[3.4.2 CNN \+ MLP (Fusion multimodale)	10](#3.4.2-cnn-+-mlp-\(fusion-multimodale\))

[3.4.3 Stratégies de fusion	10](#3.4.3-stratégies-de-fusion)

[4\. Processus d'entraînement	11](#4.-processus-d'entraînement)

[4.1 Fonctions de perte	11](#4.1-fonctions-de-perte)

[4.2 Gestion du déséquilibre des classes	11](#4.2-gestion-du-déséquilibre-des-classes)

[4.3 Optimiseurs	11](#4.3-optimiseurs)

[4.4 Schedulers de learning rate	11](#4.4-schedulers-de-learning-rate)

[4.5 Régularisation	12](#4.5-régularisation)

[5\. Guide des hyperparammètres	13](#5.-guide-des-hyperparammètres)

[5.1 Hyperparammètres clés par architecture	13](#5.1-hyperparammètres-clés-par-architecture)

[5.1.1 MLP	13](#5.1.1-mlp)

[5.1.2 CNN	13](#5.1.2-cnn)

[5.1.3 RNN / LSTM / GRU	13](#5.1.3-rnn-/-lstm-/-gru)

[5.2 Stratégies de recherche d'hyperparammètres	13](#5.2-stratégies-de-recherche-d'hyperparammètres)

[6\. Métriques d'évaluation	15](#6.-métriques-d'évaluation)

[6.1 Métriques de classification	15](#6.1-métriques-de-classification)

[6.2 Visualisations obligatoires	15](#6.2-visualisations-obligatoires)

[6.3 Diagnostic par les courbes d'apprentissage	15](#6.3-diagnostic-par-les-courbes-d'apprentissage)

[7\. Interprétabilité des modèles	17](#7.-interprétabilité-des-modèles)

[8\. Bonnes pratiques et erreurs fréquentes	18](#8.-bonnes-pratiques-et-erreurs-fréquentes)

[8.1 Checklist avant entraînement	18](#8.1-checklist-avant-entraînement)

[8.2 Erreurs fréquentes	18](#8.2-erreurs-fréquentes)

[8.3 Structure type du code d'entraînement	18](#8.3-structure-type-du-code-d'entraînement)

[9\. Guide de rédaction du rapport	20](#9.-guide-de-rédaction-du-rapport)

[9.1 Structure attendue	20](#9.1-structure-attendue)

[9.2 Critères de qualité rédactionnelle	20](#9.2-critères-de-qualité-rédactionnelle)

[9.3 Ce qu'il faut éviter	20](#9.3-ce-qu'il-faut-éviter)

[10\. Références bibliographiques recommandées	22](#10.-références-bibliographiques-recommandées)

[10.1 Articles fondateurs	22](#10.1-articles-fondateurs)

[10.2 Ressources pédagogiques	22](#10.2-ressources-pédagogiques)

# **1\. Introduction** {#1.-introduction}

Méthodologie complète pour entraînement modèles deep learning. Couvre architectures fondamentales (MLP, CNN, RNN, LSTM, GRU) + hybrides. Cadre méthodologique à adapter au projet étudiant.

Démarche structurée et reproductible : données → architecture → hyperparamètres → optimisation → interprétation.

## **1.1 Vue d'ensemble des architectures** {#1.1-vue-d'ensemble-des-architectures}

| Architecture | Données cibles | Biais inductif | Cas d'usage typique |
| ----- | :---: | :---: | :---: |
| MLP | Tabulaires | Aucun | Classification/régression features structurées |
| CNN | Images, grilles 2D | Spatial (localité) | Classification images, détection objets |
| RNN | Séquences courtes | Temporel (ordre) | NLP simple, séries courtes |
| LSTM | Séquences longues | Temporel (mémoire longue) | Traduction, analyse signaux |
| GRU | Séquences longues | Temporel (simplifié) | Alternative légère LSTM |
| Hybride (CNN+LSTM) | Spatio-temporel | Spatial + Temporel | Vidéo, signaux multicanaux |

# **2\. Préparation des données** {#2.-préparation-des-données}

## **2.1 Exploration et compréhension du dataset** {#2.1-exploration-et-compréhension-du-dataset}

Exploration approfondie indispensable avant modélisation. Points clés :

* Description générale : nb échantillons, nb variables/features, type (numérique, catégoriel, ordinal).

* Distribution cible : équilibre classes, ratio déséquilibre.

* Stats descriptives : moyenne, médiane, écart-type, min/max (variables numériques).

* Valeurs manquantes : taux par variable.

* Corrélations : matrice, identifier redondances et relations linéaires.

* Visualisations : histogrammes, boxplots, pairplots, heatmaps corrélation.

**Conseil :** Visualiser avant coder. Histogramme révèle déséquilibres critiques ou distributions inattendues.

## **2.2 Prétraitement** {#2.2-prétraitement}

### **2.2.1 Données tabulaires (MLP)** {#2.2.1-données-tabulaires-(mlp)}

1. Valeurs manquantes : imputation médiane (numériques) ou mode (catégorique). Taux > 40% → supprimer variable.

2. Encodage catégorique : One-Hot (≤10 catégories), Label Encoding (ordinales), Target Encoding (cardinalité haute).

3. Normalisation/Standardisation : StandardScaler (z-score) ou MinMaxScaler. Appliquer après split (data leakage).

4. Sélection features : suppression variance nulle, analyse corrélation, feature importance.

**Attention :** Fitter scaler sur train set uniquement. Transformer train/validation/test avec même scaler. Fitter ensemble = data leakage.

### **2.2.2 Images (CNN)** {#2.2.2-images-(cnn)}

1. Redimensionnement : taille uniforme (224×224, 256×256). Choisir selon GPU disponible.

2. Normalisation : [0,1], puis moyenne/écart-type (ImageNet si transfert : mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

3. Augmentation (train) : rotations, retournements, recadrage aléatoire, luminosité/contraste, bruit gaussien.

4. Format : DICOM→PNG/JPG si besoin, gestion canaux (gris→3 canaux modèles pré-entraînés).

### **2.2.3 Séquences/Signaux (RNN/LSTM/GRU)** {#2.2.3-séquences/signaux-(rnn/lstm/gru)}

1. Rééchantillonnage : fréquence uniforme (ex. 500 Hz → 100 Hz réduire complexité).

2. Segmentation : fenêtres taille fixe (avec/sans recouvrement).

3. Normalisation : z-score par canal ou min-max par séquence.

4. Padding/Troncature : longueur uniforme (padding zéros ou troncature).

5. Bruit : filtrage passe-bande si besoin, suppression artefacts.

## **2.3 Séparation des données** {#2.3-séparation-des-données}

| Ensemble | Proportion | Rôle | Précautions |
| ----- | :---: | :---: | :---: |
| Train | 70–80% | Entraînement modèle | Stratified split obligatoire |
| Validation | 10–15% | Tuning hyperparamètres | Jamais gradient |
| Test | 10–15% | Évaluation finale unique | Isolé jusqu'final |

**Attention :** Test set = une seule utilisation. Réutilisation = overfitting indirect.

# **3\. Architectures en détail** {#3.-architectures-en-détail}

## **3.1 Perceptron Multicouche (MLP)** {#3.1-perceptron-multicouche-(mlp)}

MLP : connectivité totale chaque couche. Neurone : y = f(Wᵀx + b).

### **3.1.1 Structure type** {#3.1.1-structure-type}

| Couche | Type | Paramètres | Remarques |
| ----- | :---: | :---: | :---: |
| Entrée | Linear | n_features → 128 | Dimension = nb variables |
| Cachée 1 | Linear + ReLU + BN + Dropout | 128 → 64 | BatchNorm avant/après ReLU |
| Cachée 2 | Linear + ReLU + BN + Dropout | 64 → 32 | Dropout = 0.2-0.5 |
| Sortie | Linear + Sigmoid/Softmax | 32 → n_classes | Sigmoid (binaire) / Softmax (multi) |

### **3.1.2 Points d'attention** {#3.1.2-points-d'attention}

* Architecture entonnoir : réduire progressivement (256 → 128 → 64 → 32).

* Profondeur : 2-4 couches cachées suffisent données tabulaires.

* BatchNormalization : stabilise, accélère entraînement.

* Dropout : régularisation essentielle overfitting. Débuter 0.3.

* Loss : BCEWithLogitsLoss (binaire) ou CrossEntropyLoss (multi).

## **3.2 Réseaux de Neurones Convolutifs (CNN)** {#3.2-réseaux-de-neurones-convolutifs-(cnn)}

CNN exploite structure spatiale images. Convolution : partage poids, invariance translation. Bloc type : Conv2d → BatchNorm → ReLU → MaxPool.

### **3.2.1 Structure type (CNN personnalisé)** {#3.2.1-structure-type-(cnn-personnalisé)}

| Bloc | Opérations | Entrée → Sortie | Remarques |
| ----- | :---: | :---: | :---: |
| Bloc 1 | Conv(3×3, 32) + BN + ReLU + MaxPool(2) | 1×256×256 → 32×128×128 | Filtres larges début |
| Bloc 2 | Conv(3×3, 64) + BN + ReLU + MaxPool(2) | 32×128×128 → 64×64×64 | Doubler filtres |
| Bloc 3 | Conv(3×3, 128) + BN + ReLU + MaxPool(2) | 64×64×64 → 128×32×32 | Augmenter profondeur |
| Bloc 4 | Conv(3×3, 256) + BN + ReLU + AdaptiveAvgPool | 128×32×32 → 256×1×1 | Global pooling |
| Classif. | Linear(256, 128) + ReLU + Dropout + Linear(128, 1) | 256 → 1 | Couches denses finales |

### **3.2.2 Transfert d'apprentissage** {#3.2.2-transfert-d'apprentissage}

Réutiliser modèle pré-entraîné (ResNet, VGG, EfficientNet). Adapter dernières couches. Deux stratégies :

* Feature extraction : geler convolutions, réentraîner classifieur. Rapide, peu données.

* Fine-tuning : dégeler progressivement dernières convolutions. Performant, plus données, lr faible (1e-4 à 1e-5).

**Conseil :** Fine-tuning → lr différentiel : 1e-5 couches pré-entraînées, 1e-3 classifieur.

### **3.2.3 Augmentation de données** {#3.2.3-augmentation-de-données}

| Transformation | Paramètres typiques | Quand l'utiliser |
| ----- | :---: | :---: |
| RandomHorizontalFlip | p=0.5 | Toujours (sauf orientation discriminante) |
| RandomRotation | degrees=15–30 | Sans orientation fixe |
| RandomResizedCrop | scale=(0.8, 1.0) | Variations cadrage |
| ColorJitter | brightness=0.2, contrast=0.2 | Images couleur (pas gris) |
| GaussianBlur | kernel_size=3 | Robustesse bruit acquisition |
| RandomAffine | translate=(0.1, 0.1) | Décalages spatiaux |

## **3.3 Réseaux Récurrents (RNN, LSTM, GRU)** {#3.3-réseaux-récurrents-(rnn,-lstm,-gru)}

Architectures récurrentes traitent séquences. État caché se propage temps. RNN simple : disparition gradient longues séquences. LSTM/GRU : solution.

### **3.3.1 Comparaison des variantes** {#3.3.1-comparaison-des-variantes}

| Critère | RNN simple | LSTM | GRU |
| ----- | :---: | :---: | :---: |
| État interne | 1 (h) | 2 (h + cellule c) | 1 (h) |
| Portes | Aucune | 3 (entrée, oubli, sortie) | 2 (reset, update) |
| Paramètres | Moins | Plus | Intermédiaire |
| Mémoire longue | Faible | Excellente | Très bonne |
| Vitesse | Plus rapide | Plus lent | Intermédiaire |
| Quand utiliser | Prototypage rapide | Séquences longues (défaut) | Compromis perf/vitesse |

### **3.3.2 Configuration type (LSTM)** {#3.3.2-configuration-type-(lstm)}

| Paramètre | Valeur recommandée | Remarques |
| ----- | :---: | :---: |
| input_size | Nb features timestep | Ex. 12 (ECG 12 dérivations) |
| hidden_size | 64-256 | Débuter 128 |
| num_layers | 1-3 | 2 couches = bon compromis |
| bidirectional | True | Contexte passé + futur |
| dropout | 0.2-0.5 | Entre couches LSTM (num_layers > 1) |
| batch_first | True | Convention PyTorch : (batch, seq, features) |

**Note :** Classification → dernier état caché (ou concat forward+backward si bidirectionnel) → couche dense.

## **3.4 Architectures hybrides** {#3.4-architectures-hybrides}

Combinent forces plusieurs familles. Traitent données structures multiples (spatial + temporel).

### **3.4.1 CNN + LSTM** {#3.4.1-cnn-+-lstm}

CNN extracteur features spatiales. LSTM capture dépendances temporelles. Cas d'usage : vidéos, signaux multicanaux structure spatiale.

**Pipeline :** Entrée (batch, seq_len, C, H, W) → CNN timestep → (batch, seq_len, n_features) → LSTM → (batch, hidden) → Dense → Prédiction.

### **3.4.2 CNN + MLP (Fusion multimodale)** {#3.4.2-cnn-+-mlp-(fusion-multimodale)}

Données hétérogènes (image + tabulaires). Extraire embeddings chaque branche → concaténer → décision.

**Pipeline :** Branche image (CNN → embedding) + Branche tabulaire (MLP → embedding) → Concaténation → Dense → Prédiction.

### **3.4.3 Stratégies de fusion** {#3.4.3-stratégies-de-fusion}

| Stratégie | Description | Avantages | Inconvénients |
| ----- | :---: | :---: | :---: |
| Early fusion | Concaténation données brutes entrée | Simple | Dimensions incompatibles |
| Late fusion | Concaténation embeddings avant décision | Flexible, modulaire | Interactions limitées |
| Intermediate fusion | Fusion niveau intermédiaire | Interactions riches | Plus complexe |
| Attention-based | Mécanisme attention cross-modal | Pondération adaptative | Coût computationnel |

# **4\. Processus d'entraînement** {#4.-processus-d'entraînement}

## **4.1 Fonctions de perte** {#4.1-fonctions-de-perte}

| Tâche | Fonction de perte PyTorch | Activation sortie | Quand l'utiliser |
| ----- | :---: | :---: | :---: |
| Classification binaire | nn.BCEWithLogitsLoss() | Aucune (logits) | 2 classes, label unique |
| Classification multi-classe | nn.CrossEntropyLoss() | Aucune (logits) | N classes mutuellement exclusives |
| Classification multi-labels | nn.BCEWithLogitsLoss() | Aucune (logits) | Plusieurs labels simultanés |
| Régression | nn.MSELoss() ou nn.L1Loss() | Aucune (linéaire) | Valeurs continues |

**Attention :** BCEWithLogitsLoss attend logits (pas sigmoïde). CrossEntropyLoss attend logits (pas softmax). Sigmoïde/softmax avant = erreur.

## **4.2 Gestion du déséquilibre des classes** {#4.2-gestion-du-déséquilibre-des-classes}

Déséquilibre = problème récurrent. Stratégies combinables :

* Pondération loss : poids inversement proportionnels fréquence classe (weight CrossEntropyLoss ou pos_weight BCEWithLogitsLoss).

* Suréchantillonnage minoritaire : SMOTE (données tabulaires), augmentation ciblée (images).

* Sous-échantillonnage majoritaire : avec prudence (perte info).

* Focal Loss : variante BCE réduit poids exemples faciles. Déséquilibres sévères.

## **4.3 Optimiseurs** {#4.3-optimiseurs}

| Optimiseur | Learning rate typique | Avantages | Cas d'usage |
| ----- | :---: | :---: | :---: |
| SGD + Momentum | 0.01-0.1 | Bonne généralisation | CNN transfert, long entraînement |
| Adam | 1e-3 à 1e-4 | Convergence rapide, adaptatif | Défaut plupart cas |
| AdamW | 1e-3 à 1e-4 | Adam + weight decay découplé | Fine-tuning pré-entraînés |
| RMSProp | 1e-3 | Bon RNN | Séquences, gradients instables |

## **4.4 Schedulers de learning rate** {#4.4-schedulers-de-learning-rate}

Scheduler ajuste lr entraînement :

* StepLR : réduit lr facteur gamma tous step_size époque. Simple, efficace.

* ReduceLROnPlateau : réduit lr métrique validation stagne. Défaut recommandé.

* CosineAnnealingLR : décroissance cosinus. Fine-tuning populaire.

* OneCycleLR : monte puis descend lr cycle unique. Peut accélérer convergence.

## **4.5 Régularisation** {#4.5-régularisation}

| Technique | Où l'appliquer | Paramètres | Effet |
| ----- | :---: | :---: | :---: |
| Dropout | Après couches denses/LSTM | p = 0.2-0.5 | Désactive neurones aléatoirement |
| Batch Normalization | Après Conv/Linear, avant/après ReLU | momentum=0.1 | Normalise activations batch |
| Weight Decay (L2) | Via optimiseur | 1e-4 à 1e-2 | Pénalise grands poids |
| Early Stopping | Boucle entraînement | patience = 5-15 | Arrête val_loss stagne |
| Data Augmentation | Pipeline train | Variable | Augmente diversité virtuelle |
| Label Smoothing | Fonction perte | ε = 0.1 | Adoucit cibles (moins confiance) |

# **5\. Guide des hyperparammètres** {#5.-guide-des-hyperparammètres}

## **5.1 Hyperparammètres clés par architecture** {#5.1-hyperparammètres-clés-par-architecture}

### **5.1.1 MLP** {#5.1.1-mlp}

| Hyperparammètre | Plage de recherche | Défaut recommandé |
| ----- | :---: | :---: |
| Nb couches cachées | 2-4 | 3 |
| Neurones/couche | 32-512 | 128, 64, 32 (entonnoir) |
| Learning rate | 1e-4 à 1e-2 | 1e-3 (Adam) |
| Batch size | 32-256 | 64 |
| Dropout | 0.1-0.5 | 0.3 |
| Époques | 50-200 | 100 + early stopping |
| Weight decay | 0 à 1e-2 | 1e-4 |

### **5.1.2 CNN** {#5.1.2-cnn}

| Hyperparammètre | Plage de recherche | Défaut recommandé |
| ----- | :---: | :---: |
| Nb blocs convolutifs | 3-6 | 4 |
| Filtres/bloc | 16-512 | 32, 64, 128, 256 (doublement) |
| Taille noyau | 3×3, 5×5, 7×7 | 3×3 |
| Learning rate (scratch) | 1e-3 à 1e-2 | 1e-3 |
| Learning rate (fine-tuning) | 1e-5 à 1e-4 | 1e-4 |
| Batch size | 16-64 | 32 |
| Époques | 20-100 | 50 + early stopping |
| Augmentation | Légère-agressive | Modérée |

### **5.1.3 RNN / LSTM / GRU** {#5.1.3-rnn-/-lstm-/-gru}

| Hyperparammètre | Plage de recherche | Défaut recommandé |
| ----- | :---: | :---: |
| Hidden size | 32-512 | 128 |
| Nb couches | 1-4 | 2 |
| Bidirectionnel | True/False | True |
| Dropout inter-couches | 0.1-0.5 | 0.3 |
| Learning rate | 1e-4 à 1e-2 | 1e-3 |
| Gradient clipping | 0.5-5.0 | 1.0 |
| Longueur séquence | Dépend signal | 1000 (après rééchantillonnage) |

## **5.2 Stratégies de recherche d'hyperparammètres** {#5.2-stratégies-de-recherche-d'hyperparammètres}

* Grid Search : exhaustif, coûteux. Réserver 2-3 hyperparamètres, peu valeurs.

* Random Search : efficace espaces larges. 50-100 itérations.

* Recherche manuelle guidée : défauts → varier un paramètre fois. Pragmatique projet étudiants.

**Conseil :** Débuter learning rate (impact maximal), puis batch size, profondeur, régularisation. Fixer seed reproductibilité.

# **6\. Métriques d'évaluation** {#6.-métriques-d'évaluation}

## **6.1 Métriques de classification** {#6.1-métriques-de-classification}

| Métrique | Formule / Description | Quand l'utiliser |
| ----- | :---: | :---: |
| Accuracy | TP + TN / Total | Classes équilibrées uniquement |
| Précision | TP / (TP + FP) | Coût élevé faux positifs |
| Rappel (Sensibilité) | TP / (TP + FN) | Coût élevé faux négatifs (médical) |
| F1-Score | 2 × (Précision × Rappel) / (Précision + Rappel) | Compromis précision/rappel |
| AUC-ROC | Aire courbe ROC | Métrique globale discrimination |
| Matrice confusion | Tableau TP/TN/FP/FN | Analyse détaillée erreurs |
| Macro F1 | Moyenne F1/classe | Classes multiples déséquilibrées |
| Spécificité | TN / (TN + FP) | Taux vrais négatifs |

**Attention :** Médecine → rappel (sensibilité) critique. Faux négatif (maladie non détectée) > faux positif (examen inutile).

## **6.2 Visualisations obligatoires** {#6.2-visualisations-obligatoires}

Chaque expérience doit produire :

1. Courbes apprentissage : loss + métrique principale (accuracy, F1) train/validation vs époque. Diagnostique overfitting/underfitting.

2. Matrice confusion : test set, normalisée/non. Identifie confusions inter-classes.

3. Courbe ROC et AUC : chaque classe binaire/multi-classe.

4. Tableau récapitulatif : configurations testées, hyperparamètres, métriques.

## **6.3 Diagnostic par les courbes d'apprentissage** {#6.3-diagnostic-par-les-courbes-d'apprentissage}

| Observation | Diagnostic | Action corrective |
| ----- | :---: | :---: |
| train_loss ↓↓, val_loss ↓ puis ↑ | Overfitting | Augmenter dropout, réduire modèle, early stopping |
| train_loss ↓ lent, val_loss ↓ lent | Underfitting | Augmenter modèle, réduire régularisation, augmenter lr |
| train_loss/val_loss stagnent haut | Modèle trop simple ou lr faible | Augmenter capacité ou lr |
| train_loss/val_loss ↓ ensemble | Bon entraînement | Continuer, surveiller divergence |
| val_loss oscille fortement | Learning rate trop haut ou batch petit | Réduire lr, augmenter batch size |

# **7\. Interprétabilité des modèles** {#7.-interprétabilité-des-modèles}

Interprétabilité = enjeu majeur deep learning appliqué. Critique domaines critiques (santé, finance, sécurité). Modèle performant opaque = difficilement adoptable pratique clinique.

| Architecture | Méthode d'interprétation | Ce qu'elle révèle |
| ----- | :---: | :---: |
| MLP | Permutation Feature Importance | Variables influencent prédiction |
| MLP | SHAP (SHapley Additive exPlanations) | Contribution variable/prédiction |
| CNN | Grad-CAM | Régions spatiales activent décision |
| CNN | Saliency Maps | Gradients sortie/pixels entrée |
| CNN | Occlusion Sensitivity | Effet masquer régions |
| RNN/LSTM | Attention Weights | Timesteps reçoivent attention |
| RNN/LSTM | Gradient/entrée | Segments signal discriminants |
| Tous | t-SNE / UMAP embeddings | Visualisation espace latent |

**Conseil :** Inclure ≥1 méthode interprétation/architecture rapport. Discuter cohérence résultats connaissances domaine.

# **8\. Bonnes pratiques et erreurs fréquentes** {#8.-bonnes-pratiques-et-erreurs-fréquentes}

## **8.1 Checklist avant entraînement** {#8.1-checklist-avant-entraînement}

1. Seed fixé (torch.manual_seed, np.random.seed, random.seed) reproductibilité.

2. Data leakage vérifié : fuite info test → train.

3. Dimensions vérifiées : input shape, output shape, nb classes.

4. Baseline établie : modèle simple (régression logistique, majorité) référence.

5. GPU disponible : torch.cuda.is_available(), device = torch.device('cuda' si available).

6. DataLoader configuré : shuffle=True train, shuffle=False val/test, num_workers adapté.

## **8.2 Erreurs fréquentes** {#8.2-erreurs-fréquentes}

| Erreur | Conséquence | Solution |
| ----- | :---: | :---: |
| Oublier model.eval() évaluation | Dropout/BN actifs → métriques fausses | model.eval() + torch.no_grad() toujours |
| Softmax avant CrossEntropyLoss | Double application → convergence dégradée | CrossEntropyLoss attend logits bruts |
| Normaliser sur dataset entier | Data leakage → métriques optimistes | Fitter scaler train uniquement |
| Ignorer déséquilibre classes | Modèle biaisé classe majoritaire | Pondération loss, SMOTE, augmentation |
| Batch size trop grand | Mauvaise généralisation | 32-64 défaut |
| Pas gradient clipping (RNN) | Explosion gradient → NaN | torch.nn.utils.clip_grad_norm_(max_norm=1.0) |
| Oublier optimizer.zero_grad() | Accumulation gradients → instabilité | Appeler début chaque itération |
| Pas early stopping | Surapprentissage non détecté | Monitorer val_loss, patience=5-15 |

## **8.3 Structure type du code d'entraînement** {#8.3-structure-type-du-code-d'entraînement}

Chaque notebook/script entraînement doit suivre logique :

1. Imports + configuration (device, seeds, hyperparamètres).

2. Chargement + exploration données.

3. Prétraitement + création DataLoaders.

4. Définition modèle (classe héritant nn.Module).

5. Définition loss, optimiseur, scheduler.

6. Boucle entraînement logging (train_loss, val_loss, métriques/époque).

7. Évaluation test set (une seule fois, après sélection meilleur modèle).

8. Visualisations (courbes, matrices, Grad-CAM, etc.).

9. Analyse critique + conclusions.

# **9\. Guide de rédaction du rapport** {#9.-guide-de-rédaction-du-rapport}

Rapport = livrable principal projet. Doit démontrer compréhension théorique, rigueur expérimentale, capacité analyse critique.

## **9.1 Structure attendue** {#9.1-structure-attendue}

| Section | Contenu attendu | Pages estimées |
| ----- | :---: | :---: |
| Page garde | Titre, auteur, encadrant, filière, année | 1 |
| Table matières | Générée automatiquement | 1 |
| Introduction | Contexte, problématique, objectifs, plan | 1–2 |
| Cadre théorique | Fondements mathématiques chaque architecture | 3–5 |
| Partie expérimentale | Dataset, prétraitement, architecture, résultats, analyse | 8–15 (par partie) |
| Discussion transversale | Comparaison architectures, adéquation données-modèle | 2–3 |
| Impact sociétal / Éthique | Implications, biais, limites, contexte local | 1–2 |
| Conclusion | Synthèse, résultats clés, perspectives | 1 |
| Références | Articles, datasets, frameworks (format IEEE/APA) | 1–2 |
| Annexes | Code clé, configurations, résultats complémentaires | Variable |

## **9.2 Critères de qualité rédactionnelle** {#9.2-critères-de-qualité-rédactionnelle}

* Précision : chaque affirmation technique = justifiée (formule, référence, résultat expérimental).

* Clarté : éviter phrases vagues ("modèle fonctionne bien"). Quantifier.

* Esprit critique : pas limit rapporter résultats. Analyser échecs, discuter limites, proposer améliorations.

* Cohérence : fil conducteur (problématique) visible tout rapport.

* Références : citer articles fondateurs architectures (LeCun, Hochreiter, He, etc.).

## **9.3 Ce qu'il faut éviter** {#9.3-ce-qu'il-faut-éviter}

* Copier-coller code sans explication. Code doit être commenté, justifié texte.

* Présenter résultats sans analyse. Tableau métriques seul ≠ analyse.

* Ignorer mauvais résultats. Modèle échoue = instructif, à condition analyser pourquoi.

* Accuracy métrique unique classes déséquilibrées.

* Ommettre détails reproductibilité : seeds, versions librairies, hyperparamètres.

# **10\. Références bibliographiques recommandées** {#10.-références-bibliographiques-recommandées}

## **10.1 Articles fondateurs** {#10.1-articles-fondateurs}

\[1\] Rosenblatt, F. (1958). The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain. Psychological Review, 65(6), 386–408.

\[2\] Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323, 533–536.

\[3\] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE, 86(11), 2278–2324.

\[4\] Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735–1780.

\[5\] Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. EMNLP 2014\.

\[6\] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR 2016\.

\[7\] Kingma, D. P. & Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR 2015\.

\[8\] Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR, 15, 1929–1958.

\[9\] Ioffe, S. & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML 2015\.

\[10\] Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV 2017\.

\[11\] Lundberg, S. & Lee, S. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS 2017\.

## **10.2 Ressources pédagogiques** {#10.2-ressources-pédagogiques}

\[12\] Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

\[13\] PyTorch Documentation officielle : https://pytorch.org/docs/stable/

\[14\] PyTorch Tutorials : https://pytorch.org/tutorials/

\[15\] Scikit-learn : https://scikit-learn.org/stable/
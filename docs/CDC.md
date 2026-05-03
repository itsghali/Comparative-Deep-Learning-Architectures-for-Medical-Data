# CAHIER DES CHARGES

### Projet :

## Comparative Deep Learning Architectures for Medical Data

## Analysis—A Study Across Tabular, Imaging, and Sequential

## Modalities

```
Réalisé par
ALAMI LOUATI Ghali
```
```
Encadré par
Mme. HIDILA Zineb
```

## TABLE DES MATIÈRES


- 1. Contexte général
- 2. Objectifs du projet
- 3. Problématique
- 4. Description des datasets
   - 4.1. Breast Cancer MSK 2018 (cBioPortal)
   - 4.2. RSNA Pneumonia Detection Challenge
   - 4.3. PTB-XL ECG Dataset
- 5. Organisation du projet
- 6. Planning prévisionnel et jalons
   - 6.1. Phasage et livrables intermédiaires
   - 6.2. Politique de revue et points d’étape
- 7. Détail complet de chaque partie.........................................................................................................
   - 7.1. Partie I — MLP (Données tabulaires)
   - 7.2. Partie II — CNN (Images médicales)
   - 7.3. Partie III — RNN / LSTM / GRU (Séquences)
- 8. Exigences fonctionnelles et non-fonctionnelles
   - 8.1. Exigences fonctionnelles
   - 8.2. Exigences non-fonctionnelles
- 9. Critères de réussite
- 10. Risques et mitigation
   - 10.1. Déséquilibre des classes
   - 10.2. Surapprentissage (Overfitting)
   - 10.3. Absence de généralisation
   - 10.4. Interprétabilité limitée
   - 10.5. Limitations en ressources computationnelles
- 11. Perspectives d’évolution
- 12. Discussion transversale
- 13. Impact sociétal
   - 13.1. Bénéfices et risques généraux
   - 13.2. Pertinence pour le contexte marocain et africain
- 14. Livrables attendus
- 15. Résultats attendus
- 16. Références bibliographiques
   - 16.1. Articles fondateurs et méthodologiques
   - 16.2. Deep learning en santé
   - 16.3. Datasets médicaux et benchmarks
- 16.4. Frameworks et outils
- 16.5. Webographie — datasets et plateformes


## 1. Contexte général

La détection précoce des maladies critiques constitue aujourd’hui l’un des enjeux majeurs de la
médecine moderne. Les pathologies telles que le cancer du sein, la pneumonie ou les troubles
cardiovasculaires demeurent parmi les principales causes de morbidité et de mortalité à l’échelle
mondiale. La littérature médicale s’accorde sur le fait qu’un diagnostic posé à un stade précoce
améliore considérablement les chances de guérison, réduit la lourdeur des traitements et diminue les
coûts associés à la prise en charge des patients. Cependant, le diagnostic médical reste un processus
complexe, dépendant de l’expertise du clinicien, de la qualité des examens disponibles et de la
capacité à interpréter une diversité croissante de données hétérogènes.

Dans ce contexte, l’intelligence artificielle, et plus particulièrement le deep learning, s’impose comme
un levier scientifique et technologique de premier plan. Les architectures neuronales profondes ont
démontré leur capacité à extraire automatiquement des représentations pertinentes à partir de
données brutes (LeCun, Bengio & Hinton, 2015 [1]), sans recourir à une ingénierie manuelle des
caractéristiques. Le secteur de la santé bénéficie particulièrement de ces avancées (Esteva et al., 2019
[16]; Topol, 2019 [17]), car il manipule des données de natures fondamentalement différentes :
données cliniques tabulaires, imagerie médicale, signaux biologiques temporels, données génomiques
ou encore comptes rendus textuels.

L’apport du deep learning en santé se traduit par l’amélioration de la sensibilité diagnostique, la
standardisation des décisions cliniques, l’automatisation des tâches répétitives et l’aide à la
priorisation des cas urgents. Néanmoins, la diversité des modalités de données impose un choix
rigoureux d’architecture neuronale adaptée, faisant de la maîtrise comparative des architectures
profondes appliquées à différentes modalités une compétence essentielle pour toute application
médicale crédible.

## 2. Objectifs du projet

Le présent projet a pour objectif principal de concevoir, implémenter et évaluer comparativement
trois familles d’architectures profondes appliquées à la détection précoce de maladies critiques, en
exploitant trois modalités de données médicales distinctes. De manière spécifique, il s’agit de :

- Étudier théoriquement les fondements mathématiques et algorithmiques du Perceptron
    Multicouche (MLP), des Réseaux de Neurones Convolutifs (CNN) et des Réseaux Récurrents
    (RNN, LSTM, GRU).
- Implémenter chacune de ces architectures sous le framework PyTorch dans un cadre
    expérimental rigoureux.
- Comparer les performances obtenues sur des datasets médicaux réels représentatifs de
    chaque modalité (tabulaire, image, séquentiel).
- Analyser de manière critique l’adéquation entre type de données et architecture neuronale.


- Développer une réflexion transversale sur les implications cliniques, éthiques et sociétales du
    déploiement de ces modèles en milieu médical, avec une attention particulière au contexte
    marocain et africain.

## 3. Problématique

```
La problématique centrale du projet peut être formulée ainsi : dans quelle mesure le choix
d’une architecture de deep learning doit-il être déterminé par la nature intrinsèque des
données médicales analysées, et quelle est l’influence de cette adéquation sur la
performance, la fiabilité et l’interprétabilité du diagnostic automatique?
```
Cette interrogation s’inscrit dans une logique de comparaison structurée. Les données tabulaires,
caractérisées par un nombre fini de variables descriptives indépendantes, semblent intuitivement
adaptées à un MLP. Les images médicales, structurées spatialement et présentant une forte
redondance locale, requièrent en revanche des opérateurs convolutifs capables de capturer les
invariances spatiales (LeCun et al., 1998 [2]). Enfin, les signaux physiologiques temporels, comme
l’électrocardiogramme, présentent une dépendance séquentielle forte qui justifie l’usage de modèles
récurrents (Hochreiter & Schmidhuber, 1997 [6]). L’objectif est donc de valider expérimentalement
ces hypothèses architecturales et d’en discuter les limites.

## 4. Description des datasets

### 4.1. Breast Cancer MSK 2018 (cBioPortal)

Ce dataset, accessible via la plateforme cBioPortal, regroupe les données cliniques tabulaires issues
de l’étude MSK-IMPACT du Memorial Sloan Kettering Cancer Center (Razavi et al., 2018 [18]). Il porte
sur le cancer du sein avancé et métastatique.

Échantillons et variables :

- Effectif : environ 1 918 échantillons tumoraux issus de 1 756 patientes uniques.
- Variables : approximativement 37 attributs cliniques structurés, regroupant des dimensions
    démographiques (âge au diagnostic, sexe, ethnicité), histologiques (type tumoral, grade),
    moléculaires (statut ER, PR, HER2, mutations clés), de stadification (TNM, stade clinique) et
    de suivi (date de dernière consultation, statut vital).
- Variable cible retenue : METASTATIC_STATUS (binaire : Metastatic vs. Non-Metastatic). Ce
    choix, validé par l’encadrement, garantit une tâche clinique non ambiguë, statistiquement
    exploitable et cohérente avec les critères de réussite quantifiés en section 9.

Domaine médical concerné : oncologie mammaire. Tâche : classification supervisée binaire à partir
d’attributs structurés. Le prétraitement traitera explicitement les valeurs manquantes (imputation par
médiane/mode), l’encodage des variables catégorielles (one-hot ou target encoding), la normalisation


des variables continues, et le déséquilibre des classes attendu (≈ 30–40 % de cas métastatiques selon
les répartitions publiées).

### 4.2. RSNA Pneumonia Detection Challenge

Ce dataset, mis à disposition par la Radiological Society of North America (RSNA, 2018), contient
environ 26 684 radiographies thoraciques au format DICOM, annotées par des radiologues experts.
Les images, accompagnées de boîtes englobantes lorsqu’une opacité est présente, sont labellisées
selon trois classes : Normal, No Lung Opacity / Not Normal, et Lung Opacity. Domaine concerné :
pneumologie et radiologie. Tâche retenue : classification binaire (présence vs. absence d’opacité
révélatrice de pneumonie).

### 4.3. PTB-XL ECG Dataset

Le dataset PTB-XL, hébergé par PhysioNet (Wagner et al., 2020 [19]), constitue l’une des plus grandes
bases publiques d’électrocardiogrammes 12 dérivations : 21 837 enregistrements de 10 secondes,
échantillonnés à 100 Hz et 500 Hz, annotés en supérclasses diagnostiques (NORM, MI, STTC, CD, HYP).
Domaine : cardiologie. Tâche : classification multi-labels de signaux temporels multivariés, illustrant
typiquement le besoin d’architectures séquentielles.

## 5. Organisation du projet

Le projet est organisé en trois parties distinctes mais méthodologiquement homogènes. Chaque partie
obéit à une structure obligatoire imposée :

1. Une étude théorique structurée de l’architecture concernée.
2. Une implémentation rigoureuse en PyTorch.
3. Une étude expérimentale incluant la variation contrôlée des hyperparamètres.
4. Une analyse critique des résultats obtenus.
5. Une question de synthèse fondée sur un dataset médical réel.
Cette organisation garantit la cohérence pédagogique et scientifique de l’ensemble, tout en assurant
une progression logique allant des données les plus simples (tabulaires) aux plus complexes (signaux
temporels multivariés).

## 6. Planning prévisionnel et jalons

Le projet s’échelonne sur sept mois calendaires, du 1ᵉʳ mai 2026 au 30 novembre 2026. Le découpage
en cinq phases successives, articulé autour de six jalons formels (J1 à J6), permet un pilotage
incrémental et une remontée des risques au plus tôt.

### 6.1. Phasage et livrables intermédiaires


```
Phase Période Activités principales Livrables / Jalons
```
**P1** 01 – 5 mai 2026 (^) Cadrage rapide, revue
bibliographique
ciblée, setup
environnement
(PyTorch, CUDA,
MLflow), préparation
et accès datasets
**J1 — CDC validé +
environnement prêt
P2** 0 6 mai – 15 mai (^2026) Modèle MLP : EDA
Breast Cancer MSK
2018, prétraitement,
baseline MLP,
premiers tests et
tuning
**J2 — MLP fonctionnel +**
résultats préliminaires
**P3** 16 mai – 23 mai (^2026) Modèles avancés :
CNN (RSNA,
transfer learning
ResNet-50 si
possible) +
RNN/LSTM
J3 **—** Modèles CNN/RNN
opérationnels + analyse
comparative
**P4** 22 mai– 23 mai (^2026) Consolidation,
rédaction finale,
figures, préparation
soutenance
J4 — Récurrents opérationnels +
rapport intermédiaire III (
septembre)

### 6.2. Politique de revue et points d’étape

Un suivi très régulier est maintenu (échanges courts et fréquents avec l’encadrement,
idéalement tous les 2–3 jours). Chaque jalon (J1 à J3) sert de point de validation rapide afin
de garantir la progression dans un délai contraint.

En raison de la durée réduite du projet, aucune phase tampon n’est prévue : tout retard sur un
jalon entraîne une re-priorisation immédiate des tâches, en privilégiant la robustesse des
résultats plutôt que l’exhaustivité des expérimentations.

## 7. Détail complet de chaque partie.........................................................................................................

### 7.1. Partie I — MLP (Données tabulaires)

**_7.1.1. Objectif_**
L’objectif est de concevoir un Perceptron Multicouche (Rumelhart, Hinton & Williams, 1986 [3])
capable de prédire le statut métastatique (binaire) à partir des variables cliniques tabulaires du dataset
Breast Cancer MSK 2018. Cette première étape constitue le socle conceptuel du projet.


**_7.1.2. Étude théorique_**
Formalisation mathématique du neurone artificiel, organisation en couches entièrement connectées,
fonctions d’activation (sigmoïde, tanh, ReLU et variantes Leaky/PReLU), algorithme de
rétropropagation du gradient (Rumelhart et al., 1986 [3]), fonctions de coût (entropie croisée binaire),
mécanismes d’optimisation (SGD, Adam (Kingma & Ba, 2015 [10]), RMSProp), régularisation L1/L2,
dropout (Srivastava et al., 2014 [9]) et batch normalization (Ioffe & Szegedy, 2015 [8]).

**_7.1.3. Implémentation PyTorch_**
Modèle hérité de nn.Module avec nombre paramétrable de couches denses, activations
configurables, couches de régularisation. Pipeline : prétraitement (imputation, normalisation,
encodage), séparation train/validation/test stratifiée, instanciation optimiseur et fonction de perte.

**_7.1.4. Étude expérimentale_**
Variation contrôlée de la profondeur, de la largeur, des activations, du taux d’apprentissage, de la
taille du batch, du nombre d’époques et du taux de dropout. Chaque configuration est tracée via
MLflow pour reproductibilité.

**_7.1.5. Métriques d’évaluation_**
Exactitude, précision, rappel, F1-score, matrice de confusion, AUC-ROC, AUC-PR (particulièrement
adaptée en classes déséquilibrées).

**_7.1.6. Analyse critique_**
Sensibilité au déséquilibre, difficulté d’interprétation des poids appris, risque de surapprentissage,
limites face à des dépendances non linéaires complexes. Une analyse d’importance des variables sera
menée via permutation importance et SHAP (Lundberg & Lee, 2017 [12]).

**_7.1.7. Question de synthèse_**

```
En quoi l’architecture d’un Perceptron Multicouche est-elle adaptée à la nature tabulaire
du dataset Breast Cancer MSK 2018, et quelles sont ses limites face à des données cliniques
caractérisées par une forte hétérogénéité statistique et un déséquilibre fréquent des
classes?
```
### 7.2. Partie II — CNN (Images médicales)

**_7.2.1. Objectif_**
Concevoir un Réseau de Neurones Convolutif capable de classifier les radiographies thoraciques RSNA
selon la présence ou l’absence de pneumonie.

**_7.2.2. Étude théorique_**
Opérateurs fondamentaux : convolution discrète 2D, filtre, stride, padding, partage de poids. Pooling
(max, moyen) comme mécanisme de réduction dimensionnelle et de robustesse spatiale.
Architectures de référence : LeNet-5 (LeCun et al., 1998 [2]), AlexNet (Krizhevsky et al., 2012 [4]), VGG


(Simonyan & Zisserman, 2015 [5]), ResNet et connexions résiduelles (He et al., 2016 [7]). Transfert
d’apprentissage et augmentation de données.

**_7.2.3. Implémentation PyTorch_**
CNN composé de blocs convolutifs successifs suivis de couches denses. Pipeline : redimensionnement
(384×384 ou 512×512), normalisation, augmentation (rotation, recadrage, retournement, jitter),
recours à des modèles pré-entraînés via torchvision (ResNet-50 fine-tuné).

**_7.2.4. Étude expérimentale_**
Impact du nombre de filtres, de la profondeur, de la taille des noyaux, des stratégies de régularisation,
et du transfert d’apprentissage. Métriques complétées par courbes d’apprentissage et cartes
d’activation.

**_7.2.5. Comparaison CNN vs MLP_**
Confrontation directe entre un MLP appliqué aux pixels aplatis et un CNN exploitant la structure
spatiale, mettant en évidence le gain apporté par l’inductive bias spatial.

**_7.2.6. Analyse critique_**
Sensibilité aux artefacts radiologiques, biais d’acquisition (constructeur, exposition, position du
patient), coût computationnel et difficulté d’interprétation clinique malgré Grad-CAM (Selvaraju et al.,
2017 [11]).

**_7.2.7. Question de synthèse_**

```
Pourquoi un CNN constitue-t-il une architecture intrinsèquement plus pertinente qu’un
MLP pour l’analyse des radiographies pulmonaires du dataset RSNA, et dans quelles
conditions ses performances peuvent-elles être compromises par les biais d’acquisition
propres à l’imagerie médicale?
```
### 7.3. Partie III — RNN / LSTM / GRU (Séquences)

**_7.3.1. Objectif_**
Concevoir et comparer plusieurs architectures récurrentes dédiées à la classification multi-labels des
signaux ECG du dataset PTB-XL.

**_7.3.2. Étude théorique_**
Formalisme mathématique du RNN (propagation d’un état caché à travers le temps), Backpropagation
Through Time (BPTT), phénomènes d’évanouissement et d’explosion du gradient (Bengio et al., 1994
[21]). Architectures LSTM (Hochreiter & Schmidhuber, 1997 [6]), GRU (Cho et al., 2014 [13]) et leurs
portes (oubli, entrée, sortie, mise à jour, réinitialisation). Les architectures Transformer (Vaswani et
al., 2017 [14]) sont mentionnées à titre de perspective uniquement et ne font pas partie du périmètre
expérimental du présent projet (cf. section 11).


**_7.3.3. Implémentation PyTorch_**
Modules nn.RNN, nn.LSTM, nn.GRU. Pipeline : rééchantillonnage (100 Hz), segmentation,
normalisation par dérivation, formulation bidirectionnelle. Couche dense terminale + sigmoïde pour
la classification multi-labels.

**_7.3.4. Étude expérimentale_**
Comparaison RNN simple, LSTM, GRU en faisant varier nombre de couches, dimension de l’état caché,
longueur de la séquence d’entrée et régularisation. Métriques adaptées au multi-labels (F1 macro,
AUC moyenne, exactitude par classe).

**_7.3.5. Analyse critique_**
Stabilité d’apprentissage, sensibilité au bruit physiologique, impact de la longueur des séquences sur
la mémoire effective, compromis performance / coût computationnel. La convergence du RNN simple
sera analysée à la lumière des résultats classiques de Bengio et al. (1994) [21] sur la difficulté
d’apprentissage des dépendances longues par descente de gradient : sur les segments PTB-XL de 1
000 pas (10 s à 100 Hz), une dégradation expérimentale par rapport aux variantes LSTM/GRU est
attendue, et son ampleur effective sera mesurée et discutée. Référence comparative aux benchmarks
Strodthoff et al. (2020) [20].

**_7.3.6. Question de synthèse_**

```
Dans quelle mesure les architectures LSTM et GRU surpassent-elles le RNN classique pour
la classification multi-labels des signaux ECG du dataset PTB-XL, et quels facteurs liés à la
nature physiologique du signal expliquent cette supériorité?
```
## 8. Exigences fonctionnelles et non-fonctionnelles

### 8.1. Exigences fonctionnelles

- Partie I (MLP) : classifier les patientes du dataset Breast Cancer MSK 2018 selon le statut
    métastatique (binaire). Le modèle doit produire une probabilité de prédiction pour chaque
    classe.
- Partie II (CNN) : détecter la présence ou l’absence de pneumonie sur les radiographies
    thoraciques RSNA. Le modèle doit fournir un score de confiance entre 0 et 1.
- Partie III (RNN/LSTM/GRU) : classifier les signaux ECG PTB-XL selon les cinq supérclasses
    diagnostiques (NORM, MI, STTC, CD, HYP). Le modèle doit supporter la classification multi-
    labels.
- Tout modèle doit être entraînable, validable et testable avec une séparation
    train/validation/test reproductible.
- Les implémentations doivent produire des visualisations (courbes d’apprentissage, matrices
    de confusion, courbes ROC) pour chaque configuration expérimentale.


### 8.2. Exigences non-fonctionnelles

- Reproductibilité : seeds fixes, journalisation MLflow, documentation complète des
    hyperparamètres.
- Performance computationnelle : entraînement d’une expérience MLP ≤ 5 minutes sur GPU ;
    CNN ResNet-50 ≤ 2 heures sur RTX 3070 (avec AMP) ; RNN/LSTM/GRU ≤ 1 heure par
    expérience.
- Qualité du code : conventions PEP-8, docstrings, modularisation (datasets/, models/, train.py,
    evaluate.py), notebooks exécutables séquentiellement.
- Traçabilité : chaque expérience enregistrée avec timestamp, hash Git, hyperparamètres et
    métriques.

## 9. Critères de réussite

Les seuils retenus sont définis comme des objectifs raisonnables au regard des résultats publiés sur
les mêmes datasets ou des datasets cliniquement comparables. Ils sont positionnés dans la partie
médiane des performances rapportées dans la littérature, afin de tenir compte des ressources
matérielles disponibles (cf. section 10.5) et de la nature pédagogique du projet.

```
Partie Critères ciblés Référence littérature Justification du seuil
I — MLP Breast
Cancer MSK
2018
```
```
Accuracy ≥ 0.80 AUC-
ROC ≥ 0.85 F1-score ≥
0.
```
```
Rajkomar et al. 2018
[27] : deep learning
sur EHR atteint AUC
0.83–0.93 selon les
tâches (mortalité,
réadmission,
diagnostics). Beam &
Kohane 2018 [22] et
Esteva et al. 2019
[16] confirment la
fourchette 0.80–0.
pour MLP/RF sur
données cliniques
structurées.
```
```
Seuil AUC 0.85 = médiane
de la fourchette publiée ;
réaliste compte tenu du
déséquilibre attendu
(≈30–40 %
métastatiques).
```
#### II — CNN RSNA

```
Pneumonia
```
```
Sensibilité ≥ 0.
Spécificité ≥ 0.80 AUC-
ROC ≥ 0.
```
```
CheXNet (Rajpurkar
et al. 2017 [15]) :
AUC 0.768 sur
ChestX-ray14 —
résultat donné à titre
indicatif, dataset
distinct du RSNA.
Top-1 Kaggle RSNA ≈
0.25 mAP. ResNet- 50
fine-tuné sur RSNA :
AUC 0.87–0.93.
```
```
AUC 0.85 conservateur vs.
top Kaggle ; sensibilité
0.85 alignée avec les
recommandations
cliniques minimales pour
un outil d’aide au tri
(évite les faux négatifs
critiques).
```

```
Partie Critères ciblés Référence littérature Justification du seuil
III —
RNN/LSTM/GRU
PTB-XL
```
```
Macro F1 ≥ 0.72 AUC
moyen ≥ 0.
Exactitude ≥ 0.
```
```
Strodthoff et al. 2020
[20] : RNN/LSTM
atteignent macro
AUC ≈ 0.85–0.89 sur
les supérclasses PTB-
XL ; CNN profonds
(xresnet1d)
atteignent 0.93.
```
```
Macro F1 0.72 — cible
compatible avec une
LSTM bidirectionnelle
d’échelle moyenne.
L’écart attendu vis-à-vis
des CNN 1D (xresnet1d,
≈0.93) est cohérent : la
convolution 1D capture
mieux les motifs
morphologiques courts
(ondes P, complexe QRS,
T) que la propagation
séquentielle pure d’un
RNN.
```
Critères qualitatifs complémentaires : (a) rapport scientifique structuré incluant théorie,
implémentation, résultats et analyse critique pour chaque partie ; (b) code PyTorch documenté,
modulaire et reproductible ; (c) analyse critique substantielle (forces, limites, implications cliniques)
pour chaque partie ; (d) discussion transversale articulant les trois architectures.

## 10. Risques et mitigation

### 10.1. Déséquilibre des classes

Risque : les datasets médicaux contiennent souvent des classes déséquilibrées. Sur Breast Cancer MSK
2018, la classe métastatique représente ≈30–40 % des cas ; sur RSNA, ≈ 30 % d’opacité ; sur PTB-XL,
NORM domine.

Mitigation : pertes pondérées (weighted CE, focal loss), suralimentation contrôlée (SMOTE pour la
Partie I), stratified k-fold cross-validation, évaluation par F1 macro, AUC-PR et matrice de confusion
plutôt que par exactitude seule.

### 10.2. Surapprentissage (Overfitting)

Risque : le modèle mémorise les données d’entraînement plutôt que d’apprendre des représentations
généralisables.

Mitigation : dropout (Srivastava et al., 2014 [9]), batch normalization (Ioffe & Szegedy, 2015 [8]),
régularisation L1/L2, early stopping basé sur les performances de validation, data augmentation
(images, signaux).

### 10.3. Absence de généralisation


Risque : modèles entraînés sur des datasets publics potentiellement non représentatifs des
populations cibles.

Mitigation : k-fold stratifié, analyse d’erreurs par sous-groupe démographique, tests de robustesse
aux variations d’hyperparamètres, discussion explicite des biais de population.

### 10.4. Interprétabilité limitée

Risque : les modèles profonds restent des « boîtes noires ».

Mitigation : Grad-CAM (Selvaraju et al., 2017 [11]) pour les CNN, permutation importance et SHAP
(Lundberg & Lee, 2017 [12]) pour le MLP, visualisation des poids d’attention pour les variantes
récurrentes attentives.

### 10.5. Limitations en ressources computationnelles

Risque : la GPU disponible (NVIDIA RTX 3070, 8 Go VRAM) est en limite basse pour l’entraînement
d’un ResNet-50 à résolution 512×512 sur les 26 684 radiographies RSNA. Une analyse quantitative est
impérative avant écrou expérimental.

**_Calcul estimatif de la consommation VRAM — ResNet-50 @ 512×_**
Hypothèses : entrée RGB 3×512×512, ResNet-50 (≈25,6 M de paramètres), optimiseur Adam (deux
états m, v par paramètre), activations stockées pour la rétropropagation, FP32.

```
Composant mémoire Calcul FP32 FP16 (AMP)
Paramètres 25,6 M × 4 octets ≈ 100 Mo ≈ 50 Mo
Gradients idem paramètres ≈ 100 Mo ≈ 50 Mo
États Adam (m, v) 2 × paramètres (FP
même en AMP)
```
```
≈ 200 Mo ≈ 200 Mo
```
```
Activations forward
(batch=32)
```
```
≈ 200 Mo / image à
512×512 (estimation
par sommation des
feature maps stockées)
```
```
≈ 6,4 Go ≈ 3,2 Go
```
```
Buffers / workspace cuDNN ≈ 0,5–1 Go ≈ 0,8 Go ≈ 0,5 Go
TOTAL pic — ≈ 7,6 Go ≈ 4,0 Go
```
Lecture du tableau : avec un batch size de 32 en précision FP32, la VRAM cumulée atteint ≈7,6 Go, soit
la quasi-totalité des 8 Go disponibles, sans marge pour la fragmentation, le worker DataLoader ou les
pics transitoires PyTorch — le risque de crash CUDA OOM est élevé. La conclusion initiale du document
v1 (« ressources négligeables ») était donc inexacte.

**_Stratégie de mitigation retenue_**


- Mixed Precision Training (torch.cuda.amp) : divise la mémoire d’activations par 2 — ramené
    à ≈4,0 Go avec batch=32, soit une marge confortable.
- Batch size ciblé : 16 à 24 en FP32, 32 à 48 en AMP. Plage à ajuster en fonction du profil
    mémoire réel (torch.cuda.memory_summary()).
- Gradient accumulation : 2–4 micro-batches pour simuler un batch effectif équivalent sans
    dépasser la VRAM.
- Résolution alternative : 384×384 (≈ 56 % de la mémoire d’activations de 512×512) si tension
    persistante.
- Gradient checkpointing (torch.utils.checkpoint) : recalcul sélectif des activations en backward,
    divise la mémoire par ≈1,5 à 2 au prix de ≈20 % de surcoût en temps.
- Suivi continu : monitoring nvidia-smi et torch.cuda.max_memory_allocated() lors des
    premières époques pour valider l’enveloppe.

Conclusion : la RTX 3070 reste exploitable pour la Partie II, mais uniquement sous réserve d’appliquer
systématiquement AMP + batch size adapté. Une marge de sécurité a été intégrée dans le planning
(P3 durée 6 semaines, contre 4 strictement nécessaires) pour absorber d’éventuels problèmes
mémoire.

## 11. Perspectives d’évolution

Au-delà du scope actuel (trois pipelines unimodaux comparés), plusieurs directions de recherche
peuvent être explorées :

- Apprentissage véritablement multimodal : fusion (early, late ou cross-attention) des trois
    modalités dans une architecture unique pour exploiter les synergies entre types de données.
- Mécanismes d’attention et Transformers (Vaswani et al., 2017 [14]) : Vision Transformer pour
    l’imagerie, Transformer 1D pour l’ECG.
- Transfert d’apprentissage avancé : modèles pré-entraînés spécifiques au domaine médical
    (CheXNet, RadImageNet, BioBERT).
- Explainability avancée : SHAP, contrastive explanations, saliency maps, concept activation
    vectors (TCAV).
- Validation clinique : collaboration avec des institutions médicales pour valider sur cohortes
    externes (idéalement marocaines).
- Apprentissage fédéré : entraînement décentralisé sur des données sensibles.
- Déploiement : optimisation (quantization, ONNX) pour des déploiements edge ou hôpitaux
    équipés en infrastructure modérée.

## 12. Discussion transversale

La discussion transversale articulera les résultats des trois parties autour de la question centrale de
l’adéquation entre architecture et modalité de données. Le MLP, malgré sa simplicité, restera


pertinent pour les données tabulaires structurées, mais il atteindra rapidement ses limites face à la
complexité spatiale ou temporelle. Le CNN s’impose lorsque la donnée présente une organisation
spatiale exploitable, comme l’imagerie médicale, en raison de son partage de poids et de son
invariance locale (LeCun et al., 1998 [2]). Les architectures récurrentes demeurent privilégiées pour
les signaux séquentiels, bien que les modèles modernes fondés sur l’attention (Vaswani et al., 2017
[14]) constituent des alternatives compétitives sur les benchmarks ECG (Strodthoff et al., 2020 [20]).

Cette comparaison permettra de justifier les choix architecturaux non pas selon une logique
universelle, mais en fonction des propriétés statistiques et structurelles des données traitées. Elle
soulignera également la complémentarité potentielle des approches dans une perspective
multimodale (cf. section 11).

## 13. Impact sociétal

### 13.1. Bénéfices et risques généraux

Le déploiement de modèles de deep learning en santé suscite des bénéfices considérables :
amélioration de la précision diagnostique, démocratisation de l’expertise médicale dans des zones
sous-dotées, réduction des délais de prise en charge et soutien à la décision clinique. Ces avancées
contribuent à une médecine plus préventive, plus personnalisée et plus accessible (Topol, 2019 [17]).

Toutefois, ces bénéfices s’accompagnent de risques significatifs : biais des données d’entraînement
perpétuant des inégalités cliniques, erreurs de prédiction aux conséquences potentiellement graves,
interprétabilité limitée des modèles profonds, préoccupations éthiques sur la confidentialité des
données et la responsabilité juridique.

### 13.2. Pertinence pour le contexte marocain et africain

Le contexte marocain et africain confère à ce projet une dimension particulièrement saillante. Selon
les statistiques de l’Organisation mondiale de la santé (OMS), le Maroc présente une densité médicale
d’environ 7,3 médecins pour 10 000 habitants, très en deçà du seuil OCDE (≈35) et insuffisante au
regard du minimum recommandé par l’OMS pour assurer une couverture sanitaire universelle. La
répartition territoriale est en outre marquée par une concentration urbaine prononcée : les régions
de l’Atlas, du Rif et certaines zones rurales du Sud souffrent d’un accès très limité aux radiologues,
cardiologues et oncologues, ces spécialités étant essentiellement disponibles dans les CHU de
Casablanca, Rabat, Marrakech et Fès.

Dans ce contexte, des outils d’aide au diagnostic fondés sur le deep learning peuvent jouer un rôle
d’amplification de l’expertise médicale là où elle est la plus rare. Trois cas d’usage couverts par le
présent projet illustrent cette pertinence :

- Détection automatisée de pneumonies sur radiographies (Partie II) : utile dans les centres de
    santé communautaires équipés d’appareils de radiographie mais sans radiologue résident, en
    complément de la téléradiologie.


- Analyse d’ECG (Partie III) : pertinente dans les zones rurales où l’accès à un cardiologue est
    différé, et pour le triage en médecine d’urgence.
- Stratification du risque oncologique à partir de données cliniques structurées (Partie I) : appui
    aux décisions de pré-orientation dans les centres régionaux de cancer (RCROM, INO).

À l’échelle continentale, des initiatives comme le programme Africa CDC « AI for Public Health », le
centre AI Movement de l’Université Mohammed VI Polytechnique (UM6P), ou les travaux du
laboratoire IHEAL/MAScIR au Maroc témoignent d’un investissement croissant dans l’IA appliquée à
la santé. Cependant, ces déploiements doivent tenir compte de spécificités locales :

- Biais épidémiologique : les datasets occidentaux (RSNA, MSK, PTB-XL) reflètent des
    populations caucasiennes à majorité nord-américaine ou européenne ; leur transposition au
    contexte marocain nécessite une validation locale rigoureuse pour éviter des dérives
    diagnostiques.
- Profil de pathologies : le poids de la tuberculose, plus fréquente en Afrique du Nord et
    subsaharienne, peut interférer avec la détection de pneumonies bactériennes par un modèle
    entraîné hors de ce contexte.
- Infrastructure : les zones rurales présentent un accès réseau limité ; un déploiement edge
    (modèles compressés, inférence locale) est préférable à une architecture cloud.
- Cadre réglementaire : la loi 09-08 sur la protection des données personnelles et le cadre du
    ministère de la Santé imposent une vigilance accrue sur la confidentialité et la souveraineté
    des données de santé.

Ce projet, bien que de nature pédagogique, s’inscrit ainsi dans une réflexion plus large : celle de la
formation d’une compétence locale en deep learning médical, condition préalable au développement
de solutions adaptées aux réalités sanitaires marocaines et africaines.

## 14. Livrables attendus

Les livrables attendus à l’issue du projet comprennent :

- Un rapport scientifique structuré présentant la démarche théorique, expérimentale et
    critique pour chacune des trois parties (≈ 80 à 100 pages).
- Le code source PyTorch dans son intégralité (dépôt Git public ou privé, structuré en datasets/,
    models/, train/, evaluate/).
- Un notebook Jupyter exécutable par partie, permettant la reproduction des expériences.
- Les artefacts expérimentaux : courbes d’apprentissage, matrices de confusion, courbes
    ROC/PR, cartes d’activation Grad-CAM, analyses SHAP.
- Un document de synthèse répondant aux trois questions associées aux parties I, II et III.
- Une présentation de soutenance (≈ 20 à 25 slides).

## 15. Résultats attendus


À l’issue du projet, les compétences acquises porteront sur la maîtrise théorique et pratique des trois
grandes familles d’architectures de deep learning, ainsi que sur la capacité à les appliquer
rigoureusement à des problématiques médicales réelles. L’étudiant développera également une
aptitude à interpréter les résultats expérimentaux, à formuler une analyse critique fondée sur des
éléments quantitatifs et à dégager des conclusions transversales. L’interprétation des modèles, bien
que partielle, sera abordée comme un enjeu scientifique et éthique essentiel, et fournira la base d’une
réflexion mature sur le rôle du deep learning dans la médecine contemporaine, en particulier dans le
contexte marocain et africain.

## 16. Références bibliographiques

### 16.1. Articles fondateurs et méthodologiques

[1] LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436–444.

[2] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document
recognition. Proceedings of the IEEE, 86(11), 2278–2324.

[3] Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-
propagating errors. Nature, 323(6088), 533–536.

[4] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional
neural networks. Advances in Neural Information Processing Systems (NIPS), 25, 1097–1105.

[5] Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image
recognition. International Conference on Learning Representations (ICLR). arXiv:1409.1556.

[6] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8),
1735 – 1780.

[7] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. IEEE CVPR,
770 – 778.

[8] Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing
internal covariate shift. ICML, 448–456.

[9] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A
simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1),
1929 – 1958.

[10] Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. ICLR. arXiv:1412.6980.

[11] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM:
Visual explanations from deep networks via gradient-based localization. IEEE ICCV, 618–626.

[12] Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NIPS,
30, 4765–4774.


[13] Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y.
(2014). Learning phrase representations using RNN encoder–decoder for statistical machine
translation. EMNLP, 1724–1734.

[14] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin,
I. (2017). Attention is all you need. NIPS, 30, 59 98 – 6008.

[21] Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning long-term dependencies with gradient
descent is difficult. IEEE Transactions on Neural Networks, 5(2), 157–166.

### 16.2. Deep learning en santé

[15] Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., ... & Ng, A. Y. (2017). CheXNet:
Radiologist-level pneumonia detection on chest X-rays with deep learning. arXiv:1711.05225.

[16] Esteva, A., Robicquet, A., Ramsundar, B., Kuleshov, V., DePristo, M., Chou, K., ... & Dean, J. (2019).
A guide to deep learning in healthcare. Nature Medicine, 25(1), 24–29.

[17] Topol, E. J. (2019). High-performance medicine: the convergence of human and artificial
intelligence. Nature Medicine, 25(1), 44–56.

[22] Beam, A. L., & Kohane, I. S. (2018). Big data and machine learning in health care. JAMA, 319(13),
1317 – 1318.

[27] Rajkomar, A., Oren, E., Chen, K., Dai, A. M., Hajaj, N., Hardt, M., ... & Dean, J. (2018). Scalable and
accurate deep learning with electronic health records. NPJ Digital Medicine, 1(1), 18.

### 16.3. Datasets médicaux et benchmarks

[18] Razavi, P., Chang, M. T., Xu, G., Bandlamudi, C., Ross, D. S., Vasan, N., ... & Baselga, J. (2018). The
genomic landscape of endocrine-resistant advanced breast cancers. Cancer Cell, 34(3), 427–438.
(Cohorte MSK 2018)

[19] Wagner, P., Strodthoff, N., Bousseljot, R. D., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T.
(2020). PTB-XL, a large publicly available electrocardiography dataset. Scientific Data, 7(1), 154.

[20] Strodthoff, N., Wagner, P., Schaeffter, T., & Samek, W. (2020). Deep learning for ECG analysis:
Benchmarks and insights from PTB-XL. IEEE Journal of Biomedical and Health Informatics, 25(5), 1519–
1528.

[23] RSNA. (2018). RSNA Pneumonia Detection Challenge. Radiological Society of North America.
https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

## 16.4. Frameworks et outils

[24] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch:
An imperative style, high-performance deep learning library. NeurIPS, 32, 8024–8035.


[25] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E.
(2011). Scikit-learn: Machine learning in Python. JMLR, 12, 2825–2830.

[26] Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

## 16.5. Webographie — datasets et plateformes

- Breast Cancer MSK 2018 — cBioPortal :
    https://www.cbioportal.org/study/clinicalData?id=breast_msk_
- RSNA Pneumonia Detection Challenge : https://www.rsna.org/ai/rsna-pneumonia-detection-
    challenge
- PTB-XL ECG Dataset — PhysioNet : https://physionet.org/content/ptb-xl/1.0.3/
- PyTorch : https://pytorch.org
- Scikit-learn : https://scikit-learn.org
- MLflow (suivi d’expériences) : https://mlflow.org
# Projet de fin de module DL

```
## 1. Intitulé

**Conception, implémentation, comparaison et analyse critique de modèles de
deep learning pour données tabulaires, images et séquences**

## 2. Nature du travail

Ce projet de fin de module est conçu comme un **examen final intégrateur**. Il doit per-
mettre d’évaluer simultanément :

```
— la compréhension théorique des concepts fondamentaux du module ;
— la maîtrise de l’implémentation sous PyTorch ;
— la capacité d’analyse expérimentale et comparative ;
— la qualité de l’interprétation scientifique des résultats ;
— la capacité de synthèse sur des jeux de données réels.
```
**Le projet est strictement individuel.** Chaque étudiant(e) doit produire seul(e) l’en-
semble du travail demandé : préparation, implémentation, expérimentation, rédaction et
discussion critique.


## 3. Objectif général

L’objectif du projet est de montrer que l’étudiant(e) sait adapter le choix d’une architec-
ture de deep learning à la nature des données étudiées. Le travail doit couvrir trois cadres
complémentaires :

```
— les données tabulaires avec un MLP ;
— les données images avec un CNN ;
— les données séquentielles avec des modèles RNN / LSTM / GRU / Seq2Seq.
```
L’étudiant(e) doit justifier ses choix d’architecture, expliquer les fondements théoriques,
conduire des expériences comparatives rigoureuses et proposer une interprétation scientifique
des résultats obtenus.

## 4. Organisation générale du projet

Le projet est organisé en **trois parties indépendantes mais cohérentes** , chacune cor-
respondant à une grande famille de modèles étudiés dans le module.

Chaque partie doit obligatoirement contenir :

1. une étude théorique structurée ;
2. une implémentation sous PyTorch ;
3. une étude expérimentale ;
4. une analyse critique ;
5. une **question de synthèse portant sur un dataset réel**.

## 5. Partie I – MLP et ingénierie PyTorch

**Thème**

**Classification supervisée sur données tabulaires réelles avec perceptron multi-
couche (MLP).**

**Objectifs spécifiques**

Dans cette partie, l’étudiant(e) doit montrer qu’il/elle maîtrise la construction d’un modèle
avec nn.Module, la différence entre nn.Sequential et une classe personnalisée, la gestion
des paramètres, l’initialisation, la sauvegarde/rechargement ainsi que l’utilisation correcte
du CPU/GPU.


**Dataset**

Choisir un **dataset tabulaire réel** de classification, par exemple :

```
— Wine Quality,
— Breast Cancer Wisconsin,
— Adult Income,
— ou tout autre dataset équivalent validé par l’enseignant(e).
```
**Travail demandé**

1. Présenter les concepts fondamentaux : nn.Module, paramètres, gradient, state_dict,
    _device_ , propagation avant et rétropropagation.
2. Préparer correctement les données : nettoyage, encodage, normalisation, séparation
    apprentissage/validation/test.
3. Implémenter deux versions d’un MLP :
    — une version avec nn.Sequential,
    — une version avec classe personnalisée.
4. Inspecter et commenter les paramètres du modèle à l’aide de named_parameters()
    et state_dict().
5. Tester au moins trois stratégies d’initialisation : gaussienne, constante et Xavier.
6. Sauvegarder puis recharger le meilleur modèle.
7. Exécuter le modèle sur le _device_ disponible et vérifier la cohérence entre données et
    modèle.
8. Évaluer les performances à l’aide de métriques adaptées : accuracy, precision, recall,
    F1-score, matrice de confusion.

**Question de synthèse sur dataset réel**

**Question de synthèse – Partie I :**

_Dans quelle mesure un MLP bien paramétré constitue-t-il une solution pertinente pour la
classification tabulaire sur un dataset réel, et quelles sont ses principales limites au regard
de la structure statistique des données étudiées?_

La réponse devra articuler théorie, choix méthodologiques, résultats expérimentaux et ana-
lyse critique.


## 6. Partie II – CNN et vision par ordinateur

**Thème**

**Classification d’images avec réseaux de neurones convolutionnels (CNN) et ana-
lyse des opérations convolutionnelles.**

**Objectifs spécifiques**

Cette partie doit montrer que l’étudiant(e) comprend la structure spatiale des images, la
corrélation croisée 2D, le rôle du _padding_ , du _stride_ , du _pooling_ , des canaux multiples, de la
convolution 1 × 1 et des architectures de type LeNet.

**Dataset**

Choisir un **dataset réel d’images** , par exemple :

```
— MNIST,
— Fashion-MNIST,
— CIFAR-10,
— ou tout autre dataset équivalent validé par l’enseignant(e).
```
**Travail demandé**

1. Expliquer pourquoi un MLP est peu adapté aux images et rappeler les idées fonda-
    trices des CNN : localité, partage des poids et hiérarchie des représentations.
2. Effectuer des calculs manuels de corrélation croisée, de taille de sortie en convolution
    et de taille de sortie après pooling.
3. Programmer une version simple de :
    — la corrélation croisée 2D,
    — le max-pooling,
    — l’average-pooling.
4. Comparer ces implémentations avec les couches PyTorch correspondantes.
5. Implémenter un CNN inspiré de **LeNet** ou une variante améliorée.
6. Étudier expérimentalement l’influence de plusieurs choix architecturaux :
    — padding,
    — stride,
    — type de pooling,


```
— nombre de filtres,
— présence ou absence de convolution 1 × 1.
```
7. Visualiser et interpréter quelques cartes de caractéristiques.
8. Comparer un MLP simple et un CNN sur le même dataset d’images.

**Question de synthèse sur dataset réel**

**Question de synthèse – Partie II :**

_Pourquoi un CNN est-il plus pertinent qu’un MLP pour une tâche de classification d’images
sur un dataset réel, et comment les choix de padding, stride, pooling et profondeur influencent-
ils réellement les performances du modèle?_

La réponse devra relier calculs dimensionnels, théorie des CNN, résultats expérimentaux et
interprétation des représentations internes.

## 7. Partie III – RNN, LSTM, GRU et Seq2Seq

**Thème**

**Modélisation de séquences et génération/traduction sur données textuelles réelles.**

**Objectifs spécifiques**

L’étudiant(e) doit démontrer qu’il/elle maîtrise les modèles de langage, les RNN, la rétropro-
pagation à travers le temps, le _gradient clipping_ , les cellules LSTM et GRU, les architectures
encodeur–décodeur, le _teacher forcing_ , les métriques de type BLEU et les stratégies de dé-
codage comme le beam search.

**Dataset**

Choisir un **dataset textuel réel** ou un corpus séquentiel réel, par exemple :

```
— un corpus de prédiction du prochain token,
— un corpus parallèle simple pour traduction automatique,
— IMDb,
— Tatoeba / fra-eng simplifié,
— ou tout autre corpus validé par l’enseignant(e).
```

**Travail demandé**

1. Expliquer l’objectif probabiliste d’un modèle de langage et la factorisation d’une
    séquence par la règle de chaîne.
2. Présenter la notion de perplexité et son interprétation.
3. Implémenter successivement un RNN simple, un LSTM et un GRU.
4. Comparer ces modèles en termes de stabilité, performance, capacité à mémoriser le
    contexte et coût de calcul.
5. Expliquer le principe du BPTT et illustrer expérimentalement l’effet du _gradient_
    _clipping_.
6. Préparer les données : tokenisation, vocabulaire, tokens spéciaux, padding, masquage
    et mini-lots.
7. Construire un mini système **Seq2Seq** avec encodeur et décodeur récurrents.
8. Tester au moins deux stratégies de décodage :
    — décodage glouton,
    — beam search.
9. Évaluer les performances à l’aide d’une métrique adaptée, par exemple la perplexité
    ou BLEU selon la tâche choisie.

**Question de synthèse sur dataset réel**

**Question de synthèse – Partie III :**

_Dans quelle mesure les architectures récurrentes permettent-elles de modéliser efficacement
une séquence réelle, et comment justifier le passage d’un RNN simple vers un LSTM/GRU
puis vers un schéma encodeur–décodeur pour une tâche de génération ou de traduction?_

La réponse doit articuler modélisation probabiliste, mémoire, entraînement, qualité du dé-
codage et limites observées expérimentalement.

## 8. Question transversale finale

Rédiger une discussion scientifique globale répondant à la problématique suivante :

```
Comment le deep learning adapte-t-il ses architectures à la structure des données
```
_- tabulaire, image et séquentielle – et pourquoi un même paradigme d’apprentis-
sage supervisé doit-il être décliné différemment selon la géométrie, la dépendance
locale, la temporalité et la représentation des données?_


Cette synthèse finale doit faire apparaître les liens conceptuels et méthodologiques entre
MLP, CNN et modèles séquentiels.

## 9. Livrables attendus

Chaque étudiant(e) doit remettre les éléments suivants :

1. un **rapport scientifique structuré** comprenant : introduction, objectifs, méthodo-
    logie, implémentation, résultats, interprétation, limites et conclusion ;
2. le **code source complet** et bien commenté ;
3. un **notebook** ou un script principal exécutable ;
4. une **annexe expérimentale** comportant courbes, tableaux comparatifs, métriques
    et visualisations utiles.

## 10. Barème indicatif sur 100 points

**Partie I – MLP / PyTorch : 30 points**

```
— compréhension théorique : 6 points ;
— préparation des données : 5 points ;
— qualité de l’implémentation : 7 points ;
— initialisation / sauvegarde / GPU : 5 points ;
— analyse critique : 4 points ;
— question de synthèse sur dataset réel : 3 points.
```
**Partie II – CNN : 35 points**

```
— théorie et calculs : 8 points ;
— implémentation manuelle et PyTorch : 7 points ;
— modèle CNN : 8 points ;
— expérimentation comparative : 6 points ;
— interprétation des représentations : 3 points ;
— question de synthèse sur dataset réel : 3 points.
```
**Partie III – RNN / Seq2Seq : 35 points**

```
— théorie des séquences : 8 points ;
```

```
— implémentation RNN / LSTM / GRU : 7 points ;
— préparation des données : 5 points ;
— système Seq2Seq : 7 points ;
— décodage et évaluation : 5 points ;
— question de synthèse sur dataset réel : 3 points.
```
## 11. Consignes de qualité

Le projet sera évalué avec un niveau d’exigence élevé. Une attention particulière sera accor-
dée à :

```
— la rigueur scientifique ;
— la clarté des justifications ;
— la qualité du code ;
— la pertinence des expériences ;
— la qualité des interprétations ;
— la capacité à relier théorie et pratique.
```
Un simple enchaînement de cellules de notebook ou de résultats bruts sans analyse ne sera
pas considéré comme suffisant.

## 12. Consigne finale

Le document remis doit être rédigé dans un style académique, clair, structuré et rigoureux.
Les résultats expérimentaux doivent être commentés de manière précise. Toute figure, tout
tableau ou toute métrique doit être interprété(e) en lien avec les objectifs de la partie
correspondante.

```
Bon courage à toutes et à tous.
```
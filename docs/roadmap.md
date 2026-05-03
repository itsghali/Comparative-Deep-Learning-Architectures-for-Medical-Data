# Roadmap — Projet Fin de Module Deep Learning

# (EMSI 2025–2026)

Période : 1 → 23 mai 2026 · Travail individuel · Barème : 30 / 35 / 35

## Stratégie globale

Trois parties indépendantes mais cohérentes, alignées sur le barème :

```
Phase Contenu Jours Points
P0 Setup environnement 30 avril —
P1 MLP / PyTorch (tabulaire) 1–5 mai 30
P2 CNN (image) 6–13 mai 35
P3 RNN / LSTM / GRU / Seq2Seq (séquentiel) 14–21 mai 35
P4 Rédaction + synthèse transversale 22–23 mai (transversal)
```
Datasets retenus (tous dans la liste suggérée du CDC, donc validés implicitement) :
MLP → Breast Cancer Wisconsin (UCI, 569 × 30, binaire)
CNN → Fashion-MNIST (10 classes, 28×28, niveaux de gris) ; option CIFAR-10 si
temps disponible pour comparaison MLP vs CNN
RNN / Seq2Seq → Tatoeba fra–eng simplifié
Côté monolingue (anglais) → modélisation de langue (RNN / LSTM / GRU +
perplexité)
Paires parallèles → Seq2Seq + BLEU
Principes non négociables :
1. To u t e s t i m p l é m e n t é sous PyTorch.
2. Chaque partie répond à sa question de synthèse + l’ensemble alimente la


```
question transversale finale.
3. Reproductibilité : seed fixée, requirements.txt, scripts exécutables.
4. Chaque journée se termine par un commit Git.
```
## P0 — Setup (30 avril)

### Structure projet

```
dl_emsi_project/
├── data/
│ ├── raw/ # données brutes téléchargées
│ └── processed/ # tenseurs prêts à l'entraînement
├── notebooks/ # 1 notebook par sous-partie
|── docs/
├── src/
│ ├── mlp/
│ ├── cnn/
│ │ └── manual/ # impl. from-scratch (corr. croisée, pooling)
│ ├── seq/
│ │ ├── rnn/
│ │ └── seq2seq/
│ └── utils/ # seed, dataloaders, metrics, viz
├── models/ # checkpoints .pt
├── outputs/
│ ├── figures/
│ ├── tables/
│ └── logs/
├── reports/ # rapport LaTeX/Word
├── requirements.txt
└── README.md
```
### Tâches

```
python -m venv venv puis activation
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn
jupyter tqdm sacrebleu nltk
Vérification GPU : torch.cuda.is_available(), torch.cuda.get_device_name(0)
git init + .gitignore (data/raw, models, venv, pycache)
Module src/utils/seed.py qui fixe random, numpy, torch, torch.cuda et active
```

```
torch.backends.cudnn.deterministic = True
```
## P1 — MLP / PyTorch · Breast Cancer Wisconsin (1–5 mai)

```
Cible : 30/30. Question de synthèse Partie I à la fin.
```
### Jour 1 (1 mai) — Données + théorie

```
Té l é c h a r g e m e n t B r e a s t C a n c e r W i s c o n s i n v i a
sklearn.datasets.load_breast_cancer()
EDA dans notebooks/01_mlp_eda.ipynb : missing values, distribution des classes,
corrélations, types
Théorie à rédiger en parallèle (markdown dans le notebook) :
nn.Module, paramètres apprenables, state_dict, device
propagation avant / rétropropagation (chaîne de dérivation)
rôle du gradient et de l’optimiseur
Pipeline preprocessing : imputation médiane, encodage si besoin, StandardScaler,
split 70/15/15 stratifié
Sauvegarde tenseurs : data/processed/bcw_{train,val,test}.pt
```
### Jour 2 (2 mai) — Deux versions du MLP

```
V1 dans src/mlp/model_seq.py : nn.Sequential(Linear → ReLU → Dropout → Linear
→ ReLU → Linear → Sigmoid)
V2 dans src/mlp/model_class.py : sous-classe nn.Module avec __init__ +
forward, exposant un hook pour récupérer les activations intermédiaires
Comparer explicitement dans le rapport : flexibilité, lisibilité, accès aux activations
Inspection : pour chaque modèle, afficher named_parameters() (shape,
requires_grad) et state_dict().keys() ; commenter
```
### Jour 3 (3 mai) — Stratégies d’initialisation

```
src/mlp/init_strategies.py : 3 fonctions
Gaussienne : nn.init.normal_(w, 0, 0.01)
```

```
Constante : nn.init.constant_(w, 0.1) (pour montrer la pathologie de
symétrie)
Xavier : nn.init.xavier_uniform_(w)
Entraîner les 3 versions (mêmes hyperparams), mêmes 30 époques, même seed
Tracer côte-à-côte : courbe loss train/val sur les 3 inits
Ta b l e a u c o m p a r a t i f : a c c u r a c y f i n a l e , é p o q u e d e c o nve r g e n c e , o b s e r va t i o n
qualitative
Conclusion attendue : init constante stagne (symétrie), Xavier converge le plus
stablement
```
### Jour 4 (4 mai) — Entraînement, sauvegarde, GPU

```
Boucle d’entraînement propre : BCEWithLogitsLoss, Adam, scheduler optionnel, early
stopping sur val loss
Sauvegarde : torch.save(model.state_dict(), 'models/mlp_best.pt')
Rechargement dans un notebook séparé pour vérifier que les prédictions sont
identiques
Vérification CPU vs GPU :
model.to(device) + x.to(device)
assertion explicite assert next(model.parameters()).device == x.device
Logs : pour chaque expé, sauver {config, metrics_per_epoch, best_metric} en
JSON
```
### Jour 5 (5 mai) — Évaluation + question de synthèse

```
Métriques sur test : accuracy, precision, recall, F1, ROC-AUC, matrice de confusion
Figures dans outputs/figures/mlp/ : courbe ROC, matrice de confusion, courbes
loss
Réponse à la question de synthèse Partie I (1–2 pages) :
Dans quelle mesure un MLP bien paramétré constitue-t-il une solution pertinente
pour la classification tabulaire [...], et quelles sont ses principales limites au
regard de la structure statistique des données étudiées?
Articuler : performances obtenues + comparaison rapide vs régression logistique
```

```
baseline
Limites : pas d’invariance spatiale/temporelle, sensibilité à la dimension,
interprétabilité limitée, exigence de normalisation, comportement sur petit
dataset
```
### Mapping barème Partie I (30 pts)

```
Critère Pts Livré
Compréhension
théorique^6
```
```
Section théorie J1 + commentaires
named_parameters
Préparation des données 5 Pipeline J1 + tenseurs sauvegardés
Qualité d’implémentation 7 Deux versions (Sequential + classe)
Init / save / GPU 5 J3 + J
Analyse critique 4 Comparaison inits + limites
Question de synthèse 3 Section dédiée J
```
## P2 — CNN · Fashion-MNIST (6–13 mai)

```
Cible : 35/35. Partie la plus exigeante (calculs manuels + ablations + interprétation).
```
### Jour 6 (6 mai) — Théorie + calculs manuels

Dans notebooks/02_cnn_theory.ipynb :
Pourquoi un MLP est mauvais pour les images : explosion paramétrique, perte de
structure spatiale, pas d’invariance à la translation
Idées fondatrices CNN : localité, partage des poids, hiérarchie des représentations
Calculs manuels à faire à la main, photos / LaTeX :
Une corrélation croisée 2D explicite (matrice 4×4, noyau 3×3, sortie 2×2 calculée
case par case)
Formule de taille de sortie : H_out = floor((H_in + 2P − K) / S) + 1 appliquée
sur 3 cas (K=3 P=0 S=1, K=5 P=2 S=1, K=3 P=1 S=2)


```
Ta i l l e a p r è s p o o l i n g : 2×2 stride 2 sur 28× 28
Préparation Fashion-MNIST : torchvision.datasets.FashionMNIST, normalisation,
DataLoader
```
### Jour 7 (7 mai) — Implémentations manuelles

```
src/cnn/manual/ :
corr2d(X, K) : corrélation croisée 2D pure NumPy/PyTorch (boucles imbriquées,
pas de nn.Conv2d)
max_pool2d(X, size, stride) : max-pooling from scratch
avg_pool2d(X, size, stride) : average-pooling from scratch
Te s t s u n i t a i r e s : s u r p e t i t e m a t r i c e 5×5, comparer bit à bit la sortie vs F.conv2d /
F.max_pool2d / F.avg_pool2d
Notebook démo : appliquer les 3 fonctions sur une image Fashion-MNIST + visualiser
la sortie
```
### Jour 8 (8 mai) — LeNet baseline

```
src/cnn/lenet.py :
Conv(1→6, 5×5, pad 2) → ReLU → AvgPool 2× 2
Conv(6→16, 5×5) → ReLU → AvgPool 2× 2
Flatten → FC(120) → FC(84) → FC(10)
Entraînement Fashion-MNIST, ~10 époques, Adam, log par époque
Sauvegarde checkpoint comme baseline de référence
Comparaison avec un MLP simple sur le même dataset (FC 784 → 256 → 128 →
10) → tableau résultats : accuracy, nb paramètres, temps/époque
```
### Jour 9 (9 mai) — Ablation padding & stride

```
4 expés contrôlées (changer une seule variable à la fois) :
Référence : padding=2, stride=
padding=
stride=2 (dans la première conv)
```

```
kernel 3×3 au lieu de 5× 5
Ta b l e a u : a c c u r a c y t e s t , t a i l l e d e s o r t i e d e c h a q u e fe a t u r e m a p , n b p a r a m è t r e s
```
### Jour 10 (10 mai) — Ablation pooling & filtres

```
Variantes :
AvgPool vs MaxPool (sinon LeNet identique)
6/16 filtres vs 16/32 filtres vs 32/64 filtres
Avec et sans convolution 1× 1 insérée entre les deux blocs convolutifs
Ta b l e a u r é c a p i t u l a t i f + c o u r b e s d e l o s s s u p e r p o s é e s
Discussion : à quel moment max-pool aide vs avg-pool, gain réel d’une 1×1 sur ce
dataset
```
### Jour 11 (11 mai) — Visualisation des cartes de caractéristiques

```
Hook PyTorch sur les sorties des deux couches convolutives
Pour 3–5 images de classes différentes, afficher les feature maps des deux couches
en grille
Interprétation : couche 1 = bords / textures, couche 2 = motifs plus abstraits ;
comparer entre classes
```
### Jour 12 (12 mai) — Consolidation comparative

```
Ta b l e a u m a s t e r Pa r t i e I I : M L P v s Le N e t b a s e l i n e v s Le N e t + 1×1 vs meilleure variante
Colonnes : accuracy, F1 macro, nb paramètres, temps d’entraînement, taille modèle
(Mo)
Figures finales propres dans outputs/figures/cnn/
```
### Jour 13 (13 mai) — Question de synthèse Partie II

Pourquoi un CNN est-il plus pertinent qu’un MLP pour une tâche de classification
d’images [...], et comment les choix de padding, stride, pooling et profondeur
influencent-ils réellement les performances du modèle?
Réponse (2–3 pages) :
Calculs dimensionnels rappelés


```
Théorie : invariance par translation, partage des poids, champ réceptif croissant
Résultats expérimentaux des ablations comme preuves
Lien avec les feature maps observées
```
### Mapping barème Partie II (35 pts)

```
Critère Pts Livré
Théorie + calculs manuels 8 J
Implémentation manuelle + PyTorch 7 J7 + tests d’équivalence
Modèle CNN (LeNet) 8 J
Expé comparative 6 J9 + J10 + J
Interprétation représentations 3 J
Question de synthèse 3 J
```
## P3 — RNN / LSTM / GRU + Seq2Seq · Tatoeba fra–eng (14–

## mai)

```
Cible : 35/35. Pas de raccourci sur Seq2Seq, c’est ce qui plombe la majorité des
projets.
```
### Jour 14 (14 mai) — Théorie modélisation de langue

Dans notebooks/03_seq_theory.ipynb :
Objectif probabiliste : P(x_1, ..., x_T) = Π_t P(x_t | x_<t) (règle de chaîne)
Perplexité : PP = exp(loss_NLL_moyen) ; interprétation = nb effectif de choix par
token
Vocabulaire, tokens spéciaux (<bos>, <eos>, <pad>, <unk>), padding et masquage
Té l é c h a r g e m e n t Ta t o e b a f r a - e n g s i m p l i f i é , p r e m i e r n e t t oya g e ( l owe r c a s e ,
ponctuation)

### Jour 15 (15 mai) — Préparation des données


```
src/seq/data.py :
To ke n i s a t i o n s i m p l e ( m o t s o u s o u s - m o t s )
Construction vocab côté src (français) et tgt (anglais)
Filtrage paires longueur > seuil (ex. 20 tokens)
DataLoader avec collate fonction (padding dynamique, retour des longueurs)
Pour la modélisation de langue (RNN/LSTM/GRU) : utiliser uniquement le côté
anglais comme corpus monolingue → fenêtres glissantes de longueur fixe (ex. 35
tokens)
```
### Jour 16 (16 mai) — RNN simple (modèle de langue)

```
src/seq/rnn/vanilla_rnn.py :
nn.RNN + tête linéaire vers la taille du vocab
Entraînement sur corpus EN, BPTT tronqué (longueur fenêtre)
Mesurer la perplexité sur train et val
Illustrer expérimentalement le gradient clipping : 2 runs identiques, un avec
clip_grad_norm_(params, max_norm=1.0), un sans → tracer norme du gradient par
étape ; commenter exploding gradient sans clipping
```
### Jour 17 (17 mai) — LSTM (même corpus)

```
nn.LSTM à la place du RNN, mêmes hyperparams sinon
Comparaison perplexité vs RNN simple
Discussion : portes, mémoire à long terme, stabilité du gradient
```
### Jour 18 (18 mai) — GRU + comparatif RNN/LSTM/GRU

```
nn.GRU, mêmes hyperparams
Ta b l e a u c o m p a r a t i f f i n a l p a r t i e m o d é l i s a t i o n : p e r p l ex i t é t r a i n / va l , t e m p s / é p o q u e , n b
paramètres, courbe de loss superposée
Section dédiée BPTT dans le rapport : équations, troncature, lien avec
exploding/vanishing gradient + résultats J
```
### Jour 19 (19 mai) — Seq2Seq encodeur–décodeur


```
src/seq/seq2seq/ :
Encodeur : embedding + GRU (bidirectionnel option)
Décodeur : embedding + GRU + tête linéaire vocab tgt
État caché final encodeur → état initial décodeur
Te a c h e r f o r c i n g pendant l’entraînement (ratio 1.0 puis schedule décroissant à 0.5 si
temps)
Loss : cross-entropy avec masquage des <pad>
Entraînement sur paires fra→eng
```
### Jour 20 (20 mai) — Décodage : glouton + beam search

```
src/seq/seq2seq/decode.py :
Décodage glouton : argmax token par token jusqu’à <eos>
Beam search : largeur k=3 et k=5, normalisation longueur, comparaison
Inférence sur 50 phrases de test, sauvegarde des sorties pour analyse qualitative
```
### Jour 21 (21 mai) — BLEU + question de synthèse Partie III

```
BLEU corpus avec sacrebleu (ou nltk.translate.bleu_score) sur le set de test
Ta b l e a u f i n a l S e q 2 S e q : B L E U g r e e d y, B L E U b e a m k = 3, B L E U b e a m k = 5
Exemples qualitatifs dans le rapport (3–5 phrases : src, ref, greedy, beam)
Réponse question de synthèse Partie III :
Dans quelle mesure les architectures récurrentes permettent-elles de modéliser
efficacement une séquence réelle, et comment justifier le passage d’un RNN
simple vers un LSTM/GRU puis vers un schéma encodeur–décodeur [...]?
Lien explicite : RNN simple insuffisant (gradient, contexte) → LSTM/GRU
améliorent la mémoire (preuves perplexité) → Seq2Seq nécessaire pour passer
de longueur N à longueur M (preuves BLEU)
```
### Mapping barème Partie III (35 pts)

```
Critère Pts Livré
Théorie séquences 8 J14 + section BPTT J
```

```
RNN / LSTM / GRU 7 J16–J
Préparation données 5 J
Système Seq2Seq 7 J
Décodage + évaluation 5 J20 + J
Question de synthèse 3 J
```
## P4 — Rédaction + question transversale (22–23 mai)

### Jour 22 (22 mai) — Rapport scientifique

Plan obligatoire (mappé sur le CDC) :
1. Introduction (problématique, plan)
2. Objectifs du projet
3. Méthodologie générale (PyTorch, reproductibilité, métriques)
4. Partie I — MLP : théorie · données · implémentation · résultats · question de synthèse
5. Partie II — CNN : théorie + calculs · implémentations manuelles · LeNet · ablations ·
interprétation · question de synthèse
6. Partie III — RNN/LSTM/GRU/Seq2Seq : théorie · LM · Seq2Seq · décodage · BLEU ·
question de synthèse
7. Discussion transversale (cf. J23)
8. Limites et pistes
9. Conclusion
10. Annexes : tableaux complets, courbes, configs
Style : académique, chaque figure / tableau commenté, références théoriques quand
pertinent.

### Jour 23 (23 mai) — Question transversale + finitions

```
Réponse à la question transversale finale (3–5 pages) :
Comment le deep learning adapte-t-il ses architectures à la structure des données —
```

tabulaire, image et séquentielle — et pourquoi un même paradigme d’apprentissage
supervisé doit-il être décliné différemment selon la géométrie, la dépendance locale,
la temporalité et la représentation des données?
Articuler explicitement :

```
Donnée Hypothèse statistique Architecture Mécanisme clé
```
```
Ta b u l a i r e features hétérogènes,pas de voisinage MLP
```
```
combinaison
non linéaire
dense
Image localité spatiale,invariance translation CNN partage despoids, hiérarchie
```
```
Séquence temporalité,dépendances longues RNN/LSTM/GRU/Seq2Seq
```
```
état caché,
portes,
encodeur-
décodeur
```
Conclure sur l’unité du paradigme (descente de gradient, rétropropagation, perte
différentiable) et la diversité nécessaire des biais inductifs.
Finitions :
Relecture rapport
Vérifier que tous les notebooks tournent de bout en bout
README final avec instructions de reproduction
Archive .zip : code + rapport + figures + tables + checkpoints (ou liens)

## Livrables finaux (checklist exhaustive CDC)

```
Rapport scientifique structuré
Code source complet et commenté
Notebooks ou script principal exécutable
Annexe expérimentale : courbes, tableaux, métriques, visualisations
3 questions de synthèse (une par partie) répondues
```

```
Question transversale finale rédigée
requirements.txt + README.md
```
## Risques et mitigations

```
Risque Probabilité Mitigation
GPU surchauffe
/ indisponible moyenne
```
```
Subset Fashion-MNIST + Tatoeba 30k paires max ;
backup sur CPU acceptable pour MLP
Seq2Seq qui ne
converge pas élevée
```
```
Démarrer avec vocab agressivement filtré (top 5k),
embeddings 128, GRU 1 couche, 256 hidden ; teacher
forcing 1.0 fixe en V
Rédaction
négligée en fin
de course
```
```
élevée Écrire pendant= consolidation, pas découverte chaque phase, pas à la fin ; J22–J
```
```
Calculs
manuels CNN
bâclés
```
```
moyenne Les faire à la main sur papier J6 puis recopierproprement, ne pas improviser
```
```
Beam search
bugué moyenne
```
```
Te s t e r d ’a b o r d s u r k= 1 ( d o i t d o n n e r l e g l o u t o n ) , p u i s
k=
```
## Règles d’exécution quotidienne

1. Objectif du jour défini en 1 phrase avant de coder
2. Code → expé → métriques sauvegardées
3. Notes courtes dans le notebook du jour (ce qui marche / ce qui surprend)
4. Commit Git avec message explicite
5. Si retard de plus d’une journée sur une phase → couper dans les ablations, jamais
dans les livrables obligatoires du barème
Ordre absolu : MLP d’abord, CNN ensuite, RNN/Seq2Seq enfin, rédaction toujours en
parallèle.
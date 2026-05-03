# 🗺️ Baby Steps Roadmap : Projet de Fin de Module Deep Learning (EMSI)

Ce document fusionne les exigences du CDC (`projet_DL.md`) et le planning détaillé (`roadmap.md`) en un plan d'action quotidien pour vous guider pas à pas vers la réussite du projet.

---

## 🛠️ Phase 0 : Préparation de l'environnement (30 avril)
- [ ] **Étape 0.1 :** Créer l'arborescence : `data/raw`, `data/processed`, `notebooks`, `src/mlp`, `src/cnn/manual`, `src/seq/rnn`, `src/seq/seq2seq`, `src/utils`, `models`, `outputs/figures`, `outputs/logs`, `outputs/tables`, `reports`.
- [ ] **Étape 0.2 :** Créer et activer l'environnement virtuel `venv`.
- [ ] **Étape 0.3 :** Installer les dépendances : `pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn jupyter tqdm sacrebleu nltk`.
- [ ] **Étape 0.4 :** Initialiser Git + `.gitignore` (ignorer `venv/`, `data/raw/`, `models/`, etc.) et vérifier le GPU (`torch.cuda.is_available()`).
- [ ] **Étape 0.5 :** Créer `src/utils/seed.py` pour garantir la reproductibilité (fixer `random`, `numpy`, `torch`, `torch.cuda`, `cudnn.deterministic`).

---

## 📊 Phase 1 : MLP et Ingénierie PyTorch (1–5 mai)
**Dataset :** Breast Cancer Wisconsin | **Objectif :** 30 points

### Jour 1 (1er mai) — Données + Théorie
- [ ] Télécharger Breast Cancer Wisconsin via `sklearn`.
- [ ] Créer `notebooks/01_mlp_eda.ipynb` : Analyse exploratoire (EDA), valeurs manquantes, déséquilibre, corrélations.
- [ ] Rédiger la théorie dans le notebook : `nn.Module`, `state_dict`, `device`, propagation avant/arrière, rôle du gradient/optimiseur.
- [ ] Pipeline de preprocessing : imputation, `StandardScaler` (sur train uniquement), split stratifié 70/15/15.
- [ ] Sauvegarder les tenseurs prétraités dans `data/processed/bcw_{train,val,test}.pt`.

### Jour 2 (2 mai) — Deux versions du MLP
- [ ] Implémenter V1 avec `nn.Sequential` dans `src/mlp/model_seq.py`.
- [ ] Implémenter V2 avec classe héritant de `nn.Module` dans `src/mlp/model_class.py` (avec hook pour activations).
- [ ] Comparer V1 et V2 dans le rapport (flexibilité, lisibilité).
- [ ] Inspecter les modèles : afficher `named_parameters()` et `state_dict().keys()`, puis commenter.

### Jour 3 (3 mai) — Stratégies d'initialisation
- [ ] Coder 3 fonctions dans `src/mlp/init_strategies.py` : Gaussienne (0, 0.01), Constante (0.1), Xavier Uniform.
- [ ] Entraîner les 3 versions (mêmes hyperparamètres, même seed).
- [ ] Tracer côte-à-côte les courbes de loss train/val.
- [ ] Créer un tableau comparatif (accuracy finale, époque de convergence) et conclure (notamment sur la symétrie de l'init constante).

### Jour 4 (4 mai) — Entraînement, Sauvegarde, GPU
- [ ] Implémenter la boucle d'entraînement propre : `BCEWithLogitsLoss`, `Adam`, Early Stopping sur validation.
- [ ] Sauvegarder le meilleur modèle : `torch.save(model.state_dict(), 'models/mlp_best.pt')`.
- [ ] Recharger le modèle dans un notebook et vérifier l'identité parfaite des prédictions.
- [ ] Vérifier le comportement CPU vs GPU (assertions `device`).
- [ ] Sauvegarder les logs (configs et métriques) en JSON.

### Jour 5 (5 mai) — Évaluation + Synthèse I
- [ ] Calculer les métriques sur le Test set : accuracy, precision, recall, F1, ROC-AUC, matrice de confusion.
- [ ] Sauvegarder les figures dans `outputs/figures/mlp/`.
- [ ] Rédiger la **Question de Synthèse I** (pertinence du MLP sur tabulaire, limites face à la structure des données).

---

## 🖼️ Phase 2 : CNN et Vision par Ordinateur (6–13 mai)
**Dataset :** Fashion-MNIST | **Objectif :** 35 points

### Jour 6 (6 mai) — Théorie + Calculs manuels
- [ ] Rédiger dans `notebooks/02_cnn_theory.ipynb` : Pourquoi un MLP est inadapté aux images (localité, invariance).
- [ ] Faire les calculs manuels demandés : corrélation croisée 2D (matrice 4x4, noyau 3x3), formules de taille de sortie (avec différents padding/stride), taille après pooling.
- [ ] Préparer le dataset Fashion-MNIST via `torchvision.datasets`.

### Jour 7 (7 mai) — Implémentations manuelles
- [ ] Coder dans `src/cnn/manual/` : `corr2d()`, `max_pool2d()`, `avg_pool2d()` from scratch (numpy ou boucles).
- [ ] Tester sur une matrice 5x5 : comparer bit à bit avec les fonctions PyTorch `F.conv2d`, `F.max_pool2d`, `F.avg_pool2d`.
- [ ] Appliquer ces 3 fonctions sur une image Fashion-MNIST et visualiser le résultat.

### Jour 8 (8 mai) — LeNet Baseline
- [ ] Implémenter une baseline LeNet dans `src/cnn/lenet.py`.
- [ ] Entraîner LeNet sur Fashion-MNIST (~10 époques, Adam), sauvegarder le modèle.
- [ ] Implémenter un MLP simple pour images (Flatten -> Dense).
- [ ] Comparer MLP et LeNet (accuracy, nombre de paramètres, temps/époque).

### Jour 9 (9 mai) — Ablation Padding & Stride
- [ ] Tester 4 configurations : Référence, `padding=0`, `stride=2`, noyau `3x3` au lieu de `5x5`.
- [ ] Dresser un tableau comparatif : accuracy test, taille des feature maps, nb paramètres.

### Jour 10 (10 mai) — Ablation Pooling & Filtres
- [ ] Tester : AvgPool vs MaxPool.
- [ ] Tester : Différents nombres de filtres (ex: 6/16 vs 32/64).
- [ ] Tester : Ajout d'une convolution `1x1` entre les blocs.
- [ ] Dresser un tableau récapitulatif + superposer les courbes de loss.

### Jour 11 (11 mai) — Visualisation des cartes de caractéristiques
- [ ] Placer un hook sur les sorties des couches convolutives.
- [ ] Afficher les feature maps (couches 1 et 2) pour 3 à 5 images de classes différentes.
- [ ] Rédiger l'interprétation : que détectent ces couches (bords vs motifs abstraits) ?

### Jour 12 (12 mai) — Consolidation comparative
- [ ] Créer le "Tableau Master" Partie II : MLP vs LeNet baseline vs LeNet+1x1 vs Meilleure variante.
- [ ] Sauvegarder les figures finales proprement dans `outputs/figures/cnn/`.

### Jour 13 (13 mai) — Synthèse II
- [ ] Rédiger la **Question de Synthèse II** (Pourquoi CNN > MLP pour images, influence réelle de padding/stride/pooling/profondeur).

---

## 📝 Phase 3 : RNN, LSTM, GRU et Seq2Seq (14–21 mai)
**Dataset :** Tatoeba (Français -> Anglais) | **Objectif :** 35 points

### Jour 14 (14 mai) — Théorie modélisation de langue
- [ ] Rédiger dans `notebooks/03_seq_theory.ipynb` : Objectif probabiliste, règle de chaîne, perplexité.
- [ ] Expliquer le rôle du vocabulaire, du padding, du masquage et des tokens spéciaux (`<bos>`, `<eos>`, `<pad>`, `<unk>`).
- [ ] Télécharger Tatoeba et faire un premier nettoyage (lowercase, ponctuation).

### Jour 15 (15 mai) — Préparation des données
- [ ] Coder `src/seq/data.py` : Tokenisation, construction du vocabulaire (src/tgt), filtrage des longueurs.
- [ ] Créer un DataLoader avec collate function (padding dynamique).
- [ ] Préparer un corpus monolingue anglais (fenêtres glissantes) pour la modélisation de langue.

### Jour 16 (16 mai) — RNN simple (Modèle de langue)
- [ ] Implémenter un Vanilla RNN (`nn.RNN`) dans `src/seq/rnn/vanilla_rnn.py` avec tête linéaire.
- [ ] Entraîner sur le corpus anglais, mesurer la perplexité (train/val).
- [ ] Illustrer l'effet du *Gradient Clipping* : 2 runs (avec/sans), tracer la norme du gradient et commenter l'explosion (exploding gradient).

### Jour 17 (17 mai) — LSTM
- [ ] Remplacer par `nn.LSTM` avec les mêmes hyperparamètres.
- [ ] Comparer la perplexité avec le RNN simple.
- [ ] Discuter des avantages : portes, mémoire à long terme, stabilité du gradient.

### Jour 18 (18 mai) — GRU & Comparatif
- [ ] Remplacer par `nn.GRU`.
- [ ] Créer un tableau comparatif final : perplexité, temps/époque, paramètres. Superposer les courbes de loss.
- [ ] Rédiger la section BPTT (Backpropagation Through Time) et troncature dans le rapport.

### Jour 19 (19 mai) — Seq2Seq (Encodeur-Décodeur)
- [ ] Implémenter `src/seq/seq2seq/` : Encodeur (Embedding + GRU) et Décodeur (Embedding + GRU + Tête Linéaire vers vocabulaire cible).
- [ ] Gérer le passage de l'état caché final de l'encodeur à l'état initial du décodeur.
- [ ] Implémenter le *Teacher Forcing*. Entraîner sur paires fr -> en (Cross-Entropy avec masquage du `<pad>`).

### Jour 20 (20 mai) — Décodage (Glouton + Beam Search)
- [ ] Implémenter dans `src/seq/seq2seq/decode.py` : Décodage glouton (argmax).
- [ ] Implémenter : Beam Search (largeurs k=3 et k=5).
- [ ] Faire l'inférence sur 50 phrases de test, sauvegarder les sorties pour analyse qualitative.

### Jour 21 (21 mai) — Évaluation BLEU + Synthèse III
- [ ] Calculer le score BLEU (Sacrebleu/NLTK) sur le test set (Comparatif : Greedy vs Beam k=3 vs Beam k=5).
- [ ] Ajouter des exemples qualitatifs de traduction dans le rapport.
- [ ] Rédiger la **Question de Synthèse III** (Justification du passage RNN simple -> LSTM/GRU -> Encodeur-Décodeur pour la traduction).

---

## 📚 Phase 4 : Rédaction et Livrables (22–23 mai)

### Jour 22 (22 mai) — Rapport scientifique
- [ ] Rédiger selon le plan obligatoire : Intro, Objectifs, Méthodologie, Partie I, Partie II, Partie III.
- [ ] Ajouter la section : Limites et pistes d'amélioration.
- [ ] Vérifier que chaque figure et tableau est rigoureusement commenté et analysé.

### Jour 23 (23 mai) — Question transversale + Finitions
- [ ] Rédiger la **Question Transversale Finale** (Comment le DL adapte ses architectures selon les structures de données : Tabulaire, Image, Séquence).
- [ ] Relire le rapport (orthographe, style académique).
- [ ] Vérifier que tous les notebooks tournent de bout en bout sans erreur (Kernel Restart & Run All).
- [ ] Rédiger le `README.md` (instructions claires pour reproduire votre travail).
- [ ] Créer l'archive `.zip` finale contenant : Code complet, Rapport PDF, figures, checkpoints (ou liens) et `requirements.txt`.

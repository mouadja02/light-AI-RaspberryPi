# Reconnaissance de Chiffres Manuscrits avec MLP et CNN

Ce dépôt présente un projet de reconnaissance de chiffres manuscrits qui compare deux approches :
- **MLP (Perceptron Multicouche)**
- **CNN (Réseau de Neurones Convolutif)**

Le projet utilise PyTorch pour l'entraînement des modèles, avec exportation des poids dans des fichiers `.h` afin de réaliser une phase d'inférence en langage C. Cette approche hybride permet d'obtenir des performances optimisées, notamment pour une intégration dans des systèmes embarqués.

---

## Table des matières

- [Aperçu](#aperçu)
- [Architecture](#architecture)
- [Prétraitement des Images](#prétraitement-des-images)
- [Entraînement et Export des Poids](#entraînement-et-export-des-poids)
- [Compilation et Inférence en C](#compilation-et-inférence-en-c)
- [Exécution du Projet](#exécution-du-projet)


---

## Aperçu

Ce projet vise à démontrer l'efficacité de deux architectures de réseaux de neurones pour la reconnaissance de chiffres manuscrits (de 0 à 9).  
- Le **MLP** atteint environ 95% de précision.
- Le **CNN** offre une meilleure robustesse et une capacité de généralisation supérieure, grâce à l'exploitation des caractéristiques spatiales des images.

Le code d'inférence en C, compilé avec les poids exportés, est destiné à des applications embarquées nécessitant des performances optimales.

---

## Architecture

Le projet comporte deux architectures principales :

### MLP
- **Input** : Image 28×28 pixels (après prétraitement)
- **Couches cachées** : Plusieurs couches linéaires avec activation ReLU
- **Output** : 10 neurones correspondant aux 10 classes (chiffres de 0 à 9)

### CNN
- **Input** : Image 28×28 pixels en niveaux de gris
- **Couches Convolutives** :  
  - Conv1 : 32 filtres, kernel 3×3, padding 1, suivi d'une activation ReLU  
  - Pooling : MaxPool 2×2  
  - Conv2 : 64 filtres, kernel 3×3, padding 1, suivi d'une activation ReLU  
  - Pooling : MaxPool 2×2  
- **Couches Fully Connected** :  
  - FC1 : Passage de 3136 (64×7×7) à 128 neurones avec activation ReLU  
  - FC2 : Passage de 128 à 10 neurones

---

## Prétraitement des Images

Avant l'entraînement et l'inférence, les images sont :
- **Cadrées**
- **Seuilées**
- **Redimensionnées** en 28×28 pixels
- **Retournées verticalement** pour assurer une homogénéité entre les données d'entraînement et d'inférence

---

## Entraînement et Export des Poids

Les modèles sont entraînés avec PyTorch.  
Les poids obtenus sont ensuite exportés dans des fichiers `.h` :
- `mlp_weights.h` pour l'architecture MLP
- `cnn_weights.h` pour l'architecture CNN

Cela permet de les utiliser dans le code C pour l'inférence.

---

## Compilation et Inférence en C

Le code C (`digits_id.c`) réalise la phase d'inférence.  
La compilation se fait en passant une macro qui sélectionne l'architecture souhaitée :
- `-D USE_MLP` pour le MLP
- `-D USE_CNN` pour le CNN

---

## Exécution du Projet

Un script shell `run_project.sh` est fourni pour simplifier l'exécution du projet.  
Ce script permet de :
- Choisir l'architecture (MLP ou CNN)
- Spécifier le chemin de l'image à tester
- Sélectionner le mode d'exécution :  
  - `"train"` pour entraîner le modèle et exporter les poids  
  - `"notrain"` pour utiliser les poids déjà exportés

### Exemple d'utilisation :

```bash
# Entraîner et tester avec le CNN
./run_project.sh CNN images/exemple.bmp train

# Tester sans réentraînement avec le MLP
./run_project.sh MLP images/exemple.bmp notrain

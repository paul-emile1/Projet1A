## Description du code

Le fichier `main.py` contient l'ensemble des simulations numériques du projet.

### 1. Résolution déterministe

Nous résolvons l'équation de Schrödinger unidimensionnelle sur [0,1], avec conditions de Dirichlet homogènes :

    -u''(x) + V(x) u(x) = f(x)

La résolution se fait d'abord par différences finies, puis à l'aide de la matrice des fonctions de Green associée à l'opérateur discrétisé.

### 2. Modèle stochastique

Nous générons un potentiel aléatoire de la forme :

    V_eps(x) = V_0(x) + sigma * mu_eps(x)

où `mu_eps` est un bruit blanc discrétisé. L'équation stochastique correspondante est ensuite résolue numériquement via différences finies.

### 3. Étude de convergence

Deux types de convergence sont étudiés (lorsque eps->0) :

- **Convergence L²** :
  
      E[ ||u_eps - u_0||^2 ] ~ C * eps    (vérifiée numériquement)

- **Convergence en distribution** :

      (u_eps(x) - u_0(x)) / sqrt(eps)  → loi gaussienne

Cette dernière est illustrée par comparaison entre la fonction de répartition empirique et celle d'une loi normale cible.

Les figures obtenues valident les résultats théoriques du rapport.


## Exécution

Installer les dépendances nécessaires :

```bash
pip install -r requirements.txt


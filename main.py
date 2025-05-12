import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.stats import norm


### CONSTANTES ###

N = 1000  # Nombre de sous-intervalles
L = 1.0  # Longueur de l'intervalle [0,1]
h = L / N  # Taille d'un élément
x_list = np.linspace(0, L, N+1)  # Points de discrétisation
sigma = 20
eps = 0.001
d = 1 # Dimension


# Second membre
def f(x):
    return x * (1 - x)




### RÉSOLUTION ÉQUATION DÉTERMINISTE (différences finies et fonctions de Green) ###

def potentiel_deterministe(x):
    return 20 * (4 + np.sin(2 * np.pi * x))

# Construction de la matrice A
def matrice_schrodinger_deterministe(V=potentiel_deterministe):
    A = np.zeros((N+1, N+1))
    for i in range(1, N):
        A[i, i] = 2 / h**2 + V(x_list[i])
        A[i, i-1] = A[i, i+1] = -1 / h**2
    # Conditions de Dirichlet : u(0) = u(1) = 0
    A[0, :] = 0
    A[0, 0] = 1
    A[N, :] = 0
    A[N, N] = 1
    return A

def solution_deterministe(V=potentiel_deterministe, y=x_list):
    A = matrice_schrodinger_deterministe()
    f_vec = np.array([f(y[i]) for i in range(N+1)])
    f_vec[0], f_vec[N] = 0, 0
    u = np.linalg.solve(A, f_vec)
    return u

# Fonction inutile
def Green(j, V=potentiel_deterministe, y=x_list):
    A = matrice_schrodinger_deterministe()
    f_vec = np.zeros(N+1)
    f_vec[j] = 1. / h  # Dirac en ce point
    f_vec[0], f_vec[N] = 0, 0
    G = np.linalg.solve(A, f_vec)
    return G

def solution_green():
    A = matrice_schrodinger_deterministe()
    G_mat = np.linalg.inv(A)
    f_vec = np.array([f(x) for x in x_list])
    f_vec[0], f_vec[N] = 0, 0
    u = np.zeros(N+1)
    for i in range(N+1):
        # Somme de Riemann pour chaque point du maillage
        for k in range(N):
            u[i] += G_mat[i, k] * f(k / N)
    return u

u_deterministe = solution_deterministe()



# Tracer les deux solutions
plt.figure(1)
plt.plot(x_list, u_deterministe, label='Différences finies')
plt.plot(x_list, solution_green(), '--', label="Fonctions de Green")
plt.legend()
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Comparaison des solutions déterministes')
plt.grid()







### RÉSOLUTION ÉQUATION STOCHASTIQUE ###

def potentiel_stochastique(x, sigma, eps):
    V0 = potentiel_deterministe(x)
    # Générer un champ aléatoire non corrélé
    num_segments = int(np.ceil(L / eps))
    mu_values = np.random.randn(num_segments) 
    # Assigner une valeur aléatoire constante à chaque segment
    mu_discontinuous = np.zeros_like(x)
    for i in range(num_segments):
        start_idx = int(i * eps / h)
        end_idx = int((i + 1) * eps / h) if i < num_segments - 1 else N
        mu_discontinuous[start_idx:end_idx] = mu_values[i]
    return V0 + sigma * mu_discontinuous

# Construction de la matrice A (équation stochastique)
def matrice_schrodinger(V):
    A = np.zeros((N+1, N+1))
    for i in range(1, N):
        A[i, i] = 2 / h**2 + V[i]
        A[i, i-1] = A[i, i+1] = -1 / h**2
    # Conditions de Dirichlet : u(0) = u(1) = 0
    A[0, :] = 0
    A[0, 0] = 1
    A[N, :] = 0
    A[N, N] = 1
    return A

def solution_stochastique(V, y=x_list):
    A = matrice_schrodinger(V)
    f_vec = np.array([f(y[i]) for i in range(N+1)])
    f_vec[0], f_vec[N] = 0, 0
    u = np.linalg.solve(A, f_vec)
    return u

# Génération du potentiel stochastique
V_stochastique = potentiel_stochastique(x_list, sigma, eps)

# Résolution du problème de Schrödinger stochastique
u_stochastique = solution_stochastique(V_stochastique)

# Affichage des résultats
plt.figure(2)
plt.plot(x_list, V_stochastique, label='Potentiel stochastique')
plt.plot(x_list, potentiel_deterministe(x_list), label='Potentiel déterministe')
plt.legend()
plt.xlabel('x')
plt.ylabel('V(x)')
plt.title('Potentiels déterministe et stochastique')
plt.grid()

plt.figure(3)

#Calcul de l'espérance des réalisations et d'un IC à 95%

n_real = 100  # nombre de réalisations
u_list = np.zeros((n_real, N+1))  

for m in range(n_real):
    V_stoch = potentiel_stochastique(x_list, sigma, eps)
    u_list[m, :] = solution_stochastique(V_stoch)

# Moyenne et écart-type
mean_u = np.mean(u_list, axis=0)
std_u = np.std(u_list, axis=0, ddof=1)  

# Intervalle de confiance à 95%
lower_bound = mean_u - 1.96 * std_u / np.sqrt(n_real)
upper_bound = mean_u + 1.96 * std_u / np.sqrt(n_real)



V_stochastique1 = potentiel_stochastique(x_list, sigma, eps)
V_stochastique2 = potentiel_stochastique(x_list, sigma, eps)

plt.plot(x_list, solution_stochastique(V_stochastique1), label='u(x, ω₁)')
plt.plot(x_list, solution_stochastique(V_stochastique2), label='u(x, ω₂)')
plt.plot(x_list, solution_deterministe(), label='u(x) déterministe')
plt.plot(x_list,mean_u, label = "E(u(x,ω))")
plt.plot(x_list, lower_bound,'--' ,color='gray', label="IC à 95%")
plt.plot(x_list,upper_bound,'--',color = 'gray')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Comparaison des solutions pour deux réalisations')
plt.legend()
plt.grid()






### CONVERGENCE ERREUR L^2 ###

def esperance_erreur_quadratique(eps,sigma = 20, nb_real=100):
    erreur_quadratique = []
    u_0 = solution_deterministe()  # solution de référence
    for _ in range(nb_real):
        V_stoch = potentiel_stochastique(x_list, sigma=sigma, eps=eps)
        u_eps = solution_stochastique(V_stoch)
        diff = u_eps - u_0
        erreur = h * np.sum(diff**2)  # norme L2 discrète
        erreur_quadratique.append(erreur)
    return sqrt(sum(erreur_quadratique) / nb_real)

eps_list = np.logspace(-3, -1.5, 20)
erreurs = [esperance_erreur_quadratique(eps) for eps in eps_list]

plt.figure(4)
plt.loglog(eps_list, erreurs, 'o-', label="Erreur quadratique moyenne")
plt.plot(eps_list, np.sqrt(eps_list), '--', label=r"$\sqrt{\varepsilon}$ (référence)")
plt.xlabel(r"$\varepsilon$", fontsize = 11)
plt.ylabel(r"$\sqrt{\mathbb{E}[\|u^\varepsilon - u^0\|^2]}$", fontsize = 14)
plt.title("Convergence de l'erreur moyenne (log-log)")
plt.legend()
plt.grid(True, which="both", ls="--")





### THEOREME CENTRALE LIMITE ###

x_1 = N//2

# Fonction pour calculer Z_eps et sigma_x1
def Z(eps, n_real):
    u_0 = solution_deterministe()
    Z_list = []
    for i in range(n_real):
        V_sto = potentiel_stochastique(x_list, sigma, eps)
        u_eps = solution_stochastique(V_sto)
        Z_list.append((u_eps[x_1] - u_0[x_1]) / sqrt(eps))
    G = np.linalg.inv(matrice_schrodinger_deterministe())
    sigma_x1 = sqrt((sigma**2/ h) * sum((G[x_1, k] * u_0[k])**2 for k in range(N+1)))
    return Z_list, sigma_x1

# Calculer Z_list et sigma_x1
Z_list, sigma_x1 = Z(eps=eps, n_real=1000)

# Calculer les valeurs de t pour la CDF
t_vals = np.linspace(min(Z_list), max(Z_list), 500)

# Fonction pour calculer la CDF empirique
def fonction_repartition(data, t_vals):
    return [np.mean(np.array(data) <= t) for t in t_vals]

# CDF empirique
F_emp = fonction_repartition(Z_list, t_vals)

# CDF théorique d'une N(0, sigma_x1^2)
F_theo = norm.cdf(t_vals, loc=0, scale=sigma_x1)

# Tracé
plt.figure(5)
plt.plot(t_vals, F_emp, label=r"$F_{Z^\varepsilon}$ (empirique)")
plt.plot(t_vals, F_theo, 'r--', label=r"$F_{\mathcal{N}(0, \sigma^2)}$")
plt.xlabel("t")
plt.ylabel("Fonction de répartition")
plt.title("Comparaison des fonctions de répartition")
plt.grid()
plt.legend()








plt.show()

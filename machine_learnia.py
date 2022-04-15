import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv', delimiter=',')[1:]
print(data)
x = [[240000.], [139800.], [150500.], [185530.], [176000.], [114800.], [166800.],  [89000.], [144500.],
  [84000.],  [82029.],  [63060.],  [74000.],  [97500.],  [67000.],  [76025.],  [48235.],  [93000.],
  [60949.],  [65674.],  [54000.],  [68500.],  [22899.],  [61789.]]
y = [[3650.], [3800.], [4400.], [4450.], [5250.], [5350.], [5800.], [5990.], [5999.], [6200.], [6390.], [6390.],
[6600.], [6800.], [6800.], [6900.], [6900.], [6990.], [7490.], [7555.], [7990.], [7990.], [7990.], [8290.]]

# np.random.seed(0) # pour toujours reproduire le meme dataset
 
# n_samples = 100 # nombre d'echantillons a générer
# x = np.linspace(0, 10, n_samples).reshape((n_samples, 1))
print(x)
# y = x + np.random.randn(n_samples, 1)
print(y)
 
 
plt.scatter(x, y) # afficher les résultats. X en abscisse et y en ordonnée
plt.show()

# ajout de la colonne de biais a X
X = np.hstack((x, np.ones(x.shape)))
print(X)
print(X.shape)
 
# création d'un vecteur parametre theta
theta = np.random.randn(2, 1)
print(theta)


def model(X, theta):
    return X.dot(theta)
 
def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)
 
def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)
 
def gradient_descent(X, y, theta, learning_rate, n_iterations):
    # création d'un tableau de stockage pour enregistrer l'évolution du Cout du modele
    cost_history = np.zeros(n_iterations) 
     
    for i in range(0, n_iterations):
        theta = theta - learning_rate * grad(X, y, theta) # mise a jour du parametre theta (formule du gradient descent)
        cost_history[i] = cost_function(X, y, theta) # on enregistre la valeur du Cout au tour i dans cost_history[i]
         
    return theta, cost_history


n_iterations = 1000
learning_rate = 0.01
 
theta_final, cost_history = gradient_descent(X, y, theta, learning_rate, n_iterations)
 
print(theta_final) # voici les parametres du modele une fois que la machine a été entrainée
 
# création d'un vecteur prédictions qui contient les prédictions de notre modele final
predictions = model(X, theta_final)
 
# Affiche les résultats de prédictions (en rouge) par rapport a notre Dataset (en bleu)
plt.scatter(x, y)
plt.plot(x, predictions, c='r')
plt.show()
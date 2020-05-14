import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

Phi0 = 100
dl = 0.1
xMax, xMin = 1, 0

yMax, yMin = 1, 0
x1_ = np.arange(xMin, xMax + dl, dl)
y1_ = np.arange(yMin, yMax + dl, dl)

x = np.tile(x1_, len(y1_))  # se crean las posiciones en x de los triangulos
y = np.repeat(y1_, len(x1_))  # se crean las posiciones en y de los triangulos
# se usa tile para unir vectores y repeat para repetir las componentes (ver vectores)

triang = tri.Triangulation(x, y)  # se crean los triangulos

# SE DEFINEN CONDICIONES DE FRONTERA
Vi = np.zeros((len(y1_), len(x1_)))
Vi[0, :] = (Phi0 / xMax) * x1_
Vi[:, -1] = (Phi0 / yMax) * (yMax - y1_)
# Vi[:, 0] = np.sin((yMax - y1_) *2* np.pi)
# Vi[:, -1] = np.sin((yMax - y1_) *5* np.pi)

# Vi[0, :] = 1
# Vi[:, -1] = 1

is_Condition = np.zeros((len(y1_), len(x1_)))
is_Condition[0, :] = 1
is_Condition[:, -1] = 1
is_Condition[-1, :] = 1
is_Condition[:, 0] = 1


VAL = []
NDP = []
for j in range(0, len(y1_)):
    for i in range(0, len(x1_)):
        if is_Condition[len(y1_) - j - 1, i] == 1:
            VAL.append(Vi[len(y1_) - j - 1, i])
            NDP.append(j * len(x1_) + i + 1)

# La forma de hallar NDP y VAL se puede mejorar, sin embargo funciona y evita copiar todos los vectores

# se procede a encontrar las demás variables necesarias para el método

NE = len(triang.triangles)  # numero de elementos
ND = len(x)  # numero de nodos
NP = len(NDP)  # numero de nodos fijos
NL = triang.triangles  # se definen trios de nodos(cada trio representa un triangulo)

##############################################################
#                                                            #
#              INICIA EL MÉTODO NÚMERICO                     #
#                                                            #
##############################################################

Ct = np.zeros((ND, ND))
# se sacan Ce de cada elemento y se insertan a Ctotal
for ii, elem in enumerate(NL):
    Ce = np.zeros((3, 3))
    x_elem = x[elem]
    y_elem = y[elem]
    Q = np.array([np.ones(len(x_elem)), x_elem, y_elem]).T
    Qinv = np.linalg.inv(Q)
    A = (1 / 2) * (
            (x_elem[1] - x_elem[0]) * (y_elem[2] - y_elem[0]) - (x_elem[2] - x_elem[0]) * (y_elem[1] - y_elem[0]))
    grad_alpha = Qinv[1:].T
    for i in range(0, 3):
        for j in range(0, 3):
            Ce[i][j] = np.dot(grad_alpha[i], grad_alpha[j]) * A
    Ct[np.ix_(elem, elem)] += Ce
# Hasta acá se tienen las matrices Ce de cada elemento y Ct total

# se procede a encontrar los nodos libres
LF = list(set(range(1, ND + 1)) - set(NDP))  # Nodos libres (sin condicion de frontera)

# se crean las matrices CFF Y CFP, para luego solucionar el sistema

CFF = np.zeros((len(LF), len(LF)))
CFP = np.zeros((len(LF), ND - len(LF)))
for i, ii in enumerate(LF):
    for j, jj in enumerate(LF):
        CFF[i][j] = Ct[ii - 1][jj - 1]
    for j, jj in enumerate(NDP):
        CFP[i][j] = -Ct[ii - 1][jj - 1]
F_phi = np.linalg.inv(CFF) @ CFP @ VAL  # este vector es el de los potenciales encontrados

# ahora debemos juntar los potenciales de condición de frontera (VAL) con los encontrados (F_phi)

# Se crea el vector Phi
u, v = 0, 0
Phi = []
# se junta VALS con F_phi
for j in range(0, ND):
    try:
        if NDP[u] == u + 1 + v:
            Phi.append(VAL[u])
            u += 1
        else:
            Phi.append(F_phi[v])
            v += 1
    except:
        Phi.append(F_phi[v])
        v += 1
Phi = np.array(Phi)
# El código anterior se puede mejorar, pero funciona, y esencialmente nos da un vector con todos los potenciales.


##############################################################
#                                                            #
#                           GRAFICAS                         #
#                                                            #
##############################################################


X_, Y_ = np.meshgrid(x1_, y1_)
Z = Phi0*X_ * Y_/(xMax*yMax)
Zphi = np.resize(Z, (1, X_.shape[0] * Y_.shape[1]))[0]  # Phi teórico

X1_, Y1_ = np.meshgrid(np.arange(xMin, xMax + 0.001, 0.001), np.arange(yMin, yMax + 0.001, 0.001))
Z1 = Phi0*X1_ * Y1_/(xMax*yMax)
plt.rcParams.update({'font.size': 12})
# # Grafica 1. Phi por Nodo
Phi = np.array(Phi)
plt.stem(np.arange(1, ND + 1), Phi)
plt.title('Potencial Por Nodo')
plt.ylabel('Potencial [V]')
plt.xlabel('# Nodo')
plt.show()

# ---------------------------------------------------------------------------------------
# Grafica 2. Numeración de nodos
# ---------------------------------------------------------------------------------------
fig1, ax1 = plt.subplots()
for i, (xx, yy) in enumerate(zip(x, y)):
    plt.text(xx, yy, i + 1, dict(size=20))

ax1.set_aspect('equal')
ax1.triplot(triang, 'bo-', lw=1)
ax1.set_title('Numeración de Nodos')
plt.show()
# ---------------------------------------------------------------------------------------
# Grafica 3. Potencial FEM
# ---------------------------------------------------------------------------------------

plt.figure()
CS3 = plt.contour(X1_, Y1_, Z1, colors='k', linestyles='dashed', linewidths=2)
CS4 = plt.tricontour(triang, Phi, colors='k', linestyles='solid', linewidths=1)
h1, _ = CS3.legend_elements()
h2, _ = CS4.legend_elements()
plt.legend([h1[0], h2[0]], ['Teórico', 'Aproximación'])
plt.title('Método Elementos Finitos Vs Análisis Teórico')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.xlim(xMin, xMax)
plt.ylim(yMin, yMax)
plt.show()
# ---------------------------------------------------------------------------------------
# Grafica 4. Comparación de Gráficas
# ---------------------------------------------------------------------------------------

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
PLOT1 = ax1.contour(X_, Y_, Z, cmap='winter', linewidths=3)
PLOT2 = ax1.plot_surface(X_, Y_, Z, cstride=1, rstride=1, cmap='winter', alpha=0.6, linewidth=0,
                         antialiased=False)
plt.contourf(X_, Y_, Z, zdir='z', cmap='winter', offset=-0.1)
plt.contourf(X_, Y_, Z, zdir='x', cmap='winter', offset=xMin - 0.1, alpha=0.2)
plt.contourf(X_, Y_, Z, zdir='y', cmap='winter', offset=yMax + 0.1, alpha=0.2)
fig.colorbar(PLOT2, ax=ax1, label='Tensión \u03C6 [V]')
ax1.set_title('Potencial - Análisis Teórico')
ax1.set_xlabel('X [m]')
ax1.set_ylabel('Y [m]')
ax1.set_zlabel('Tensión \u03C6 [V]')

# segundo subplot

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
PLOT3 = ax2.plot_trisurf(x, y, Phi, triangles=triang.triangles, cmap='coolwarm', alpha=0.6, linewidth=0,
                         antialiased=False)
PLOT4 = ax2.tricontour(triang, Phi, cmap='coolwarm', linewidths=3)
ax2.tricontourf(triang, Phi, zdir='z', cmap='coolwarm', offset=np.min(Phi) - 0.1)
ax2.tricontourf(triang, Phi, zdir='x', cmap='coolwarm', offset=xMin - 0.1, alpha=0.2)
ax2.tricontourf(triang, Phi, zdir='y', cmap='coolwarm', offset=yMax + 0.1, alpha=0.2)
fig.colorbar(PLOT3, ax=ax2, label='Tensión \u03C6 [V]')

ax2.set_title('Potencial - Método de Elementos Finitos (FEM)')
ax2.set_xlabel('X [m]')
ax2.set_ylabel('Y [m]')
ax2.set_zlabel('Tensión \u03C6 [V]')

# ---------------------------------------------------------------------------------------
# ERROR
# ---------------------------------------------------------------------------------------
ERROR = np.mean(abs(Phi - Zphi))
print('el error es: ' + str(np.max(ERROR)))

# ---------------------------------------------------------------------------------------
# Grafica 5. Intensidad de Campo Eléctrico
# ---------------------------------------------------------------------------------------

fig, ax = plt.subplots()
ax.set_aspect('equal')

ax.tricontour(triang, Phi, colors='black', linewidths=1, linestyles='dashed')
ax.triplot(triang, 'b-', lw=0.1)
tpc = ax.tripcolor(triang, Phi, cmap='coolwarm', shading='gouraud')
fig.colorbar(tpc, label='Tensión \u03C6 [V]')

tci = tri.LinearTriInterpolator(triang, -Phi)
(Ex, Ey) = tci.gradient(triang.x, triang.y)
E_norm = np.sqrt(Ex ** 2 + Ey ** 2)

ex = np.resize(Ex, (len(y1_), len(x1_)))
ey = np.resize(Ey, (len(y1_), len(x1_)))
# Lineas que sobran
numPoints = 8
xstart = np.linspace(0, xMax, numPoints)
ystart = np.linspace(0, yMax, numPoints)
start = np.zeros((len(xstart) + len(ystart), 2)) + dl / 10
start[len(xstart):, 1] = ystart
start[:len(xstart), 0] = xstart
ax.streamplot(X_, Y_, ex, ey, color='k', linewidth=1, density=2)
# ax.streamplot(X_, Y_, ex, ey, color='k', start_points=start, linewidth=1, density=2)

ax.set_title('Intensidad de Campo Eléctrico (FEM)')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_xlim(xMin, xMax)
ax.set_ylim(yMin, yMax)
# ---------------------------------------------------------------------------------------

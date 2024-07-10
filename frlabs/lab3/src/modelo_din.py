import rbdl
import numpy as np


# Lectura del modelo del robot a partir de URDF (parsing)
modelo = rbdl.loadModel('../urdf/Robot.urdf')
# Grados de libertad
ndof = modelo.q_size

# Configuracion articular
q = np.array([0.4, 0.5, 0.2, 0.3, 0.8, 0.5, 0.6])
# Velocidad articular
dq = np.array([0.2, 0.8, 0.7, 0.8, 0.6, 0.9, 0.2])
# Aceleracion articular
ddq = np.array([0.1,0.2, 0.5, 0.4, 0.3, 1.0, 0.1])

# Arrays numpy
zeros = np.zeros(ndof)          # Vector de ceros
tau   = np.zeros(ndof)          # Para torque
g     = np.zeros(ndof)          # Para la gravedad
c     = np.zeros(ndof)          # Para el vector de Coriolis+centrifuga
M     = np.zeros([ndof, ndof])  # Para la matriz de inercia
e     = np.eye(ndof)            # Vector identidad
mi = np.zeros(ndof)             # Vector de ceros

# Torque dada la configuracion del robot
rbdl.InverseDynamics(modelo, q, dq, ddq, tau)

# Parte 1: Calcular vector de gravedad, vector de Coriolis/centrifuga,
# y matriz M usando solamente InverseDynamics

print("Parte 1:")
# vector de gravedad: g = ID(q,0,0)
rbdl.InverseDynamics(modelo, q, zeros, zeros, g)
print("Vector de gravedad (g):")
print(np.round(g,3))
print("\n")

# vector de coriolis: c = (ID(q,dq,0)-g)/dq
rbdl.InverseDynamics(modelo, q, dq, zeros, c)
c = c-g
print("Vector de Coriolis + Centrifuga (c):")
print(np.round(c,3))
print("\n")

# matriz de inercia: M[1,:] = (ID(dq,0,e[1,:]) )/e[1,:]
for i in range(ndof):
  rbdl.InverseDynamics(modelo, q, zeros, e[i,:], mi)
  M[i,:] = mi - g
print("Matriz de Inercia (M):")
print(np.round(M,3))
print("\n")


# Parte 2: Calcular M y los efectos no lineales b usando las funciones
# CompositeRigidBodyAlgorithm y NonlinearEffects. Almacenar los resultados
# en los arreglos llamados M2 y b2
b2 = np.zeros(ndof)          # Para efectos no lineales
M2 = np.zeros([ndof, ndof])  # Para matriz de inercia

print("Parte 2:")

rbdl.CompositeRigidBodyAlgorithm(modelo, q, M2)
print('MATRIZ INERCIA - CRBA')
print(np.round(M2,3))
print(' ')

rbdl.NonlinearEffects(modelo,q,dq , b2)
print('EFECTOS NO LINEALES - NLE')
print(np.round(b2,3))
print(' ')

print('VECTOR FUERZA/CORIOLIS - NLE')
print(np.round(b2-g,3))
print(' ')

# Parte 2: Verificacion de valores

error_b2=b2-c-g
print('VERIFICACION DE B2')
print(np.round(error_b2,3))
if np.linalg.norm(error_b2)<(0.001):
	print('VALORES DE B2 SON IGUALES')
else:
	print('VALORES DE B2 NO SON IGUALES')
print(' ')

error_M2 = M2 -M
print('VERIFICACION DE M2')
print(np.round(error_M2,3))
if np.linalg.norm(error_M2)<(0.001):
	print('VALORES DE M2 SON IGUALES')
else:
	print('VALORES DE M2 NO SON IGUALES')
print(' ')


# Parte 3: Verificacion de la expresion de la dinamica
print('TAU HALLADO AL PRINCIPIO')
print(np.round(tau,3))
tau2 = M.dot(ddq)+c+g ################################# ERROR
print('TAU HALLADO CON LA ECUACION 1')
print(np.round(tau2,3))
print(' ')

if np.linalg.norm(tau2-tau)<(0.01):
	print('CUMPLE LA ECUACION DINAMICA')
else:
	print('NO CUMPLE LA ECUACION DINAMICA')
print(' ')

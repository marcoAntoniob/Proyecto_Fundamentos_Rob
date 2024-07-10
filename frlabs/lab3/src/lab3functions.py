import numpy as np
from copy import copy

cos=np.cos; sin=np.sin; pi=np.pi


def dh(d, theta, a, alpha):
    """
    Calcular la matriz de transformacion homogenea asociada con los parametros
    de Denavit-Hartenberg.
    Los valores d, theta, a, alpha son escalares.
    """
    # Escriba aqui la matriz de transformacion homogenea en funcion de los valores de d, theta, a, alpha
    T = np.array([[cos(theta),-cos(alpha)*sin(theta),sin(alpha)*sin(theta),a*cos(theta)],
                  [sin(theta),cos(alpha)*cos(theta),-sin(alpha)*cos(theta),a*sin(theta)],
                  [0,sin(alpha),cos(alpha),d],
                  [0,0,0,1]])

    return T
    
    

def fkine_es200(q):
    """
    Calcular la cinematica directa del robot UR5 dados sus valores articulares. 
    q es un vector numpy de la forma [q1, q2, q3, q4, q5, q6]
    """
    # Longitudes (en metros)
    # Matrices DH (completar), emplear la funcion dh con los parametros DH para cada articulacion
    T0 =dh(0,0,0.395,-pi/2)
    T1 =dh(-q[0]+0.562,0,0,pi/2)
    T2 =dh(0.76403,q[1],0.73999,pi/2)
    T3 =dh(0,q[2]+pi/2,1.145,0)
    T4=dh(0,-q[3],0,pi/2)
    T5=dh(0,0,0.24989,0)
    T6 =dh(1.226,q[4],0,-pi/2)
    T7 =dh(0.034,0,0,0)
    T8 =dh(0,q[5],0,pi/2)

    T9 =dh(0.24996,q[6]+pi/2,0,pi/2)

    T =T0.dot(T1).dot(T2).dot(T3).dot(T4).dot(T5).dot(T6).dot(T7).dot(T8).dot(T9)
    return T




def jacobian(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion. Retorna una matriz de 3x6 y toma como
    entrada el vector de configuracion articular q=[q1, q2, q3, q4, q5, q6]
    """
    # Crear una matriz 3x6
    J = np.zeros((3, 7))
    # Transformacion homogenea inicial (usando q)
    T = fkine_es200(q)


    # Iteracion para la derivada de cada columna
    for i in range(7):
        # Copiar la configuracion articular inicial
        dq = copy(q)
        T = fkine_es200(dq)
        # Incrementar la articulacion i-esima usando un delta
        dq[i] = dq[i] + delta
        # Transformacion homogenea luego del incremento (q+delta)
        dT = fkine_es200(dq)

        T_inc = fkine_es200(dq)
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        if (i==1 or i==6):
            J[0:3,i]=(T_inc[0:3, 3]-T[0:3, 3])*(0.01)/delta
        else:
            J[0:3,i]=(T_inc[0:3, 3]-T[0:3, 3])/delta
    return J


def ikine_es200(xdes, q0):
    """
    Calcular la cinematica inversa de ES200 numericamente a partir de la configuracion articular inicial de q0. 
    Emplear el metodo de newton
    """
    epsilon = 0.001
    max_iter = 1000
    delta    = 0.00001
    q = copy(q0)
    errors = []

    for i in range(max_iter):
        J = jacobian(q)
        f = fkine_es200(q)[0:3, 3]
        e = xdes - f
        errors.append(np.linalg.norm(e))
        q = q + np.dot(np.linalg.pinv(J), e)
        if np.linalg.norm(e) < epsilon:
            break

    return q, errors




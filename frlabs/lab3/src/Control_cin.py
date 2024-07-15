#!/usr/bin/env python3
import os
import rospy
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import JointState
import matplotlib
matplotlib.use('TkAgg')  # Asegurarse de que el backend correcto se usa

from markers import *
from lab3functions import *

if __name__ == '__main__':

    # Initialize the node
    rospy.init_node("testKineControlPosition")
    print('starting motion ... ')
    # Publisher: publish to the joint_states topic
    pub = rospy.Publisher('joint_states', JointState, queue_size=10)
    # Files for the logs
    fxcurrent = open("/tmp/xcurrent.txt", "w")                
    fxdesired = open("/tmp/xdesired.txt", "w")
    fq = open("/tmp/q.txt", "w")

    # Markers for the current and desired positions
    bmarker_current  = BallMarker(color['RED'])
    bmarker_desired = BallMarker(color['GREEN'])

    # Joint names
    jnames = ['base_link1', 'link1_link2', 'link2_link3','link3_link4', 'link4_link5', 'link5_link6','link6_link7']

    # Desired position
    xd = np.array([2.44, 1.6, 0.8])
    print("Posicion deseada: ")
    print(xd)
    # Initial configuration
    q0 = np.array([0.4, 0.2, 0.6, -0.1, -0.5, 0.0, 0.2])
    print("\nConfiguracion inicial: ")
    print(q0)
    # Resulting initial position (end effector with respect to the base link)
    T = fkine_es200(q0)
    x0 = T[0:3,3]
    print("Matriz de posicion y orientacion inicial: ")
    print(T)
    # Red marker shows the achieved position
    bmarker_current.xyz(x0)
    # Green marker shows the desired position
    bmarker_desired.xyz(xd)

    # Instance of the JointState message
    jstate = JointState()
    # Values of the message
    jstate.header.stamp = rospy.Time.now()
    jstate.name = jnames
    # Add the head joint value (with value 0) to the joints
    jstate.position = q0

    # Frequency (in Hz) and control period 
    freq = 200
    dt = 1.0/freq
    rate = rospy.Rate(freq)

    # Initial joint configuration
    q = copy(q0)
    k = 0.5
    count = 1
    # Main loop
    while not rospy.is_shutdown():
        # Current time (needed for ROS)
        jstate.header.stamp = rospy.Time.now()
        # Kinematic control law for position (complete here)
        # -----------------------------
        T = fkine_es200(q)  # Cinemática directa para obtener la posición actual del efector final
        x = T[0:3, 3]  # Posición actual
        e = x - xd  # Error de posición

        # Calcular la Jacobiana (matriz de derivadas) del robot en la posición actual
        J = jacobian(q)
        print("Jacobiano: ")
        print(J)
        # Verificar si la posición deseada se alcanzó
        if np.linalg.norm(e) < 0.001:
            print("Se llegó al punto deseado en {:.3} segundos".format(count*dt))
            break
        
        
        
        
        # Derivada del error
        de = -k*e
        # Variación de la configuración articular
        dq = np.linalg.pinv(J).dot(de)
        # Integración para obtener la nueva configuración articular
        q = q + dt*dq
        # Actualizar las articulaciones
        # -----------------------------
        count = count + 1 
        if(count > 100000):
            print('Max number of iterations reached')
            break
        
        # Log values                                                      
        fxcurrent.write(str(x[0])+' '+str(x[1]) +' '+str(x[2])+'\n')
        fxdesired.write(str(xd[0])+' '+str(xd[1])+' '+str(xd[2])+'\n')
        fq.write(str(q[0])+" "+str(q[1])+" "+str(q[2])+" "+str(q[3])+" "+
                 str(q[4])+" "+str(q[5])+" "+str(q[6])+"\n")
        
        # Publish the message
        jstate.position = q
        pub.publish(jstate)
        bmarker_desired.xyz(xd)
        bmarker_current.xyz(x)
        # Wait for the next iteration
        rate.sleep()

    

    print('ending motion ...')
    fxcurrent.close()
    fxdesired.close()
    fq.close()

    # Read data from log files
    qcurrent_data = np.loadtxt("/tmp/q.txt")
    xcurrent_data = np.loadtxt("/tmp/xcurrent.txt")
    xdesired_data = np.loadtxt("/tmp/xdesired.txt")
    
    # Ensure both data sets have the same length
    min_length = min(len(qcurrent_data), len(xdesired_data))
    qcurrent_data = qcurrent_data[:min_length]
    xcurrent_data = xcurrent_data[:min_length]
    xdesired_data = xdesired_data[:min_length]
    
    # Generate time vector assuming constant time step
    num_samples = qcurrent_data.shape[0]
    time = np.linspace(0, num_samples / freq, num_samples)

    # Plot all joint angles in one plot
    plt.figure(figsize=(10, 8))
    for i in range(qcurrent_data.shape[1]):
        plt.plot(time, qcurrent_data[:, i], label=f'q{i}')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [rad]')
    plt.legend()
    plt.title('Joint Angles vs Time')
    plt.tight_layout()
    plt.show()

    # Plot end effector position as function of time
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time, xcurrent_data[:, 0], label='Actual X')
    plt.plot(time, xdesired_data[:, 0], label='Desired X', linestyle='--')
    plt.xlabel('Time [s]')
    plt.ylabel('X Position [m]')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time, xcurrent_data[:, 1], label='Actual Y')
    plt.plot(time, xdesired_data[:, 1], label='Desired Y', linestyle='--')
    plt.xlabel('Time [s]')
    plt.ylabel('Y Position [m]')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time, xcurrent_data[:, 2], label='Actual Z')
    plt.plot(time, xdesired_data[:, 2], label='Desired Z', linestyle='--')
    plt.xlabel('Time [s]')
    plt.ylabel('Z Position [m]')
    plt.legend()

    plt.tight_layout()
    plt.show()

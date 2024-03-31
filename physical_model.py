import math
import torch
import numpy as np


# 定义常量
R = 8.314
F = 96485
L = 22  # 隔膜电导率参数，需要根据实际情况进行调整
A_e = 0.0025
w_e = 0.003  # 电极厚度，需要根据实际情况进行调整
V_e = A_e * w_e
w_c = 0.015  # 集流体厚度，需要根据实际情况进行调整
theta_c = 91000  # 集流体电导率，需要根据实际情况进行调整
w_m = 0.000125  # 离子交换膜厚度，需要根据实际情况进行调整
E_p0 = 1.004
E_n0 = -0.26
n_d = 2.5


def get_con(SoC, C_V0, C_Hp0, C_Hn0, C_H2Op0):
    C_2 = C_V0*SoC
    C_3 = C_V0*(1 - SoC)
    C_4 = C_V0*(1 - SoC)
    C_5 = C_V0*SoC
    C_Hn = C_Hn0 + C_V0*SoC
    C_Hp = C_Hp0 + C_V0*SoC
    C_H2Op = C_H2Op0 - (1 + n_d)*C_V0*SoC
    return C_2, C_3, C_4, C_5, C_Hn, C_Hp, C_H2Op


def E_con(T, I, Q, C_2, C_3, C_4, C_5):

    E_con_p = []
    E_con_n = []

    for i in range(len(I)):
        if I[i] > 0:
            E_con_p.append(-R*T[i]/F*(-I[i]/(1.43*math.pow(10, -4)*F*math.pow(Q[i]/A_e, 0.4)*C_4[i])))
            E_con_n.append(-R*T[i]/F*(-I[i]/(1.43*math.pow(10, -4)*F*math.pow(Q[i]/A_e, 0.4)*C_3[i])))
        

        if I[i] < 0:
            E_con_p.append(-R*T[i]/F*(-I[i]/(1.43*math.pow(10, -4)*F*math.pow(Q[i]/A_e, 0.4)*C_5[i])))
            E_con_n.append(-R*T[i]/F*(-I[i]/(1.43*math.pow(10, -4)*F*math.pow(Q[i]/A_e, 0.4)*C_2[i])))
    
    E_con_p = np.array(E_con_p)
    E_con_n = np.array(E_con_n)
    E_con_p = torch.tensor(E_con_p)
    E_con_n = torch.tensor(E_con_n)
    
    E_con = E_con_p + E_con_n
    
    return E_con


def E_act(T, I, S, k_p, k_n, C_2, C_3, C_4, C_5):
    
    E_act = []
    for i in range(len(I)):
        E_act.append(R*T[i]*2/F*(math.asinh(I[i]/S[i]/V_e/2/F/k_p[i]/math.sqrt(C_4[i]*C_5[i])) 
                                 + math.asinh(I[i]/S[i]/V_e/2/F/k_n[i]/math.sqrt(C_2[i]*C_3[i]))))
    

    E_act = np.array(E_act)
    E_act = torch.tensor(E_act)
    return E_act


def E_ohm(theta_e, T, I):
    theta_m = []
    E_ohm = []
    for i in range(len(I)):
        theta_m.append((0.5139*L - 0.326)*math.exp(1268*(1/303 - 1/T[i])))

    
    theta_m = np.array(theta_m)
    theta_m = torch.tensor(theta_m)
        
    for i in range(len(I)):
        E_ohm.append((2*w_c/theta_c + 2*w_e/theta_e[i] + w_m/theta_m[i])*I[i]/A_e)
    

    E_ohm = np.array(E_ohm)
    E_ohm = torch.tensor(E_ohm)

    return E_ohm


def E_ocv(T, C_2, C_3, C_4, C_5, C_Hp, C_Hn, C_H2Op):

    E_ocv = []
    for i in range(len(T)):
        E_ocv.append(E_p0 - E_n0 + R*T[i]/F*math.log((C_2[i]*C_5[i]*C_Hp[i]*math.pow(C_Hp[i], 2))/(C_3[i]*C_4[i]*C_Hn[i]*C_H2Op[i])))

    
    E_ocv = np.array(E_ocv)
    E_ocv = torch.tensor(E_ocv)

    return E_ocv

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
C_Hp0 = 3850
C_Hn0 = 3030
C_H2Op0 = 44600


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
    E_con_p = torch.empty(len(I))
    E_con_n = torch.empty(len(I))

    for i in range(len(I)):
        E_con_p[i] = -R*T[i]/F*torch.log(1-I[i]/(1.43*math.pow(10, -4)*F*math.pow(Q[i]/A_e, 0.4)*C_4[i]))
        E_con_n[i] = -R*T[i]/F*torch.log(1-I[i]/(1.43*math.pow(10, -4)*F*math.pow(Q[i]/A_e, 0.4)*C_3[i]))


    E_con = E_con_p + E_con_n
    
    return E_con


def E_act(T, I, S, k_p, k_n, C_2, C_3, C_4, C_5):

    E_act = R*T*2/F*(torch.asinh(I/S/V_e/2/F/k_p/torch.sqrt(C_4*C_5)) + torch.asinh(I/S/V_e/2/F/k_n/torch.sqrt(C_2*C_3)))

    return E_act


def E_ohm(theta_e, T, I):
    
    theta_m = (0.5139*L - 0.326)*torch.exp(1268*(1/303 - 1/T))

    E_ohm = (2*w_c/theta_c + 2*w_e/theta_e + w_m/theta_m)*I/A_e

    return E_ohm


def E_ocv(T, C_2, C_3, C_4, C_5, C_Hp, C_Hn, C_H2Op):
    E_ocv = E_p0 - E_n0 + R*T/F*torch.log((C_2*C_5*C_Hp*torch.pow(C_Hp, 2))/(C_3*C_4*C_Hn*C_H2Op))

    return E_ocv

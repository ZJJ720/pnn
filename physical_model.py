import math


# 定义常量
R = 8.314
F = 96485
L = 22
A_e = 0.0025
w_e = 0.003
V_e = A_e*w_e
w_c = 0.015
theta_c = 0.00091
w_m = 0.000125
E_p0 = 1.004
E_n0 = -0.26


# 定义物理模型类
class PhysicalModel():
    def E_con(T, I, Q, C_2, C_3, C_4, C_5):
        if I > 0:
            E_con_p = -R*T/F*math.log(1-I/(1.43*math.pow(10, -4)*F*math.pow(Q/A_e, 0.4)*C_4))
            E_con_n = -R*T/F*math.log(1-I/(1.43*math.pow(10, -4)*F*math.pow(Q/A_e, 0.4)*C_3))
        

        if I < 0:
            E_con_p = -R*T/F*math.log(1-I/(1.43*math.pow(10, -4)*F*math.pow(Q/A_e, 0.4)*C_5))
            E_con_n = -R*T/F*math.log(1-I/(1.43*math.pow(10, -4)*F*math.pow(Q/A_e, 0.4)*C_2))
        
        
        E_con = E_con_p + E_con_n
        
        return E_con
    

    def E_act(T, I, S, k_p, k_n, C_2, C_3, C_4, C_5):
        E_act = R*T*2/F*(math.asinh(I/S/V_e/2/F/k_p/math.sqrt(C_4*C_5)) + math.asinh(I/S/V_e/2/F/k_n/math.sqrt(C_2*C_3)))
        
        return E_act
    

    def E_ohm(theta_e, T, I):
        theta_m = (0.5139*L - 0.326)*math.exp(1268*(1/303 - 1/T))
        E_ohm = (2*w_c/theta_c + 2*w_e/theta_e + w_m/theta_m)*I/A_e
        
        return E_ohm
    

    def E_ocv(T, C_2, C_3, C_4, C_5, C_Hp, C_Hn, C_H2Op):
        E_ocv = E_p0 - E_n0 + R*T/F*math.log((C_2*C_5*C_Hp*math.pow(C_Hp, 2))/(C_3*C_4*C_Hn*C_H2Op))

        return E_ocv 

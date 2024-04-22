# Physical-Constrained Neural Network for Redox Flow Battery Modeling

基于电化学模型，考虑欧姆极化 E_ohm 、浓差极化 E_con 、电化学极化 E_act 的物理约束，构建全连接神经网络，对全钒液流电池的参数进行估计。

## 参考

1. Physics-constrained deep neural network method for estimating parameters in a redox flow battery（整体架构参考）
2. Dynamic Flow Rate Control for Vanadium Redox Flow Batteries（浓差极化公式以及对应物理常量参考）
3. Design of A Two-Stage Control Strategy of Vanadium Redox Flow Battery Energy Storage Systems for Grid Application（浓差极化公式参考）

## 物理模型

需要根据实际电池参数，补充物理常量：

- C_Hp0
- C_Hn0
- C_H2Op0
- S0
- k_p0
- k_n0
- theta_e0

模型输入量：T(t), I(t), SoC(t), Q, C_V0

## DNN

模型输入量：T(t), I(t), SoC(t), Q, C_V0
模型输出量：k_n, k_p, S, theta_e
模型标签量：U(t)

## 训练

训练数据集：第三个循环后的放电数据集

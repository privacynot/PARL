# 方法设计

## 1. 方法概述

本研究面向大语言模型强化学习中的奖励模型优化问题，构建策略模型与奖励模型的双层优化框架，并研究稳定、计算成本可控的隐式超梯度估计方法。

在该框架中：

- PPO 负责优化大语言模型策略；
- 奖励模型负责评价模型生成结果；
- 外层根据策略更新后的真实表现优化奖励模型；
- 稳定超梯度方法用于提高奖励模型更新的可靠性。

研究重点不是重新提出双层优化，而是改进大模型奖励学习中的超梯度求解过程。

## 2. 双层优化建模

设策略模型参数为 $\theta$，奖励模型参数为 $\phi$。

内层使用奖励模型提供的奖励，通过 PPO 更新策略模型：

$$
\theta^*(\phi)=\arg\min_\theta\mathcal{L}_{\mathrm{inner}}(\theta,\phi)
$$

外层根据更新后策略的真实任务表现优化奖励模型：

$$
\min_\phi\mathcal{L}_{\mathrm{outer}}\left(\theta^*(\phi),\phi\right)
$$

外层损失可以包含任务损失、奖励监督损失和正则化损失：

$$
\mathcal{L}_{\mathrm{outer}}=\alpha\mathcal{L}_{\mathrm{task}}+\beta\mathcal{L}_{\mathrm{reward}}+\gamma\mathcal{L}_{\mathrm{reg}}
$$

其中：

- $\mathcal{L}_{\mathrm{task}}$ 由外部大模型评分得到的策略模型的真实任务表现；
- $\mathcal{L}_{\mathrm{reward}}$ 奖励模型的打分和外部大模型打分的接近程度，奖励模型给出的分数与外部大模型评分越接近，损失越小；
- $\mathcal{L}_{\mathrm{reg}}$ 用于约束奖励尺度和模型复杂度。

## 3. 外部评价的梯度估计

答案正确率、人工评分通常不能直接对大语言模型的生成过程反向传播。

设外部任务评分为 $S(x,y)$，期望任务得分为：

$$
J_{\mathrm{task}}(\theta)=\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}\left[S(x,y)\right]
$$

其策略梯度可以估计为：

$$
\nabla_\theta J_{\mathrm{task}}(\theta)=\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}\left[\left(S(x,y)-b(x)\right)\nabla_\theta\log\pi_\theta(y\mid x)\right]
$$

其中，$b(x)$ 是用于降低梯度估计方差的基线。外部评分器本身不需要可微。

## 4. 隐式超梯度

设内层损失关于策略参数的 Hessian 为：

$$
H=\nabla_{\theta\theta}^{2}\mathcal{L}_{\mathrm{inner}}
$$

奖励模型的超梯度可以表示为：

$$
\nabla_\phi\mathcal{L}_{\mathrm{outer}}=\frac{\partial\mathcal{L}_{\mathrm{outer}}}{\partial\phi}-\left(\nabla_{\phi\theta}^{2}\mathcal{L}_{\mathrm{inner}}\right)H^{-1}\nabla_\theta\mathcal{L}_{\mathrm{outer}}
$$

令：

$$
v=\nabla_\theta\mathcal{L}_{\mathrm{outer}}
$$

则需要近似求解线性系统：

$$
Hx=v
$$

得到：

$$
x\approx H^{-1}v
$$

随后计算隐式梯度：

$$
g_{\mathrm{implicit}}=-\left(\nabla_{\phi\theta}^{2}\mathcal{L}_{\mathrm{inner}}\right)x
$$

由于大语言模型参数量较大，本研究使用 Hessian 向量积和迭代求解器，避免显式构造完整 Hessian。

## 5. 稳定超梯度求解方法

### 5.1 阻尼与曲率保护

将原始线性系统修改为：

$$
(H+\lambda I)x=v
$$

其中，$\lambda>0$ 是阻尼系数。

阻尼用于降低小特征值和非正曲率导致的梯度放大风险。求解过程中还需要检测非正曲率、非有限数值和残差异常。

阻尼会引入近似偏差，因此需要同时评价稳定性和超梯度误差。

### 5.2 对角预条件

构造正的对角预条件器 $M$：

$$
M^{-1}(H+\lambda I)x=M^{-1}v
$$

预条件信息可以来自 Adam 二阶矩、梯度平方滑动平均或 Fisher 对角近似。

预条件器的目标是在相同计算预算下减少 Hessian 向量积次数，提高求解效率。

### 5.3 自适应停止

设第 $k$ 次迭代的残差为：

$$
r_k=v-(H+\lambda I)x_k
$$

定义相对残差：

$$
\rho_k=\frac{\left\|r_k\right\|_2}{\left\|v\right\|_2+\varepsilon}
$$

满足以下任一条件时停止迭代：

1. 相对残差低于目标阈值；
2. 达到最大 Hessian 向量积次数；
3. 残差连续多次不再下降；
4. 检测到非正曲率；
5. 检测到非有限数值。

## 6. 奖励模型更新

奖励模型的最终梯度由监督梯度、隐式梯度和正则化梯度组成：

$$
g_\phi=\alpha g_{\mathrm{supervised}}+\beta g_{\mathrm{implicit}}+\gamma g_{\mathrm{regularization}}
$$

其中：

- $g_{\mathrm{supervised}}$ 用于拟合外部评分或偏好；
- $g_{\mathrm{implicit}}$ 表示奖励变化对策略学习结果的影响；
- $g_{\mathrm{regularization}}$ 用于约束奖励模型。

最终使用 Adam 等一阶优化器更新奖励模型参数。

## 7. 整体训练流程

1. 大语言模型生成候选回答或决策方案；
2. 奖励模型对生成结果进行评分；
3. PPO 根据奖励信号更新策略模型；
4. 在独立外层数据上评价更新后的策略；
5. 计算外层目标对策略参数的梯度；
6. 使用稳定求解器估计逆 Hessian 向量积；
7. 计算奖励模型的直接梯度和隐式梯度；
8. 更新奖励模型参数；
9. 记录任务表现、求解残差、计算时间和显存占用。

# 问题定义

## 符号说明

| 符号 | 含义 |
|---|---|
| \(x\) | 模型输入或任务问题 |
| \(y\) | 大语言模型生成结果 |
| \(\theta\) | 策略模型参数 |
| \(\phi\) | 奖励模型参数 |
| \(\pi_\theta(y\mid x)\) | 策略模型的生成概率 |
| \(R_\phi(x,y)\) | 奖励模型输出 |
| \(S(x,y)\) | 外部任务评分 |
| \(\mathcal{L}_{\mathrm{inner}}\) | 内层策略优化损失 |
| \(\mathcal{L}_{\mathrm{outer}}\) | 外层奖励优化损失 |

## 内层策略优化

奖励模型对生成结果进行评分，PPO 根据奖励信号更新策略模型：

\[
\theta^*(\phi)
=
\arg\min_\theta
\mathcal{L}_{\mathrm{inner}}(\theta,\phi)
\]

实际训练中不会将内层问题精确求解到最优，而是执行有限次 PPO 更新。因此，\(\theta^*(\phi)\) 是对内层最优解的近似。

## 外层奖励模型优化

外层目标用于评价更新后的策略是否获得更好的真实任务表现：

\[
\min_\phi
\mathcal{L}_{\mathrm{outer}}
\left(
\theta^*(\phi),\phi
\right)
\]

外层目标可以同时包含任务损失和奖励模型监督损失：

\[
\mathcal{L}_{\mathrm{outer}}
=
\alpha\mathcal{L}_{\mathrm{task}}
+
\beta\mathcal{L}_{\mathrm{reward}}
+
\gamma\mathcal{L}_{\mathrm{reg}}
\]

其中，\(\alpha\)、\(\beta\) 和 \(\gamma\) 为权重系数。


## 隐式超梯度
奖励模型会影响 PPO 使用的奖励，PPO 的更新又会改变策略模型。因此，调整奖励模型时，不能只看奖励模型自身的监督损失，还需要考虑：

> 奖励模型发生变化以后，策略模型会怎样变化，以及这种变化会不会提升最终任务表现。

这种同时考虑直接影响和间接影响的梯度，称为**超梯度**。

当内层训练近似满足驻点条件时，奖励模型的隐式超梯度可以写为：

\[
\nabla_\phi\mathcal{L}_{\mathrm{outer}}
=
\frac{\partial\mathcal{L}_{\mathrm{outer}}}{\partial\phi}
-
\left(
\frac{\partial^2\mathcal{L}_{\mathrm{inner}}}
{\partial\phi\,\partial\theta}
\right)
H^{-1}
\nabla_\theta\mathcal{L}_{\mathrm{outer}}
\]

其中：

\[
H
=
\nabla^2_{\theta\theta}
\mathcal{L}_{\mathrm{inner}}
\]

由于无法显式构造大语言模型的 Hessian，本研究通过 Hessian 向量积和迭代线性求解器近似计算 \(H^{-1}v\)。

## 研究问题

1. 如何验证计算得到的超梯度确实对应所定义的双层目标？
2. 如何降低随机曲率估计对超梯度稳定性的影响？
3. 如何在有限 Hessian 向量积预算下提高求解效率？
4. 超梯度近似质量是否能够转化为奖励模型和策略模型的性能提升？

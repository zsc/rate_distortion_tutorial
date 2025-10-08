# 第八章：深度学习中的率失真

本章探索深度学习如何重新诠释率失真理论，从变分自编码器（VAE）到神经压缩、信息瓶颈理论。我们将看到深度学习不仅实现了率失真编码，还拓展了理论边界，在感知质量和泛化能力上超越传统方法。

**学习目标**：
- 理解VAE的率失真解释
- 掌握信息瓶颈理论的核心思想
- 了解神经压缩的端到端优化
- 建立从字典学习到深度学习的演进脉络

---

## 8.1 从字典学习到深度学习

### 8.1.1 非线性字典

**字典学习的局限**：线性表示$\mathbf{x} \approx \mathbf{D}\mathbf{s}$

**深度网络**：非线性表示$\mathbf{x} \approx f_{\theta}(\mathbf{z})$

其中$f_{\theta}$是深度网络（解码器），$\mathbf{z}$是隐变量（"码字"）。

**优势**：
- 表达能力更强（通用逼近定理）
- 层次化特征（低层→边缘，高层→语义）
- 端到端学习

**为什么非线性至关重要？**

考虑表示复杂数据（如人脸图像）：

**线性字典**（如字典学习、PCA）：
- 表示空间是线性子空间的并集
- 只能捕捉数据的线性流形结构
- 例子：不同角度的人脸形成不同的线性子空间，字典学习学习这些子空间的基

**深度网络**：
- 可以表示高度非线性的流形
- 通过多层非线性变换，逐步将复杂流形"拉直"
- 例子：人脸的姿态、光照、表情变化形成复杂流形，深度网络可以学习这个流形的紧凑表示

**数学视角**：

假设数据分布在 $d$ 维流形 $\mathcal{M} \subset \mathbb{R}^n$（$d \ll n$）。

- **线性方法**：寻找 $d$ 维线性子空间 $V$ 使 $\mathcal{M}$ 到 $V$ 的投影误差最小
  - PCA：最优线性子空间
  - 字典学习：多个线性子空间的并集
  - 局限：如果 $\mathcal{M}$ 是非线性的（如球面），线性逼近效率低

- **深度网络**：学习非线性映射 $f: \mathbb{R}^d \to \mathbb{R}^n$ 使 $\mathcal{M} \approx f(\mathbb{R}^d)$
  - 可以精确表示任意光滑流形（给定足够容量）
  - 层次化：逐层学习越来越抽象的特征

**层次化特征的例子**（图像网络）：

| 层次 | 特征 | 维度 | 例子 |
|-----|-----|-----|------|
| 第1层 | 边缘、角点 | 低级 | Gabor滤波器 |
| 第2-3层 | 纹理、简单形状 | 中级 | 眼睛、鼻子部件 |
| 第4-5层 | 物体部件 | 高级 | 人脸、车辆轮廓 |
| 顶层 | 语义概念 | 抽象 | 人脸、汽车、猫 |

这种层次化在字典学习中无法自然实现（字典是平坦的，所有原子地位相同）。

**对比**：

| 方法 | 表示 | 编码器 | 解码器 | 学习 |
|:---|:---:|:---:|:---:|:---:|
| 字典学习 | $\mathbf{x} \approx \mathbf{D}\mathbf{s}$ | 优化（OMP等） | 线性$\mathbf{D}$ | 交替 |
| 自编码器 | $\mathbf{x} \approx f_{\text{dec}}(\mathbf{z})$ | 神经网络$f_{\text{enc}}$ | 神经网络$f_{\text{dec}}$ | 端到端BP |

**具体数值例子**：

对于MNIST手写数字（28×28=784维）：
- **PCA**：需要约50个主成分才能保留95%方差
- **字典学习**（256原子）：稀疏度约5-10，有效维度仍约50
- **自编码器**（瓶颈维度20）：可以达到相似重建质量，但维度仅20
  - 原因：非线性变换更高效地捕捉数据流形

**Rule of thumb**：对于高度结构化的数据（图像、语音、自然语言），深度网络的非线性表示能力远超线性方法。字典学习适合小规模、局部特征提取；深度学习适合大规模、端到端任务。

### 8.1.2 自编码器（Autoencoder）

**标准自编码器**：

```
输入 x → 编码器 → 隐层 z → 解码器 → 重建 x̂
          f_enc           f_dec
```

**目标**：最小化重建误差

$$\min_{\theta_{\text{enc}}, \theta_{\text{dec}}} \mathbb{E}[\|\mathbf{x} - f_{\text{dec}}(f_{\text{enc}}(\mathbf{x}))\|^2]$$

**率失真视角**：
- 隐层维度$d_z$限制"码率"（瓶颈）
- 重建误差是"失真"
- 但缺少显式的率约束（$d_z$是固定的，不是优化变量）

**自编码器的架构设计**：

典型的图像自编码器（如用于MNIST）：

**编码器**：
```
输入 28×28×1 (784维)
  ↓ Conv 16滤波器, 3×3, stride=2
 14×14×16 (3136维)
  ↓ ReLU
  ↓ Conv 32滤波器, 3×3, stride=2
  7×7×32 (1568维)
  ↓ ReLU
  ↓ 全连接层
隐层 z: 32维
```

**解码器**（对称结构）：
```
隐层 z: 32维
  ↓ 全连接层
  7×7×32
  ↓ ReLU
  ↓ 反卷积 16滤波器, 3×3, stride=2
 14×14×16
  ↓ ReLU
  ↓ 反卷积 1滤波器, 3×3, stride=2
输出 28×28×1
```

**参数分析**：
- 输入维度：784
- 隐层维度：32
- 压缩比：$784 / 32 = 24.5$ 倍（理论上）
- 实际压缩比：更低，因为 $\mathbf{z}$ 仍需编码

**与PCA的对比**：

| 特性 | PCA | 自编码器 |
|-----|-----|---------|
| 映射 | 线性 | 非线性 |
| 最优性 | MSE意义下最优（对高斯） | 无理论保证，但实践中更好 |
| 训练 | 特征分解（一次） | 梯度下降（迭代） |
| 表达能力 | 有限（线性子空间） | 强（任意流形） |
| 可解释性 | 主成分有明确意义 | 隐层较难解释 |

**为什么自编码器不直接用于压缩？**

虽然自编码器学习紧凑表示 $\mathbf{z}$，但要真正压缩数据还需要：

1. **离散化**：$\mathbf{z}$ 通常是连续值，需要量化
   - 量化误差会影响重建质量
   - 需要权衡量化精度vs码率

2. **熵编码**：量化后的 $\mathbf{z}$ 需要无损编码
   - 如果 $\mathbf{z}$ 的分量不是均匀分布，可以用熵编码节省码率
   - 标准自编码器不建模 $p(\mathbf{z})$，无法优化熵

3. **联合优化**：需要同时优化重建质量和编码效率
   - 标准自编码器只优化重建，不考虑 $\mathbf{z}$ 的可压缩性

**从自编码器到神经压缩的演进**：

| 方法 | 优化目标 | 编码 | 应用 |
|-----|---------|------|------|
| 标准AE | $\min \|\mathbf{x} - \hat{\mathbf{x}}\|^2$ | 无 | 特征学习 |
| 稀疏AE | $\min \|\mathbf{x} - \hat{\mathbf{x}}\|^2 + \lambda \\|\mathbf{z}\\|_1$ | 无 | 稀疏表示 |
| VAE | $\max \text{ELBO}$ | 概率模型 | 生成+压缩 |
| 神经压缩 | $\min D + \lambda R$ | 熵编码 | 图像/视频压缩 |

**稀疏自编码器**：

引入稀疏性约束，更接近率失真思想：

$$\min \|\mathbf{x} - \hat{\mathbf{x}}\|^2 + \lambda \|\mathbf{z}\|_1$$

- $L_1$ 惩罚促进稀疏性
- 稀疏的 $\mathbf{z}$ 更易压缩（类似第七章的稀疏编码）
- 但仍缺少显式的熵模型

**Rule of thumb**：标准自编码器主要用于特征学习、降维，不直接用于压缩（缺少熵编码、概率模型）。要用于压缩，需要扩展到VAE（变分自编码器）或显式的神经压缩框架（section 8.4）。

---

## 8.2 变分自编码器（VAE）

### 8.2.1 VAE的概率框架

**生成模型**：假设数据$\mathbf{x}$由隐变量$\mathbf{z}$生成

$$p_{\theta}(\mathbf{x}) = \int p_{\theta}(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) d\mathbf{z}$$

其中$p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$（先验），$p_{\theta}(\mathbf{x}|\mathbf{z})$由解码器神经网络参数化。

**推断**：给定$\mathbf{x}$，推断后验$p(\mathbf{z}|\mathbf{x})$（困难，无闭式解）

**变分推断**：用参数化的$q_{\phi}(\mathbf{z}|\mathbf{x})$（编码器）近似$p(\mathbf{z}|\mathbf{x})$

### 8.2.2 ELBO与率失真

VAE优化**ELBO**（Evidence Lower Bound）：

$$\mathcal{L} = \mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}[\log p_{\theta}(\mathbf{x}|\mathbf{z})] - D_{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

**率失真解释**：

将ELBO重写为率失真形式（假设高斯解码器 $p_{\theta}(\mathbf{x}|\mathbf{z}) = \mathcal{N}(\mathbf{x}; f_{\theta}(\mathbf{z}), \sigma^2 \mathbf{I})$）：

$$\mathcal{L} = \underbrace{-\mathbb{E}[\|\mathbf{x} - \hat{\mathbf{x}}\|^2]}_{\text{重建项（负失真）}} - \beta \underbrace{D_{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{KL项（率）}}$$

其中 $\beta = 2\sigma^2$（对于高斯解码器）。

**深层理解：VAE是率失真编码器**

VAE的ELBO完全对应率失真理论的拉格朗日形式：

| 率失真理论 | VAE |
|:---:|:---:|
| 最小化：$I(X;\hat{X}) + \beta D$ | 最大化：$-D_{KL}(q\|\|p) + \text{const} \cdot \text{重建}$ |
| 编码分布：$p(\hat{x}\|x)$ | 编码分布：$q_\phi(\mathbf{z}\|\mathbf{x})$ |
| 先验：$p(\hat{x})$ | 先验：$p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ |
| 解码：重建$\hat{X}$ | 解码：$p_\theta(\mathbf{x}\|\mathbf{z})$ |

**关键等价性**：

KL散度项 $D_{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$ 对应**率**（传输隐变量 $\mathbf{z}$ 所需的信息量）。

**证明直觉**：

给定 $\mathbf{x}$，编码器产生分布 $q_{\phi}(\mathbf{z}|\mathbf{x})$。如果先验是 $p(\mathbf{z})$，则传输 $\mathbf{z}$ 所需的额外比特数（相对于先验）正是KL散度：

$$I(\mathbf{X}; \mathbf{Z}) = \mathbb{E}_{\mathbf{x}}[D_{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))]$$

这个互信息就是"率"。

**重建项**对应**负失真**：
$$\mathbb{E}_{q_{\phi}}[\log p_{\theta}(\mathbf{x}|\mathbf{z})] \approx -\frac{1}{2\sigma^2} \mathbb{E}[\|\mathbf{x} - f_{\theta}(\mathbf{z})\|^2] + \text{const}$$

因此，VAE的ELBO可以写成：

$$\mathcal{L} = -\left[\text{失真} + \frac{1}{\beta} \cdot \text{率}\right]$$

最大化ELBO等价于最小化率失真代价！

**$\beta$-VAE：显式控制率失真权衡**

标准VAE通常固定 $\beta=1$。**$\beta$-VAE**显式引入权衡参数：

$$\mathcal{L}_{\beta} = \mathbb{E}[\log p_{\theta}(\mathbf{x}|\mathbf{z})] - \beta \cdot D_{KL}(q_{\phi} \| p)$$

- $\beta < 1$：更关心重建质量（低失真），容忍高码率 → 生成细节丰富但码率高
- $\beta > 1$：更关心压缩隐变量（低码率），容忍高失真 → 隐变量更独立、更解耦，但重建可能模糊
- $\beta = 1$：平衡点（标准VAE）

**实际应用**：

- **压缩**：$\beta$ 大，强制隐变量接近先验 → 易压缩
- **生成**：$\beta$ 小，保留更多信息 → 高质量生成
- **表示学习**：$\beta$ 大，隐变量解耦 → 可解释特征

**数值例子**：

考虑MNIST图像（28×28 = 784维），隐变量维度 $d_z = 20$。

- **标准VAE**（$\beta=1$）：
  - KL项：约10-15 nats ≈ 14-22 比特
  - 重建PSNR：约25 dB
  - 总"有效码率"：约20-30比特/图像

- **$\beta$-VAE**（$\beta=4$）：
  - KL项：约3-5 nats ≈ 4-7 比特（更接近先验）
  - 重建PSNR：约20 dB（质量下降）
  - 隐变量更解耦（如单个维度对应旋转、厚度等）

**VAE vs 传统压缩**：

- **优势**：学习的表示，端到端优化，可生成新样本
- **劣势**：对于已知信号（如特定图像），传统方法（JPEG）仍更高效
- **适用场景**：未知分布的数据、需要生成能力、特征学习

**与率失真理论的差异**：

VAE的"率"是 $I(\mathbf{X}; \mathbf{Z})$，而传统率失真的"率"是 $I(X; \hat{X})$。两者区别：
- VAE：隐变量 $\mathbf{Z}$ 是低维瓶颈
- 率失真：重建 $\hat{X}$ 与原始 $X$ 同维

但在压缩应用中，需要额外的熵编码步骤将 $\mathbf{Z}$ 编码为比特流，此时总码率为 $H(\mathbf{Z}|model)$，接近 $I(\mathbf{X};\mathbf{Z})$（对于良好训练的模型）。

（假设$p_{\theta}(\mathbf{x}|\mathbf{z}) = \mathcal{N}(\mathbf{x}; f_{\text{dec}}(\mathbf{z}), \sigma^2 \mathbf{I})$）

**对应关系**：
- **重建项**：对应失真$D$（负号使得最大化ELBO = 最小化失真）
- **KL项**：对应率$R$，衡量$q(\mathbf{z}|\mathbf{x})$与先验$p(\mathbf{z})$的差异

**直觉**：
- KL小 → $\mathbf{z}$接近先验（易编码，低率）
- KL大 → $\mathbf{z}$远离先验（难编码，高率）

因此，VAE的ELBO本质上是率失真优化：

$$\max \text{ELBO} \equiv \min [D + \beta R]$$

**Rule of thumb**：VAE是率失真理论在深度生成模型中的自然体现。$\beta = 1$时对应标准VAE，调节$\beta$可以权衡率失真。

### 8.2.3 β-VAE

**β-VAE**显式引入权重$\beta$：

$$\mathcal{L}_{\beta} = \mathbb{E}[\log p_{\theta}(\mathbf{x}|\mathbf{z})] - \beta \cdot D_{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

**$\beta$的作用**：
- $\beta < 1$：偏向重建质量（低失真），KL可以大（高率）
- $\beta = 1$：标准VAE
- $\beta > 1$：偏向稀疏/解耦表示（低率），重建质量可以差（高失真）

**应用**：
- $\beta > 1$用于学习解耦表示（disentangled representations）
- $\beta < 1$用于高质量重建（接近感知质量）

**与率失真曲线的关系**：不同$\beta$对应率失真曲线上的不同工作点。

---

## 8.3 信息瓶颈理论

### 8.3.1 信息瓶颈问题

**设定**：输入$\mathbf{X}$，目标$\mathbf{Y}$，学习表示$\mathbf{Z}$

**目标**：$\mathbf{Z}$应该：
1. **压缩**：$I(\mathbf{X}; \mathbf{Z})$小（低复杂度）
2. **预测**：$I(\mathbf{Y}; \mathbf{Z})$大（高判别性）

**信息瓶颈（IB）**目标：

$$\max_{\mathbf{Z}} [I(\mathbf{Y}; \mathbf{Z}) - \beta I(\mathbf{X}; \mathbf{Z})]$$

或等价地（拉格朗日形式）：

$$\min_{\mathbf{Z}} [I(\mathbf{X}; \mathbf{Z}) - \frac{1}{\beta} I(\mathbf{Y}; \mathbf{Z})]$$

约束：$\mathbf{X} \to \mathbf{Z} \to \mathbf{Y}$（马尔可夫链）

**信息瓶颈的直观理解**：

想象你要总结一篇长文章（$\mathbf{X}$）来预测它的类别（$\mathbf{Y}$，如新闻/体育/娱乐）：

- **压缩原则**：摘要 $\mathbf{Z}$ 应该简洁，丢弃无关细节
  - $I(\mathbf{X}; \mathbf{Z})$ 小意味着摘要短，信息少
  - 极端情况：$\mathbf{Z}$ 是常数，$I(\mathbf{X}; \mathbf{Z}) = 0$，但完全无用

- **预测原则**：摘要 $\mathbf{Z}$ 应该保留判别信息
  - $I(\mathbf{Y}; \mathbf{Z})$ 大意味着从摘要能准确预测类别
  - 极端情况：$\mathbf{Z} = \mathbf{X}$（完整保留），$I(\mathbf{Y}; \mathbf{Z}) = I(\mathbf{Y}; \mathbf{X})$ 最大，但没有压缩

**权衡参数 $\beta$ 的作用**：

- **$\beta$ 大**：更关心压缩，$\mathbf{Z}$ 必须简洁
  - 结果：$\mathbf{Z}$ 维度低，可能损失一些预测精度
  - 应用：资源受限场景（嵌入式设备）

- **$\beta$ 小**：更关心预测，允许 $\mathbf{Z}$ 复杂
  - 结果：$\mathbf{Z}$ 维度高，预测准确但冗余
  - 应用：高精度任务（医疗诊断）

**马尔可夫约束的含义**：

$\mathbf{X} \to \mathbf{Z} \to \mathbf{Y}$ 意味着：
- $\mathbf{Z}$ 是从 $\mathbf{X}$ 计算的（编码器）
- $\mathbf{Y}$ 的预测只能基于 $\mathbf{Z}$，不能直接访问 $\mathbf{X}$
- 这确保了 $\mathbf{Z}$ 是"瓶颈"——所有信息必须经过它

**信息瓶颈平面（IB Plane）**：

在二维空间 $(I(\mathbf{X}; \mathbf{Z}), I(\mathbf{Y}; \mathbf{Z}))$ 中，可达的 $\mathbf{Z}$ 形成一条曲线：

```
I(Y;Z)
  ^
  |     理想曲线（IB最优）
  |       ╱
  |      ╱
  |     ╱
  |    ╱
  |   ╱
  |  ╱
  | ╱
  |╱___________________> I(X;Z)
  0
```

**关键性质**：
- 曲线是凸的（在某些条件下）
- 不同 $\beta$ 对应曲线上不同点
- 曲线左端：$\mathbf{Z}$ 几乎是常数，$I(\mathbf{X};\mathbf{Z}) \approx 0$, $I(\mathbf{Y};\mathbf{Z}) \approx 0$
- 曲线右端：$\mathbf{Z}$ 包含所有信息，$I(\mathbf{X};\mathbf{Z}) = H(\mathbf{X})$, $I(\mathbf{Y};\mathbf{Z}) = I(\mathbf{X};\mathbf{Y})$

**具体数值例子**（MNIST分类）：

- $\mathbf{X}$：28×28图像（784维）
- $\mathbf{Y}$：数字标签（0-9，10类）
- $\mathbf{Z}$：隐层表示

| $\beta$ | $\dim(\mathbf{Z})$ | $I(\mathbf{X};\mathbf{Z})$ (bits) | $I(\mathbf{Y};\mathbf{Z})$ (bits) | 测试准确率 |
|---------|---------------------|-----------------------------------|-----------------------------------|-----------|
| 0.001 | 256 | 180 | 3.3 | 99% |
| 0.01 | 64 | 120 | 3.2 | 98% |
| 0.1 | 32 | 60 | 3.0 | 96% |
| 1.0 | 10 | 20 | 2.5 | 90% |
| 10.0 | 5 | 8 | 1.8 | 75% |

观察：
- 随 $\beta$ 增大，$I(\mathbf{X};\mathbf{Z})$ 减小（更压缩）
- 同时 $I(\mathbf{Y};\mathbf{Z})$ 也减小（预测能力下降）
- 存在"甜点"：$\beta \approx 0.01-0.1$，平衡压缩与性能

### 8.3.2 与率失真的联系

**无监督情况**：$\mathbf{Y} = \mathbf{X}$（重建自己），信息瓶颈变为

$$\min_{\mathbf{Z}} [I(\mathbf{X}; \mathbf{Z}) - \frac{1}{\beta} I(\mathbf{X}; \mathbf{Z})] = \min [I(\mathbf{X}; \mathbf{Z})]$$

这退化为最小化互信息，但加上重建约束后：

$$\min_{p(\hat{\mathbf{x}}|\mathbf{z})} [I(\mathbf{X}; \mathbf{Z}) + \beta \mathbb{E}[d(\mathbf{X}, \hat{\mathbf{X}})]]$$

这正是**率失真问题**！

**有监督情况**：IB推广了率失真，将"重建自己"推广到"预测目标$\mathbf{Y}$"。

### 8.3.3 深度学习中的IB

**IB理论对深度学习的启示**：

1. **训练动态**：神经网络训练分两阶段
   - **拟合阶段**：增加$I(\mathbf{Y}; \mathbf{Z})$（降低训练误差）
   - **压缩阶段**：减少$I(\mathbf{X}; \mathbf{Z})$（泛化，忘记无关信息）

2. **泛化理解**：好的表示应该压缩输入、保留预测信息

3. **正则化**：Dropout、权重衰减等正则化技术可以理解为减少$I(\mathbf{X}; \mathbf{Z})$

**争议**：IB理论在深度学习中的适用性仍有争议（测量互信息困难、连续变量的定义）

**Rule of thumb**：信息瓶颈提供了一个优美的理论框架，但在实践中，直接优化IB目标（估计互信息）很困难。VAE、对比学习等方法可以看作IB的实用近似。

---

## 8.4 神经压缩

### 8.4.1 端到端学习的图像压缩

**传统方法**（JPEG、H.264）：手工设计的变换、量化、熵编码

**神经压缩**：端到端学习所有组件

```
输入x → 编码器 → 隐层y → 量化 → ŷ → 熵编码 → 比特流
                 f_enc         Q           E

比特流 → 熵解码 → ŷ → 解码器 → 重建x̂
              D          f_dec
```

### 8.4.2 核心组件

**1. 非线性变换**：编码器$f_{\text{enc}}$和解码器$f_{\text{dec}}$是卷积神经网络

$$\mathbf{y} = f_{\text{enc}}(\mathbf{x}), \quad \hat{\mathbf{x}} = f_{\text{dec}}(\hat{\mathbf{y}})$$

**2. 量化**：$\hat{\mathbf{y}} = Q(\mathbf{y}) = \text{round}(\mathbf{y})$

**挑战**：量化不可微，无法反向传播

**解决**：
- 训练时：用加噪声近似 $\hat{\mathbf{y}} = \mathbf{y} + \mathbf{u}$，$\mathbf{u} \sim \text{Uniform}(-0.5, 0.5)$（STE, Straight-Through Estimator）
- 测试时：真正量化 $\hat{\mathbf{y}} = \text{round}(\mathbf{y})$

**3. 熵编码**：对量化后的$\hat{\mathbf{y}}$，用学习的概率模型$p_{\hat{\mathbf{y}}}$进行算术编码

**码率估计**：

$$R = \mathbb{E}[-\log_2 p_{\hat{\mathbf{y}}}(\hat{\mathbf{y}})] = H(\hat{\mathbf{Y}})$$

### 8.4.3 率失真优化

**目标**：端到端最小化

$$\mathcal{L} = \mathbb{E}[\underbrace{d(\mathbf{x}, \hat{\mathbf{x}})}_{\text{失真}} + \lambda \underbrace{(-\log p_{\hat{\mathbf{y}}}(\hat{\mathbf{y}}))}_{\text{码率}}]$$

其中$d$可以是MSE、MS-SSIM、感知损失（LPIPS）等。

**训练**：
1. 前向：$\mathbf{x} \to f_{\text{enc}} \to \mathbf{y} \to$ 加噪 $\to \hat{\mathbf{y}} \to f_{\text{dec}} \to \hat{\mathbf{x}}$
2. 计算损失：$\mathcal{L} = d(\mathbf{x}, \hat{\mathbf{x}}) + \lambda H(\hat{\mathbf{y}})$
3. 反向传播，更新$f_{\text{enc}}, f_{\text{dec}}, p_{\hat{\mathbf{y}}}$

**$\lambda$扫描**：训练多个模型（不同$\lambda$），得到率失真曲线

### 8.4.4 性能对比

**神经压缩 vs 传统编码器**（实验数据）：

```
PSNR (dB)
  40 |           * 神经压缩
     |       *
  35 |   *         * VVC/H.266
     | *       *
  30 |     *       * HEVC/H.265
     |  *
  25 | *
     +-------------------→ 比特率 (bpp)
     0  0.1  0.2  0.3  0.4
```

**结论**：在MS-SSIM等感知度量上，神经压缩超越传统方法；在PSNR上，两者接近（神经压缩略优或相当）。

**优势**：
- 端到端优化，避免手工设计
- 可以针对感知质量优化
- 泛化到不同失真度量

**挑战**：
- 计算复杂度高（编码、解码都需要神经网络前向）
- 模型大（需要存储网络参数）
- 标准化困难（不同框架、硬件）

**Rule of thumb**：神经压缩目前主要用于研究和特定应用（如云端压缩），尚未广泛部署。但其性能优势和灵活性使其成为未来压缩标准的有力候选。

### 8.4.5 超先验（Hyperprior）模型

**问题**：$\mathbf{y}$的不同通道、空间位置高度相关，如果假设独立（$p(\hat{\mathbf{y}}) = \prod_i p(\hat{y}_i)$），会浪费编码效率。

**超先验**（Ballé et al., 2018）：引入额外隐层$\mathbf{z}$，建模$\mathbf{y}$的分布

```
x → f_enc → y → 量化 → ŷ
              ↓
         f_hyper_enc → z → 量化 → ẑ
```

**条件概率模型**：

$$p(\hat{\mathbf{y}}|\hat{\mathbf{z}}) = \prod_i p(\hat{y}_i | \hat{\mathbf{z}})$$

其中$p(\cdot|\hat{\mathbf{z}})$由另一个神经网络参数化。

**熵编码**：
1. 先编码$\hat{\mathbf{z}}$（用简单先验$p(\hat{\mathbf{z}})$）
2. 解码端恢复$\hat{\mathbf{z}}$
3. 用$\hat{\mathbf{z}}$作为上下文，编码$\hat{\mathbf{y}}$

**效果**：码率减少10-15%（相比独立假设）

---

## 8.5 对比：从字典学习到神经压缩

| 方法 | 字典学习 | VAE | 神经压缩 |
|:---|:---:|:---:|:---:|
| **表示** | 线性$\mathbf{D}\mathbf{s}$ | 概率$p(\mathbf{z}|\mathbf{x})$ | 确定$f_{\text{enc}}(\mathbf{x})$ |
| **编码器** | 优化（OMP） | 神经网络 | 神经网络 |
| **解码器** | 线性 | 神经网络 | 神经网络 |
| **率度量** | $\|\mathbf{s}\|_0$ | $D_{KL}(q\|p)$ | $H(\hat{\mathbf{y}})$ |
| **量化** | 隐式 | 隐式（reparameterization） | 显式（round） |
| **熵编码** | 需要额外设计 | 需要额外设计 | 端到端学习 |
| **应用** | 去噪、特征提取 | 生成、表示学习 | 图像/视频压缩 |

**演进**：字典学习→VAE→神经压缩，逐步从线性到非线性、从隐式到显式的率建模、从分离到端到端。

---

## 8.6 本章小结

**核心概念**：

1. **自编码器与字典学习**：
   - 非线性字典：$\mathbf{x} \approx f_{\text{dec}}(\mathbf{z})$
   - 瓶颈层限制"码率"

2. **VAE的率失真解释**：
   - ELBO = 重建项（负失真）+ KL项（率）
   - β-VAE：显式权衡率失真

3. **信息瓶颈理论**：
   - 压缩：最小化$I(\mathbf{X};\mathbf{Z})$
   - 预测：最大化$I(\mathbf{Y};\mathbf{Z})$
   - 无监督IB = 率失真

4. **神经压缩**：
   - 端到端学习：编码器、解码器、熵模型
   - 量化：STE技巧
   - 超先验：建模隐层分布

**关键公式**：

- VAE ELBO：$\mathcal{L} = \mathbb{E}[\log p(\mathbf{x}|\mathbf{z})] - \beta D_{KL}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$
- 信息瓶颈：$\max [I(\mathbf{Y};\mathbf{Z}) - \beta I(\mathbf{X};\mathbf{Z})]$
- 神经压缩：$\mathcal{L} = d(\mathbf{x}, \hat{\mathbf{x}}) + \lambda H(\hat{\mathbf{y}})$

---

## 8.7 常见陷阱与错误

### Gotcha #1: VAE的KL坍塌

**错误**：训练VAE时，KL项趋于0（$q(\mathbf{z}|\mathbf{x}) \approx p(\mathbf{z})$），模型忽略隐变量。

**正解**：这是"KL vanishing"问题，常见于强大的解码器。解决：
- KL退火：训练初期$\beta$小，逐渐增大
- 自由比特（free bits）：只惩罚超过阈值的KL
- 弱解码器：限制解码器容量

### Gotcha #2: 量化的可微性

**错误**：直接用round()作为量化，导致梯度为0，无法训练。

**正解**：使用STE（Straight-Through Estimator）：
- 前向：$\hat{y} = \text{round}(y)$
- 反向：$\frac{\partial \mathcal{L}}{\partial y} = \frac{\partial \mathcal{L}}{\partial \hat{y}}$（假装量化是恒等映射）

或用加噪声近似：$\hat{y} = y + u$，$u \sim \text{Uniform}(-0.5, 0.5)$。

### Gotcha #3: 熵模型的自回归依赖

**错误**：用自回归模型（如PixelCNN）建模$p(\hat{\mathbf{y}})$，解码极慢（串行）。

**正解**：
- 训练时可以用自回归（并行计算）
- 测试时用更简单的模型（如hyperprior、上下文模型）
- 或者接受慢速度（高质量场景）

### Gotcha #4: $\lambda$的选择

**错误**：用固定$\lambda$（如1.0）训练，期望适用所有码率。

**正解**：不同$\lambda$对应不同率失真工作点。实际中：
- 训练多个模型（$\lambda \in \{0.01, 0.05, 0.1, ...\}$）
- 或用可变$\lambda$的单一模型（如条件生成）

### Gotcha #5: 过拟合到训练分布

**错误**：在特定数据集（如ImageNet）训练的神经压缩器，在其他数据（如医学图像、卫星图像）上性能差。

**正解**：
- 多样化训练数据
- 或针对目标域微调
- 传统编码器（JPEG、H.265）更鲁棒，因为不依赖特定数据分布

### Gotcha #6: 模型参数的开销

**错误**：忽略神经网络参数的存储开销。

**正解**：神经压缩器的参数（几MB到几十MB）需要存储和传输。对于单张图像，这可能超过图像本身的码率。神经压缩适合批量压缩或已部署模型的场景，不适合一次性压缩单个文件。

### Gotcha #7: 互信息的估计

**错误**：直接计算连续变量的互信息$I(\mathbf{X};\mathbf{Z})$（在IB中）。

**正解**：连续变量的互信息依赖微分熵，难以准确估计。实际中使用：
- 变分上界/下界（如MINE、NWJ）
- 离散化
- 或避免显式计算互信息，用VAE等代理目标

---

**下一章预告**：第九章将汇总率失真理论在实际工程中的应用策略、调试技巧和经验法则，提供最佳实践指导。

[← 第七章](chapter7.md) | [返回目录](index.md) | [第九章：实践指南与应用案例 →](chapter9.md)

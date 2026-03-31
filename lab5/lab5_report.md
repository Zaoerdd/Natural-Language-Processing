# Lab 5 Report: BPE Tokenization and Minimal GPT Pretraining

## 1. 实验目标

本次 Lab 5 分为两部分：

- Part A: 实现 Byte-Pair Encoding (BPE) 的核心训练流程，理解子词词表是如何从字符级别逐步合并得到的。
- Part B: 基于上一节 lab 的 `GPTModel`，实现一个最小可运行的预训练流程，包括交叉熵损失、数据加载、训练循环和评估函数。

本次实验的重点不是训练出一个高质量的大模型，而是把“分词”和“预训练”这两条最核心的链路打通。


## 2. 完成内容

本次已完成并整理的文件如下：

- `lab5_bpe.ipynb`: 已补全 BPE notebook 中全部待实现代码。
- `lab5_pretrain.ipynb`: 已补全 pretraining notebook 中全部待实现代码。
- `lab5_bpe_completed.ipynb`: 完成版 BPE notebook 备份。
- `lab5_pretrain_completed.ipynb`: 完成版 pretraining notebook 备份。
- `lab5_report.md`: 本实验报告。

本次补全的核心函数包括：

- `compute_pair_freqs`
- `merge_pair`
- BPE 主训练循环
- `calc_loss_batch`
- `calc_loss_loader`
- `train_model_simple`
- `evaluate_model`


## 3. Part A: BPE 实验分析

### 3.1 实验流程

BPE 的整体流程可以概括为：

1. 对原始文本做 pre-tokenization。
2. 统计语料中每个词的频率。
3. 用所有出现过的字符构造基础词表。
4. 把每个词拆成字符序列。
5. 反复统计相邻 token pair 的频率。
6. 每次选取最高频 pair 进行 merge，直到词表达到目标大小。

本实验使用 HuggingFace 课程中的一个小语料，共 4 句话，例如：

- `This is the Hugging Face Course.`
- `This chapter is about tokenization.`
- `This section shows several tokenizer algorithms.`
- `Hopefully, you will be able to understand how they are trained and generate tokens.`

### 3.2 预分词结果与基础词表

使用 GPT-2 的 pre-tokenizer 后，语料中会出现特殊前缀字符 `臓`，它本质上是在字节级 BPE 里用来编码空格信息的可视化符号。  
例如：

- `is` 会被写成 `臓is`
- `the` 会被写成 `臓the`
- `trained` 会被写成 `臓trained`

统计全部字符后，得到：

- 字符表大小：30
- 加上特殊 token `"<|endoftext|>"` 后，基础词表大小：31

### 3.3 代码实现说明

本部分补全了三段关键代码：

- `compute_pair_freqs`：遍历每个词对应的 split，统计相邻 pair 的总频率，并按词频加权。
- `merge_pair`：在每个词的 token 序列中查找指定 pair `(a, b)`，匹配时替换为合并后的 `a+b`。
- 主循环：不断寻找最高频 pair，执行 merge，并更新 `merges` 与 `vocab`。

### 3.4 关键结果

第一次统计后，最高频 pair 为：

- `('臓', 't')`，频率为 `7`

第一次合并结果为：

- `('臓', 't') -> '臓t'`

在 notebook 后续训练过程中，以目标词表大小 `50` 为终点，最终学得：

- merge 规则数：`19`
- 最终词表大小：`50`

最终得到的代表性 merge 包括：

- `('i', 's') -> 'is'`
- `('e', 'r') -> 'er'`
- `('臓', 'a') -> '臓a'`
- `('臓t', 'o') -> '臓to'`
- `('T', 'h') -> 'Th'`
- `('Th', 'is') -> 'This'`
- `('臓tok', 'en') -> '臓token'`
- `('臓tokeni', 'z') -> '臓tokeniz'`

这说明 BPE 会优先把高频子串合并成更长的子词，从而在词表规模受限的情况下，兼顾常见词与未登录词的建模能力。

### 3.5 对 BPE 的理解

BPE 的优点主要体现在三点：

- 比纯字符建模更高效，因为常见子串会被压缩成单个 token。
- 比纯词级建模更灵活，因为生词仍然可以退化为多个子词甚至字符。
- 能自然处理词形变化，例如 `token`, `tokeni`, `tokeniz`, `tokenization` 这类共享前缀的词。


## 4. Part B: 预训练实验分析

### 4.1 模型配置

本实验使用 `utils.py` 中提供的 `GPTModel`，配置如下：

- `vocab_size = 50257`
- `context_length = 256`
- `emb_dim = 768`
- `n_heads = 12`
- `n_layers = 12`
- `drop_rate = 0.1`
- `qkv_bias = False`

这是一个“结构完整、规模缩小”的 GPT-2 风格模型，适合教学场景下演示预训练的基本流程。

### 4.2 tokenizer 观察

实验首先用 `tiktoken` 的 GPT-2 tokenizer 对几个句子进行编码。  
其中一句：

- `How about some anomalous words`

被切分为 6 个 token，其中：

- `anomalous -> anomal + ous`

这说明 GPT-2 tokenizer 采用的也是子词级切分方式，而不是简单的整词切分。

### 4.3 交叉熵损失的理解

在 toy example 中，输入是两个长度为 4 的 token 序列，模型输出的 logits 形状为：

- `torch.Size([2, 4, 50257])`

将 logits 做 softmax 后，可以取出每个位置上真实 target token 的概率。由于模型初始未训练，这些概率都很小，例如：

- `7.6919e-06`
- `1.8792e-05`
- `9.7180e-06`
- `1.9871e-05`

于是：

- 平均 log probability = `-11.1840`
- negative mean log probability = `11.1840`

这与 `F.cross_entropy(logits_flat, targets_flat)` 的结果完全一致，说明交叉熵就是“目标 token 的负对数似然”的平均值。

如果不做平均，逐元素损失 `nll_loss_all` 为：

- `tensor([11.7753, 10.8821, 11.5415, 10.8262, 11.7870, 10.0923, 11.5493, 11.0183])`

它对应的正是每个 token 的 `-log(p_target)`。

### 4.4 数据集与 DataLoader

本实验使用的文本是：

- `the-verdict.txt`

统计得到：

- 总 token 数：`5145`
- 训练/验证划分比例：`90% / 10%`
- `batch_size = 2`
- `context_length = 256`

从 DataLoader 取出的单个 batch 形状为：

- `inputs: torch.Size([2, 256])`
- `targets: torch.Size([2, 256])`

这说明每个 batch 包含 2 条长度为 256 的序列，而 `targets` 相当于 `inputs` 向右平移一位后的 next-token prediction 标签。

### 4.5 代码实现说明

本部分补全了预训练流程中的 4 个关键函数：

- `calc_loss_batch`：前向计算 logits，并用交叉熵得到单个 batch 的损失。
- `calc_loss_loader`：遍历 data loader，求平均损失。
- `train_model_simple`：实现标准训练循环，包括 `zero_grad`、`backward`、`step`、统计 `tokens_seen` 和定期评估。
- `evaluate_model`：在 `torch.no_grad()` 下分别计算训练集和验证集损失。

此外，为了避免 GPU 环境下模型和输入不在同一设备的问题，在初次评估前补上了：

- `model.to(device)`

### 4.6 训练前损失

模型刚初始化、尚未训练时，记录到的损失为：

- Train Loss: `10.989524`
- Validation Loss: `10.974542`

这与前面的 toy example 结果一致，说明随机初始化模型对真实 target 的预测仍然接近随机分布。

### 4.7 训练过程与结果

实验使用：

- 优化器：`AdamW`
- 学习率：`0.0004`
- `weight_decay = 0.1`
- 训练轮数：`10`
- 每 5 个 step 做一次评估

notebook 中记录的训练日志如下：

| Step | Train Loss | Val Loss |
| --- | ---: | ---: |
| 0 | 9.781 | 9.933 |
| 5 | 8.111 | 8.339 |
| 10 | 6.661 | 7.048 |
| 15 | 5.961 | 6.616 |
| 20 | 5.726 | 6.600 |
| 25 | 5.201 | 6.348 |
| 30 | 4.417 | 6.278 |
| 35 | 4.069 | 6.226 |
| 40 | 3.732 | 6.160 |
| 45 | 2.850 | 6.179 |
| 50 | 2.427 | 6.141 |
| 55 | 2.104 | 6.134 |
| 60 | 1.882 | 6.233 |
| 65 | 1.320 | 6.238 |
| 70 | 0.985 | 6.242 |
| 75 | 0.717 | 6.293 |
| 80 | 0.541 | 6.393 |
| 85 | 0.391 | 6.452 |

训练总耗时：

- `1.64 minutes`

### 4.8 结果分析

从训练曲线可以看出：

- 训练损失从 `9.781` 持续下降到 `0.391`，说明模型已经能够很好地记忆训练集模式。
- 验证损失从 `9.933` 下降到约 `6.134` 后趋于平台，随后略有回升。
- 在大约 step `55` 左右之后，训练损失继续快速下降，但验证损失不再同步下降，说明模型开始出现明显过拟合。

造成这种现象的主要原因是：

- 数据集很小，只有 `5145` 个 token；
- 模型参数规模相对大；
- 训练轮数较多，模型容易“记住”训练文本而不是学习可泛化模式。

因此，这个实验非常适合用来理解“预训练流程是怎样工作的”，但并不适合用来训练一个真正有泛化能力的语言模型。


## 5. 实验结论

通过本次 Lab 5，可以得到以下结论：

1. BPE 的本质是“从字符开始，按频率不断合并高频相邻子串”，它能在固定词表规模下兼顾效率与泛化能力。
2. GPT 风格预训练的核心任务是 next-token prediction，其目标函数就是交叉熵，也即目标 token 负对数似然的平均值。
3. 即使模型结构正确，如果语料太小，也很容易在训练后期出现过拟合。
4. 一个最小可运行的预训练系统至少需要 tokenizer、数据切块、模型前向、loss 计算、优化器更新和评估循环这几个部分协同工作。

整体来看，本次 lab 已经完整打通了：

- 从 BPE 分词构建词表
- 到 GPT 输入序列构造
- 再到 next-token prediction 训练

这正是后续理解大语言模型训练机制的基础。


## 6. 说明

当前终端环境中无法直接调用 Python 解释器，因此本次未在终端里重新执行 notebook。  
本报告中的数值结果来自 notebook 文件中已保存的输出；代码补全部分已静态检查完成，两个 notebook 中的占位实现也已全部补齐。

# Lab4 逐行详解文档

本文档对应 [lab4_trm_completed.ipynb](D:/OneDrive/Study/2026%20Spring/Natural%20Language%20Processing/lab4/lab4_trm_completed.ipynb)，按 cell 顺序解释 notebook 里的全部文字和代码内容。

说明：

- 我会保留每个 cell 的原始内容。
- 然后按“逐行解释”的方式说明每一行在做什么、为什么这样写。
- 空行如果没有语义，只统一说明为“用于排版分段”。

---

## Cell 00 | Markdown

原文：

```markdown
## CS310 Natural Language Processing
## Lab 4: Transformer Basics

In this lab, we will implement the multihead attention with future masking for causal language modeling.

Install the `tiktoken` library by running:
```bash
pip install tiktoken
```
```

逐行解释：

- 第 1 行 `## CS310 Natural Language Processing`
  - 这是二级标题，说明本 notebook 属于 CS310 这门自然语言处理课程。
- 第 2 行 `## Lab 4: Transformer Basics`
  - 这是实验标题，告诉你这次 lab 的主题是 Transformer 的基础组件。
- 空行
  - 用于把标题和正文分开，纯排版作用。
- 第 4 行 `In this lab, we will implement the multihead attention with future masking for causal language modeling.`
  - 这句话是全文目标说明。
  - `multihead attention` 指多头注意力。
  - `future masking` 指因果掩码，也就是不能看未来 token。
  - `causal language modeling` 指 GPT 这一类从左到右预测下一个词的语言模型。
- 空行
  - 用于分段。
- 第 6 行 `Install the \`tiktoken\` library by running:`
  - 告诉你后面会用到 `tiktoken`，这是一个 tokenizer 库，负责把文本转成 token id。
- 第 7-9 行代码块
  - 给出了安装命令 `pip install tiktoken`。
  - 这不是 notebook 内部执行的 Python 代码，而是你在终端里需要执行的命令。

## Cell 01 | Code

原文：

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
```

逐行解释：

- 第 1 行 `import numpy as np`
  - 导入 NumPy，并把名字缩写成 `np`。
  - 本 lab 里主要用它做 `np.sqrt(...)` 和 `-np.inf`。
- 第 2 行 `import torch`
  - 导入 PyTorch 主包。
  - 后面所有 tensor、随机数、矩阵乘法都依赖它。
- 第 3 行 `import torch.nn as nn`
  - 导入神经网络模块并命名为 `nn`。
  - 后面定义 `Linear`、`Parameter`、`Module` 时都会用到。
- 第 4 行 `import torch.nn.functional as F`
  - 导入函数式 API 并命名为 `F`。
  - 本 lab 里主要用它实现 `softmax`。

## Cell 02 | Markdown

原文：

```markdown
## T1. Implement Self-Attention for a Single Head

First, prepare the input data in shape $N\times d$

*Hint*: 
- Use `torch.randn` to generate a torch tensor in the correct shape.
```

逐行解释：

- 第 1 行 `## T1. Implement Self-Attention for a Single Head`
  - 表示任务 1：先从单头 self-attention 开始实现。
- 空行
  - 用于分段。
- 第 3 行 `First, prepare the input data in shape $N\times d$`
  - 先创建输入矩阵 `X`。
  - `N` 是 token 数量，`d` 是 embedding 维度。
  - 这里的输入还不是来自真实语料，而是人为构造的张量。
- 空行
  - 用于分段。
- 第 5 行 `*Hint*:`
  - 给出实现提示。
- 第 6 行 `- Use \`torch.randn\` to generate a torch tensor in the correct shape.`
  - 明确要求使用 `torch.randn` 创建一个正态分布随机张量。
  - 关键是 shape 要写成 `(N, d)`。

## Cell 03 | Code

原文：

```python
N = 3
d = 512
torch.manual_seed(0)

### START YOUR CODE ###
X = torch.randn(N, d)
### END YOUR CODE ###

# Test 
assert isinstance(X, torch.Tensor)
print('X.size():', X.size())
print('X[:,0]:', X[:,0].data.numpy())

# You should expect to see the following results:
# X.shape: (3, 512)
# X[:,0]: [-1.1258398  -0.54607284 -1.0840825 ]
```

逐行解释：

- 第 1 行 `N = 3`
  - 把 token 数量设为 3。
  - 这里等价于“序列长度是 3”。
- 第 2 行 `d = 512`
  - 把 embedding 维度设为 512。
  - 这是每个 token 向量的长度。
- 第 3 行 `torch.manual_seed(0)`
  - 固定 PyTorch 的随机种子，让每次运行都生成相同的随机数。
  - 这样测试输出才能稳定复现。
- 空行
  - 用于分段。
- 第 5 行 `### START YOUR CODE ###`
  - 这是老师留给你填写答案的起始标记。
- 第 6 行 `X = torch.randn(N, d)`
  - 生成一个形状为 `(3, 512)` 的随机张量。
  - 每一行可以理解成一个 token 的 embedding。
  - `torch.randn` 使用标准正态分布采样。
- 第 7 行 `### END YOUR CODE ###`
  - 这是答案区结束标记。
- 空行
  - 用于分段。
- 第 10 行 `# Test`
  - 下面开始做测试。
- 第 11 行 `assert isinstance(X, torch.Tensor)`
  - 检查 `X` 是否真的是一个 PyTorch tensor。
  - 如果你误写成别的类型，这一行会直接报错。
- 第 12 行 `print('X.size():', X.size())`
  - 打印 `X` 的 shape，确认它是否是 `(3, 512)`。
- 第 13 行 `print('X[:,0]:', X[:,0].data.numpy())`
  - 打印 `X` 第 0 列的所有元素。
  - 这样做是为了更细粒度验证随机种子和内容是否一致。
- 空行
  - 用于分段。
- 第 15-17 行注释
  - 告诉你预期输出是什么。
  - 如果你的结果不同，通常说明 shape、随机种子、或者 API 用错了。

## Cell 04 | Markdown

原文：

```markdown
Then, initialize weight matrices $W^Q$, $W^K$, and $W^V$. We assume they are for a single head, so $d_k=d_v=d$

Using $W^Q$ as an example
- First initialize an empty tensor `W_q` in the dimension of $d\times d_k$, using the `torch.empty()` function. Then initialize it with `nn.init.xavier_normal_()`.
- After `W_q` is initialized, obtain the query matrix `Q` with a multiplication between `X` and `W_q`, using `torch.matmul()`.
```

逐行解释：

- 第 1 行
  - 开始说明如何构造 `W^Q`、`W^K`、`W^V`。
  - 单头情况下，通常令 `d_k = d_v = d`，这样每个投影后的向量维度和原 embedding 维度相同。
- 空行
  - 用于分段。
- 第 3 行 `Using $W^Q$ as an example`
  - 先用 `W^Q` 举例，`W^K` 和 `W^V` 完全同理。
- 第 4 行
  - 先创建一个空张量 `W_q`。
  - shape 是 `d x d_k`。
  - 然后用 `xavier_normal_` 做权重初始化，这是深度学习里比较常见的初始化方式。
- 第 5 行
  - 提示你如何把输入 `X` 乘上权重矩阵得到 `Q`。
  - 这一步本质上就是一个无 bias 的线性变换。

## Cell 05 | Code

原文：

```python
torch.manual_seed(0) # Do not remove this line

n_heads = 1

### START YOUR CODE ###
d_k = d // n_heads # Compute d_k

W_q = torch.empty(d, d_k)
W_k = torch.empty(d, d_k)
W_v = torch.empty(d, d_k)

nn.init.xavier_normal_(W_q)
nn.init.xavier_normal_(W_k)
nn.init.xavier_normal_(W_v)

# Compute Q, K, V
Q = torch.matmul(X, W_q)
K = torch.matmul(X, W_k)
V = torch.matmul(X, W_v)
### END YOUR CODE ###

# Test
assert Q.size() == (N, d_k)
assert K.size() == (N, d_k)
assert V.size() == (N, d_k)

print('Q.size():', Q.size())
print('Q[:,0]:', Q[:,0].data.numpy())
print('K.size():', K.size())
print('K[:,0]:', K[:,0].data.numpy())
print('V.size():', V.size())
print('V[:,0]:', V[:,0].data.numpy())

# You should expect to see the following results:
# Q.size(): torch.Size([3, 512])
# Q[:,0]: [-0.45352045 -0.40904033  0.18985942]
# K.size(): torch.Size([3, 512])
# K[:,0]: [ 1.509987   -0.5503683   0.44788954]
# V.size(): torch.Size([3, 512])
# V[:,0]: [ 0.43034226  0.00162293 -0.1317436 ]
```

逐行解释：

- 第 1 行 `torch.manual_seed(0) # Do not remove this line`
  - 再次固定随机种子。
  - 这里不能删，因为后面的 `W_q/W_k/W_v` 初始化也依赖随机数。
- 空行
  - 用于分段。
- 第 3 行 `n_heads = 1`
  - 明确当前只考虑单头 attention。
- 空行
  - 用于分段。
- 第 5 行 `### START YOUR CODE ###`
  - 答题区开始。
- 第 6 行 `d_k = d // n_heads # Compute d_k`
  - 计算 key/query 的维度。
  - 这里 `512 // 1 = 512`，所以单头时 `d_k = 512`。
- 空行
  - 用于分段。
- 第 8-10 行
  - 创建三个空权重矩阵：`W_q`、`W_k`、`W_v`。
  - 它们的 shape 都是 `(512, 512)`。
- 空行
  - 用于分段。
- 第 12-14 行
  - 对三个权重矩阵进行 Xavier 正态初始化。
  - 这一步让权重不是未定义垃圾值，而是可用于计算的合理初值。
- 空行
  - 用于分段。
- 第 16 行 `# Compute Q, K, V`
  - 注释，说明下面开始做投影。
- 第 17 行 `Q = torch.matmul(X, W_q)`
  - 把输入 `X` 投影到 query 空间。
  - `X` 是 `(3, 512)`，`W_q` 是 `(512, 512)`，结果 `Q` 是 `(3, 512)`。
- 第 18 行 `K = torch.matmul(X, W_k)`
  - 同理，把输入投影到 key 空间。
- 第 19 行 `V = torch.matmul(X, W_v)`
  - 同理，把输入投影到 value 空间。
- 第 20 行 `### END YOUR CODE ###`
  - 答题区结束。
- 空行
  - 用于分段。
- 第 23-26 行 `assert`
  - 检查 `Q/K/V` 的 shape 都是不是 `(N, d_k)`。
- 空行
  - 用于分段。
- 第 28-33 行 `print`
  - 打印 shape 和第一列数值，用来核对内容。
- 空行
  - 用于分段。
- 第 35-41 行注释
  - 给出预期结果，方便你自检。

## Cell 06 | Markdown

原文：

```markdown
Lastly, compute the attention scores $\alpha$ and the weighted output

Following the equation:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

*Hint*:
- $\alpha = \text{softmax}(\frac{QK^\top}{\sqrt{d_k}})$, where you can use `torch.nn.functional.softmax()` to compute the softmax. Pay attention to the `dim` parameter.
- The weighted output is the multiplication between $\alpha$ and $V$. Pay attention to their dimensions: $\alpha$ is of shape $N\times N$, and $\alpha_{ij}$ is the attention score from the $i$-th to the $j$-th word.
- The weighted output is of shape $N\times d_v$, and here we assume $d_k=d_v$.
```

逐行解释：

- 第 1 行
  - 开始介绍单头 attention 的最后一步：计算注意力权重和输出。
- 空行
  - 用于分段。
- 第 3 行 `Following the equation:`
  - 提醒你下面是核心公式。
- 第 4-6 行公式
  - 这是标准的 scaled dot-product attention 公式。
  - 先算 `QK^T / sqrt(d_k)`，再过 softmax，最后乘上 `V`。
- 空行
  - 用于分段。
- 第 8 行 `*Hint*:`
  - 开始给实现提示。
- 第 9 行
  - 先定义 attention 权重矩阵 `alpha`。
  - 关键细节是 `softmax` 的 `dim` 参数必须写对。
- 第 10 行
  - 提醒你 `alpha` 的 shape 是 `N x N`。
  - 其中 `alpha_ij` 表示第 `i` 个 token 对第 `j` 个 token 的注意力权重。
- 第 11 行
  - 提醒你最后输出是 `N x d_v`。
  - 由于本题设定 `d_k = d_v`，所以它也等于 `N x d_k`。

## Cell 07 | Code

原文：

```python
### START YOUR CODE ###
alpha = F.softmax(torch.matmul(Q, K.T) / np.sqrt(d_k), dim=-1)
output = torch.matmul(alpha, V)
### END YOUR CODE ###

# Test
assert alpha.size() == (N, N)
assert output.size() == (N, d_k)

print('alpha.size():', alpha.size())
print('alpha:', alpha.data.numpy())
print('output.size():', output.size())
print('output[:,0]:', output[:,0].data.numpy())

# You should expect to see the following output:
# alpha.size(): torch.Size([3, 3])
# alpha: [[0.78344566 0.14102352 0.07553086]
#  [0.25583813 0.18030964 0.5638523 ]
#  [0.09271843 0.2767209  0.63056064]]
# output.size(): torch.Size([3, 512])
# output[:,0]: [ 0.32742795  0.03610666 -0.04272257]
```

逐行解释：

- 第 1 行 `### START YOUR CODE ###`
  - 答题区开始。
- 第 2 行 `alpha = F.softmax(torch.matmul(Q, K.T) / np.sqrt(d_k), dim=-1)`
  - `K.T` 把 `K` 从 `(N, d_k)` 转成 `(d_k, N)`。
  - `torch.matmul(Q, K.T)` 得到 `N x N` 的相关性矩阵。
  - `/ np.sqrt(d_k)` 是缩放项，避免数值太大。
  - `F.softmax(..., dim=-1)` 把每一行变成一个概率分布。
- 第 3 行 `output = torch.matmul(alpha, V)`
  - 用 attention 权重对 `V` 做加权求和。
  - 结果 shape 是 `(N, d_k)`。
- 第 4 行 `### END YOUR CODE ###`
  - 答题区结束。
- 空行
  - 用于分段。
- 第 7-8 行 `assert`
  - 检查 `alpha` 的 shape 是 `(N, N)`。
  - 检查 `output` 的 shape 是 `(N, d_k)`。
- 空行
  - 用于分段。
- 第 11-14 行 `print`
  - 打印注意力矩阵和输出，用于和题目参考答案比对。
- 空行
  - 用于分段。
- 第 16-22 行注释
  - 给出参考结果。
  - 如果你 `softmax` 的维度写错，这里的数值几乎一定对不上。

## Cell 08 | Markdown

原文：

```markdown
## T2. Mask Future Tokens

First, create a binary mask tensor of size $N\times N$, which is lower triangular, with the diagonal and upper triangle set to 0.

*Hint*: Use `torch.tril` and `torch.ones`.
```

逐行解释：

- 第 1 行 `## T2. Mask Future Tokens`
  - 任务 2：实现因果 mask。
- 空行
  - 用于分段。
- 第 3 行
  - 要求你构造一个 `N x N` 的二值掩码矩阵。
  - 这个矩阵是下三角形式，表示只能看当前位置及其之前的位置。
- 空行
  - 用于分段。
- 第 5 行
  - 明确提示使用 `torch.tril` 和 `torch.ones`。
  - `tril` 是 lower triangular 的缩写。

## Cell 09 | Code

原文：

```python
### START YOUR CODE ###
mask = torch.tril(torch.ones(N, N))
### END YOUR CODE ###

# Test
print('mask:', mask.data.numpy())

# You should expect to see the following output:
# mask: [[1. 0. 0.]
#  [1. 1. 0.]
#  [1. 1. 1.]]
```

逐行解释：

- 第 1 行 `### START YOUR CODE ###`
  - 答题区开始。
- 第 2 行 `mask = torch.tril(torch.ones(N, N))`
  - 先创建一个全 1 的 `N x N` 矩阵。
  - 再取其下三角部分。
  - 对于 `N=3`，结果是：
    - 第 1 行只能看第 1 个位置。
    - 第 2 行能看前 2 个位置。
    - 第 3 行能看前 3 个位置。
- 第 3 行 `### END YOUR CODE ###`
  - 答题区结束。
- 空行
  - 用于分段。
- 第 5-6 行
  - 打印 mask 矩阵，确认它是不是下三角。
- 空行
  - 用于分段。
- 第 8-10 行注释
  - 给出参考输出。

## Cell 10 | Markdown

原文：

```markdown
Use the mask to fill the corresponding future cells in $QK^\top$ with $-\infty$ (`-np.inf`), and then pass it to softmax to compute the new attention scores.

*Hint*: Use `torch.Tensor.masked_fill` function to selectively fill the upper triangle area of the result.
```

逐行解释：

- 第 1 行
  - 说明下一步要把未来位置对应的 attention score 改成 `-inf`。
  - 这样经过 softmax 后，那些位置的概率会变成 0。
- 空行
  - 用于分段。
- 第 3 行
  - 明确提示使用 `masked_fill`。
  - 它会根据布尔条件选择性地替换张量里的部分元素。

## Cell 11 | Code

原文：

```python
### START YOUR CODE ###
masked_scores = torch.matmul(Q, K.T) / np.sqrt(d_k)
masked_scores = masked_scores.masked_fill(mask == 0, -np.inf)
new_alpha = F.softmax(masked_scores, dim=-1)
### END YOUR CODE ###

# Test
print('new_alpha:', new_alpha.data.numpy())

# You should expect to see the following results:
# new_alpha: [[1.         0.         0.        ]
#  [0.5865858  0.41341412 0.        ]
#  [0.09271843 0.2767209  0.63056064]]
```

逐行解释：

- 第 1 行 `### START YOUR CODE ###`
  - 答题区开始。
- 第 2 行 `masked_scores = torch.matmul(Q, K.T) / np.sqrt(d_k)`
  - 重新计算未加 mask 的 score 矩阵。
  - 之所以单独存到 `masked_scores`，是为了不覆盖原来的 `alpha` 逻辑。
- 第 3 行 `masked_scores = masked_scores.masked_fill(mask == 0, -np.inf)`
  - 找到 mask 中等于 0 的位置，也就是未来位置。
  - 把这些位置的 score 改成 `-inf`。
  - 这样 softmax 后这些位置的概率就会是 0。
- 第 4 行 `new_alpha = F.softmax(masked_scores, dim=-1)`
  - 对加过 mask 的分数做 softmax，得到新的注意力权重。
- 第 5 行 `### END YOUR CODE ###`
  - 答题区结束。
- 空行
  - 用于分段。
- 第 8-9 行
  - 打印新的注意力矩阵。
  - 你会发现未来位置的概率被归零了。
- 空行
  - 用于分段。
- 第 11-14 行注释
  - 给出参考结果。
  - 第一行只能看自己，所以是 `[1, 0, 0]`。

## Cell 12 | Markdown

原文：

```markdown
Lastly, the output should also be updated:
```

逐行解释：

- 第 1 行
  - 提醒你：既然 attention 权重变了，那么最后的输出向量也必须重新计算。

## Cell 13 | Code

原文：

```python
### START YOUR CODE ###
new_output = torch.matmul(new_alpha, V)
### END YOUR CODE ###

# Test
print('new_output.size():', new_output.size())
print('new_output[:,0]:', new_output[:,0].data.numpy())

# You should expect to see the following results:
# new_output.size(): torch.Size([3, 512])
# new_output[:,0]: [ 0.43034226  0.2531036  -0.04272257]
```

逐行解释：

- 第 1 行 `### START YOUR CODE ###`
  - 答题区开始。
- 第 2 行 `new_output = torch.matmul(new_alpha, V)`
  - 用带 mask 的注意力权重重新对 `V` 做加权求和。
  - 得到新的上下文表示。
- 第 3 行 `### END YOUR CODE ###`
  - 答题区结束。
- 空行
  - 用于分段。
- 第 5-7 行
  - 打印输出的 shape 和第一列数值。
  - 用于验证 mask 之后输出确实发生了变化。
- 空行
  - 用于分段。
- 第 9-11 行注释
  - 给出参考答案。

## Cell 14 | Markdown

原文：

```markdown
## T3. Integrate Multiple Heads

Finally, integrate the above implemented functions into the `ToyMultiHeadAttention` class.

For the convenience of later integration, we define a configuration dictionary using the hyper-parameter setep from GPT-2.

**Note**:

- In this class, the weight matrices `W_q`, `W_k`, and `W_v` are defined as tensors of size $d\times d$. Thus, the output $Q=XW^Q$ is of size $N\times d$.

- Then we reshape $Q$ (and $K$, $V$ as well) into the tensor `Q_` of shape $N\times h\times d_k$, where $h$ is the number of heads (`n_heads`) and $d_k = d // h$. Similar operations are applied to $K$ and $V$.

- The multiplication $QK^\top$ is now between two tensors of shape $N\times h\times d_k$, `Q_` and `K_`, and the output is of size $h\times N \times N$. Thus, you need to use `torch.matmul` and `torch.permute` properly to make the dimensions of `Q_`, `K_`, and `V_` be in the correct order.

- Also, remember to apply the future mask to each attention head's output. You can use `torch.repeat` to replicate the mask for `n_heads` times.
```

逐行解释：

- 第 1 行 `## T3. Integrate Multiple Heads`
  - 任务 3：把前面单头 attention 的逻辑整合成多头版本。
- 空行
  - 用于分段。
- 第 3 行
  - 说明这次不再分散写几个小步骤，而是把逻辑放进一个类里。
- 空行
  - 用于分段。
- 第 5 行
  - 说下面会用 GPT-2 风格的配置字典。
  - 这样后面的模块可以更容易复用。
- 空行
  - 用于分段。
- 第 7 行 `**Note**:`
  - 开始强调实现细节。
- 空行
  - 用于分段。
- 第 9 行
  - 说明本类里的 `W_q/W_k/W_v` 是 `d x d`。
  - 所以先投影得到的 `Q/K/V` 还是 `N x d`。
- 空行
  - 用于分段。
- 第 11 行
  - 说明后面要把 `Q/K/V` reshape 成 `N x h x d_k`。
  - 这里 `h` 是头数，`d_k = d // h`。
- 空行
  - 用于分段。
- 第 13 行
  - 说明多头情况下矩阵乘法的 shape 会变化。
  - 最后 score 张量变成 `h x N x N`，也就是每个 head 都有一张注意力图。
- 空行
  - 用于分段。
- 第 15 行
  - 说明 mask 也要复制到每个 head 上。
  - `repeat` 的作用就是把同一张下三角 mask 复制 `n_heads` 份。

## Cell 15 | Code

原文：

```python
# GPT configuration
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

# Toy MultiHeadAttention
class ToyMultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super(ToyMultiHeadAttention, self).__init__()
        d_model = cfg["emb_dim"]
        n_heads = cfg["n_heads"]
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))

        nn.init.xavier_normal_(self.W_q)
        nn.init.xavier_normal_(self.W_k)
        nn.init.xavier_normal_(self.W_v)
        
    def forward(self, X):
        N = X.size(0)
        
        ### START YOUR CODE ###
        Q = torch.matmul(X, self.W_q)
        K = torch.matmul(X, self.W_k)
        V = torch.matmul(X, self.W_v)
        Q_ = Q.view(N, self.n_heads, self.d_k).permute(1, 0, 2)
        K_ = K.view(N, self.n_heads, self.d_k).permute(1, 0, 2)
        V_ = V.view(N, self.n_heads, self.d_k).permute(1, 0, 2)

        # Raw attention scores
        alpha = torch.matmul(Q_, K_.permute(0, 2, 1)) / np.sqrt(self.d_k)
        # Apply the mask
        mask = torch.tril(torch.ones(N, N, device=X.device)).unsqueeze(0).repeat(self.n_heads, 1, 1)
        alpha = alpha.masked_fill(mask == 0, -np.inf)
        # Softmax
        alpha = F.softmax(alpha, dim=-1)

        output = torch.matmul(alpha, V_).permute(1, 0, 2).contiguous().view(N, -1)
        ### END YOUR CODE ###

        return output
```

逐行解释：

- 第 1 行 `# GPT configuration`
  - 注释，说明下面是 GPT 的超参数配置。
- 第 2-10 行 `GPT_CONFIG_124M = {...}`
  - 定义一个字典，保存模型常用超参数。
  - `vocab_size`：词表大小。
  - `context_length`：最大上下文长度。
  - `emb_dim`：embedding 维度。
  - `n_heads`：多头数量。
  - `n_layers`：Transformer block 数量。
  - `drop_rate`：dropout 比例。
  - `qkv_bias`：Q/K/V 线性层是否使用 bias。
- 空行
  - 用于分段。
- 第 13 行 `# Toy MultiHeadAttention`
  - 注释，说明下面是一个教学用的简化版多头注意力类。
- 第 14 行 `class ToyMultiHeadAttention(nn.Module):`
  - 定义一个继承 `nn.Module` 的模块类。
- 第 15 行 `def __init__(self, cfg):`
  - 构造函数，接收配置字典。
- 第 16 行 `super(ToyMultiHeadAttention, self).__init__()`
  - 调用父类初始化函数。
- 第 17 行 `d_model = cfg["emb_dim"]`
  - 从配置中读出 embedding 维度。
- 第 18 行 `n_heads = cfg["n_heads"]`
  - 从配置中读出头数。
- 第 19 行 `assert d_model % n_heads == 0`
  - 检查 embedding 维度能否被头数整除。
  - 如果不能整除，就无法平均切分多个 head。
- 第 20 行 `self.d_k = d_model // n_heads`
  - 计算每个 head 的维度。
- 第 21 行 `self.n_heads = n_heads`
  - 保存头数，供后续 forward 使用。
- 空行
  - 用于分段。
- 第 23-25 行
  - 创建三组可学习参数 `W_q/W_k/W_v`。
  - `nn.Parameter` 表示这些张量会被优化器更新。
- 空行
  - 用于分段。
- 第 27-29 行
  - 用 Xavier 正态分布初始化三个参数矩阵。
- 空行
  - 用于分段。
- 第 31 行 `def forward(self, X):`
  - 定义前向传播逻辑。
- 第 32 行 `N = X.size(0)`
  - 取出序列长度。
  - 这里 `X` 的 shape 是 `(N, d_model)`。
- 空行
  - 用于分段。
- 第 34 行 `### START YOUR CODE ###`
  - 开始填写核心多头逻辑。
- 第 35-37 行
  - 先分别计算 `Q/K/V`。
  - 这里结果 shape 都还是 `(N, d_model)`。
- 第 38-40 行
  - 先 `view(N, n_heads, d_k)`，把 embedding 维度切成多个 head。
  - 再 `permute(1, 0, 2)`，把维度顺序变成 `(n_heads, N, d_k)`。
  - 这样更方便后面按 head 做 batch 矩阵乘法。
- 空行
  - 用于分段。
- 第 42 行 `# Raw attention scores`
  - 注释，说明下面开始算每个 head 的原始分数。
- 第 43 行 `alpha = torch.matmul(Q_, K_.permute(0, 2, 1)) / np.sqrt(self.d_k)`
  - `K_.permute(0, 2, 1)` 把 `K_` 变成 `(h, d_k, N)`。
  - 与 `Q_` 的 `(h, N, d_k)` 相乘后，得到 `(h, N, N)`。
  - 最后除以 `sqrt(d_k)` 做缩放。
- 第 44 行 `# Apply the mask`
  - 注释，说明下面开始加因果 mask。
- 第 45 行 `mask = torch.tril(torch.ones(N, N, device=X.device)).unsqueeze(0).repeat(self.n_heads, 1, 1)`
  - 先构造单张 `(N, N)` 的下三角 mask。
  - `device=X.device` 保证 mask 和输入在同一个设备上。
  - `unsqueeze(0)` 变成 `(1, N, N)`。
  - `repeat(self.n_heads, 1, 1)` 复制成 `(h, N, N)`。
- 第 46 行 `alpha = alpha.masked_fill(mask == 0, -np.inf)`
  - 把未来位置的 score 改成 `-inf`。
- 第 47 行 `# Softmax`
  - 注释，说明下一步做归一化。
- 第 48 行 `alpha = F.softmax(alpha, dim=-1)`
  - 对最后一维做 softmax。
  - 每个 head、每个 query 位置都会对所有 key 位置形成一组概率分布。
- 空行
  - 用于分段。
- 第 50 行 `output = torch.matmul(alpha, V_).permute(1, 0, 2).contiguous().view(N, -1)`
  - 先让 `(h, N, N)` 的注意力矩阵乘上 `(h, N, d_k)` 的 `V_`。
  - 得到 `(h, N, d_k)` 的每头输出。
  - 再 `permute(1, 0, 2)` 变成 `(N, h, d_k)`。
  - `contiguous()` 让内存布局连续，方便后面 reshape。
  - 最后 `view(N, -1)` 把多头拼回 `(N, d_model)`。
- 第 51 行 `### END YOUR CODE ###`
  - 答题区结束。
- 空行
  - 用于分段。
- 第 53 行 `return output`
  - 返回最终多头注意力输出。

## Cell 16 | Code

原文：

```python
# Test
torch.manual_seed(0)

MY_CONFIG = GPT_CONFIG_124M.copy()
MY_CONFIG["emb_dim"] = 512
MY_CONFIG["n_heads"] = 1

multi_head_attn = ToyMultiHeadAttention(MY_CONFIG)
output = multi_head_attn(X)

assert output.size() == (N, d)
print('output.size():', output.size())
print('output[:,0]:', output[:,0].data.numpy())

# You should expect to see the following results:
# output.size(): torch.Size([3, 512])
# output[:,0]: [ 0.43034226  0.2531036  -0.04272257]
```

逐行解释：

- 第 1 行 `# Test`
  - 开始测试刚才写好的多头类。
- 第 2 行 `torch.manual_seed(0)`
  - 固定随机种子，使参数初始化稳定。
- 空行
  - 用于分段。
- 第 4 行 `MY_CONFIG = GPT_CONFIG_124M.copy()`
  - 复制一份配置，避免直接修改原始配置。
- 第 5 行 `MY_CONFIG["emb_dim"] = 512`
  - 把 embedding 维度改成和前面单头实验一致的 512。
- 第 6 行 `MY_CONFIG["n_heads"] = 1`
  - 把头数设成 1。
  - 这样多头模块应该退化成单头模块。
- 空行
  - 用于分段。
- 第 8 行 `multi_head_attn = ToyMultiHeadAttention(MY_CONFIG)`
  - 用这个配置实例化模块。
- 第 9 行 `output = multi_head_attn(X)`
  - 把之前的输入 `X` 送进去，得到输出。
- 空行
  - 用于分段。
- 第 11 行 `assert output.size() == (N, d)`
  - 检查输出 shape 是否是 `(3, 512)`。
- 第 12-13 行 `print`
  - 打印 shape 和第一列的数值，用来和前面单头 masked output 对比。
- 空行
  - 用于分段。
- 第 15-17 行注释
  - 给出参考结果。

## Cell 17 | Markdown

原文：

```markdown
**Note** that the above output size and values should be the same as the previous one, as we used `n_heads=1`.
```

逐行解释：

- 第 1 行
  - 强调一个重要 sanity check。
  - 因为这里设置 `n_heads = 1`，所以多头实现应当退化成前面单头 masked attention 的结果。
  - 如果数值对不上，通常说明 reshape、permute 或 mask 逻辑有问题。

## Cell 18 | Markdown

原文：

```markdown
## T4. FeedForward and LayerNorm

OK, next we go ahead a define other components needed in a Transformer block: `FeedForward` and `LayerNorm`.

`FeedForward` is basically a small neural network with two linear layers and a non-linear activation in-between. A variant of ReLU, Gaussian Error Linear Unit (GELU) is used, which was the origianl option in GPT-2.
```

逐行解释：

- 第 1 行 `## T4. FeedForward and LayerNorm`
  - 任务 4：介绍 Transformer block 中 attention 之外的另外两个重要组件。
- 空行
  - 用于分段。
- 第 3 行
  - 告诉你接下来要实现 `FeedForward` 和 `LayerNorm`。
  - 这两个模块在 Transformer block 中和 attention 一样重要。
- 空行
  - 用于分段。
- 第 5 行
  - 解释 `FeedForward` 是一个两层线性层加中间非线性激活的小网络。
  - 本 lab 里选择 GELU 作为激活函数，因为 GPT-2 原始实现就是这样做的。

## Cell 19 | Code

原文：

```python
# Gaussian Error Linear Unit (GELU) activation
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
```

逐行解释：

- 第 1 行
  - 注释，说明下面定义的是 GELU 激活函数。
- 第 2 行 `class GELU(nn.Module):`
  - 定义一个自定义激活函数类。
- 第 3 行 `def __init__(self):`
  - 构造函数。
- 第 4 行 `super().__init__()`
  - 初始化父类。
- 空行
  - 用于分段。
- 第 6 行 `def forward(self, x):`
  - 定义前向传播。
- 第 7-10 行 `return ...`
  - 这是 GELU 的常见近似公式。
  - `0.5 * x * (1 + tanh(...))` 会让激活函数比 ReLU 更平滑。
  - `torch.pow(x, 3)` 计算三次项。
  - `torch.sqrt(torch.tensor(2.0 / torch.pi))` 是公式中的常数部分。

## Cell 20 | Code

原文：

```python
import matplotlib.pyplot as plt

gelu, relu = GELU(), nn.ReLU()

# Some sample data
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)

plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

plt.tight_layout()
plt.show()
```

逐行解释：

- 第 1 行 `import matplotlib.pyplot as plt`
  - 导入画图库，后面画激活函数曲线。
- 空行
  - 用于分段。
- 第 3 行 `gelu, relu = GELU(), nn.ReLU()`
  - 分别创建 GELU 和 ReLU 实例。
- 空行
  - 用于分段。
- 第 5 行 `# Some sample data`
  - 注释，说明下面生成示例输入。
- 第 6 行 `x = torch.linspace(-3, 3, 100)`
  - 在区间 `[-3, 3]` 上均匀取 100 个点。
- 第 7 行 `y_gelu, y_relu = gelu(x), relu(x)`
  - 分别把这些点送进 GELU 和 ReLU，得到两条曲线的纵坐标。
- 空行
  - 用于分段。
- 第 9 行 `plt.figure(figsize=(8, 3))`
  - 创建画布，宽 8、高 3。
- 第 10 行 `for i, (y, label) in enumerate(...):`
  - 用循环分别画两张子图。
- 第 11 行 `plt.subplot(1, 2, i)`
  - 把当前图放到 1 行 2 列布局中的第 `i` 个位置。
- 第 12 行 `plt.plot(x, y)`
  - 画曲线。
- 第 13 行 `plt.title(f"{label} activation function")`
  - 设置子图标题。
- 第 14 行 `plt.xlabel("x")`
  - x 轴标签。
- 第 15 行 `plt.ylabel(f"{label}(x)")`
  - y 轴标签。
- 第 16 行 `plt.grid(True)`
  - 打开网格，方便观察曲线形状。
- 空行
  - 用于分段。
- 第 18 行 `plt.tight_layout()`
  - 自动调整子图间距，防止重叠。
- 第 19 行 `plt.show()`
  - 显示图像。

## Cell 21 | Code

原文：

```python
# FeedForward
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
```

逐行解释：

- 第 1 行
  - 注释，说明下面定义的是前馈网络模块。
- 第 2 行 `class FeedForward(nn.Module):`
  - 定义一个继承 `nn.Module` 的前馈网络类。
- 第 3 行 `def __init__(self, cfg):`
  - 构造函数，接收配置字典。
- 第 4 行 `super().__init__()`
  - 初始化父类。
- 第 5-9 行 `self.layers = nn.Sequential(...)`
  - 用 `Sequential` 串起整个前馈网络。
  - 第 6 行把维度从 `emb_dim` 扩展到 `4 * emb_dim`。
  - 第 7 行插入 GELU 非线性。
  - 第 8 行再把维度投影回 `emb_dim`。
  - 这是 GPT/Transformer 里很常见的 FFN 结构。
- 空行
  - 用于分段。
- 第 11 行 `def forward(self, x):`
  - 定义前向传播。
- 第 12 行 `return self.layers(x)`
  - 直接把输入送进这个小网络并返回结果。

## Cell 22 | Code

原文：

```python
# Test FeedForward
ffn = FeedForward(GPT_CONFIG_124M)

x = torch.rand(2, 3, 768) # input shape: [batch_size, num_token, emb_size]
out = ffn(x)
print(out.shape)
```

逐行解释：

- 第 1 行
  - 注释，说明下面开始测试 FFN。
- 第 2 行 `ffn = FeedForward(GPT_CONFIG_124M)`
  - 用 GPT 配置实例化前馈网络。
- 空行
  - 用于分段。
- 第 4 行 `x = torch.rand(2, 3, 768)`
  - 构造一个随机输入。
  - `2` 是 batch size，`3` 是 token 数量，`768` 是 embedding 维度。
- 第 5 行 `out = ffn(x)`
  - 把输入送进前馈网络。
- 第 6 行 `print(out.shape)`
  - 打印输出 shape。
  - FFN 不改变最后的 embedding 维度，所以输出仍然是 `(2, 3, 768)`。

## Cell 23 | Markdown

原文：

```markdown
Next, we implement `LayerNorm`, which centers the activations from a neural network layer around mean of 0, and normalized variance to 1.
```

逐行解释：

- 第 1 行
  - 介绍下一部分：LayerNorm。
  - 它的核心作用是让某一层的激活在最后一个维度上均值接近 0、方差接近 1。

## Cell 24 | Code

原文：

```python
# Example
torch.manual_seed(0)

x = torch.randn(2, 5) # 2 examples with 5 dimensions each
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(x)

print("Output before normalization:")
print(out)
```

逐行解释：

- 第 1 行
  - 注释，说明下面给出一个 LayerNorm 之前的示例。
- 第 2 行 `torch.manual_seed(0)`
  - 固定随机种子，方便复现结果。
- 空行
  - 用于分段。
- 第 4 行 `x = torch.randn(2, 5)`
  - 创建 2 个样本，每个样本 5 维。
- 第 5 行 `layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())`
  - 定义一个简单的小网络：线性层后接 ReLU。
- 第 6 行 `out = layer(x)`
  - 把输入送入小网络，得到一组激活值。
- 空行
  - 用于分段。
- 第 8-9 行
  - 打印归一化前的输出。
  - 这一步是为了后面观察 LayerNorm 的效果。

## Cell 25 | Code

原文：

```python
# Compute and means and vars for the outputs
means = out.mean(dim=-1, keepdim=True)
vars = out.var(dim=-1, keepdim=True)

print(f"Means: {means}")
print(f"Vars: {vars}")
```

逐行解释：

- 第 1 行
  - 注释，说明下面计算每个样本的均值和方差。
- 第 2 行 `means = out.mean(dim=-1, keepdim=True)`
  - 沿最后一个维度计算均值。
  - `keepdim=True` 保留维度，便于后面做广播减法。
- 第 3 行 `vars = out.var(dim=-1, keepdim=True)`
  - 同样沿最后一个维度计算方差。
- 空行
  - 用于分段。
- 第 5-6 行
  - 打印均值和方差，看看归一化之前的数据分布。

## Cell 26 | Code

原文：

```python
# Subtracting the means and dividing by the square root of vars
out_norm = (out - means) / torch.sqrt(vars + 1e-5)

print("Output after normalization:")
print(out_norm)
```

逐行解释：

- 第 1 行
  - 注释，说明下面开始做标准化。
- 第 2 行 `out_norm = (out - means) / torch.sqrt(vars + 1e-5)`
  - 先减去均值，再除以标准差。
  - `1e-5` 是一个很小的数，用来防止除零。
- 空行
  - 用于分段。
- 第 4-5 行
  - 打印归一化后的结果。
  - 这一步是手工演示 LayerNorm 的核心思想。

## Cell 27 | Code

原文：

```python
# Examine the new means and vars
new_means = out_norm.mean(dim=-1, keepdim=True)
new_vars = out_norm.var(dim=-1, keepdim=True)

torch.set_printoptions(sci_mode=False) # disable scientific notation for readability
print(f"New means: {new_means}")
print(f"New vars: {new_vars}")
```

逐行解释：

- 第 1 行
  - 注释，说明下面检查归一化后的均值和方差。
- 第 2 行 `new_means = out_norm.mean(dim=-1, keepdim=True)`
  - 重新计算均值。
- 第 3 行 `new_vars = out_norm.var(dim=-1, keepdim=True)`
  - 重新计算方差。
- 空行
  - 用于分段。
- 第 5 行 `torch.set_printoptions(sci_mode=False)`
  - 关闭科学计数法显示，方便直接阅读数值。
- 第 6-7 行 `print`
  - 打印新的均值和方差。
  - 你会看到均值接近 0，方差接近 1。

## Cell 28 | Markdown

原文：

```markdown
In addition to the above-demonstrated operations, `LayerNrom` introduces two additional learnable parameters `scale` and `shift`, to automatically adjust during training, i.e., learning appropriate scaling and shifting that best suit the data.

The way to add learnable parameter to the computational graph in PyTorch is to use `nn.Parameter`.
```

逐行解释：

- 第 1 行
  - 说明真正的 LayerNorm 不只是“减均值除方差”。
  - 它还带两个可学习参数：`scale` 和 `shift`。
  - 它们让模型在归一化之后还能自己学习合适的缩放和平移。
- 空行
  - 用于分段。
- 第 3 行
  - 提醒你在 PyTorch 里，如果某个张量要被训练，就应该用 `nn.Parameter` 包起来。

## Cell 29 | Code

原文：

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # unbiased=False, => biased estimator of variance
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

逐行解释：

- 第 1 行 `class LayerNorm(nn.Module):`
  - 定义自定义 LayerNorm 模块。
- 第 2 行 `def __init__(self, emb_dim):`
  - 构造函数，接收 embedding 维度。
- 第 3 行 `super().__init__()`
  - 初始化父类。
- 第 4 行 `self.eps = 1e-5`
  - 保存一个小常数，避免除零。
- 第 5 行 `self.scale = nn.Parameter(torch.ones(emb_dim))`
  - 初始化可学习缩放参数 `gamma`，初始值全为 1。
- 第 6 行 `self.shift = nn.Parameter(torch.zeros(emb_dim))`
  - 初始化可学习平移参数 `beta`，初始值全为 0。
- 空行
  - 用于分段。
- 第 8 行 `def forward(self, x):`
  - 定义前向传播。
- 第 9 行 `mean = x.mean(dim=-1, keepdim=True)`
  - 沿最后一维计算均值。
- 第 10 行 `var = x.var(dim=-1, keepdim=True, unbiased=False)`
  - 计算方差。
  - `unbiased=False` 表示使用有偏估计，这也是很多深度学习实现里的常见写法。
- 第 11 行 `norm_x = (x - mean) / torch.sqrt(var + self.eps)`
  - 先中心化，再标准化。
- 第 12 行 `return self.scale * norm_x + self.shift`
  - 最后进行可学习的缩放和平移。

## Cell 30 | Code

原文：

```python
# Test LayerNorm
ln = LayerNorm(emb_dim=6)
out_ln = ln(out)

means = out_ln.mean(dim=-1, keepdim=True)
vars = out_ln.var(dim=-1, unbiased=False, keepdim=True)

print(f"Means: {means}")
print(f"Vars: {vars}")
```

逐行解释：

- 第 1 行
  - 注释，说明下面测试自定义 LayerNorm。
- 第 2 行 `ln = LayerNorm(emb_dim=6)`
  - 实例化一个作用于 6 维向量的 LayerNorm。
- 第 3 行 `out_ln = ln(out)`
  - 把前面的小网络输出送进 LayerNorm。
- 空行
  - 用于分段。
- 第 5-6 行
  - 再次计算归一化后的均值和方差。
  - 这里使用 `unbiased=False` 和模块内部保持一致。
- 空行
  - 用于分段。
- 第 8-9 行
  - 打印结果，验证均值接近 0、方差接近 1。

## Cell 31 | Markdown

原文：

```markdown
The final piece of the jigsaw is **shortcut**, or residual connection.

A shortcut connection creates an alternative shorter path for gradients to flow through the network, which mitigates vanishing gradient problems.

It is achieved by adding the output from an earlier layer to the output from a later layer, skipping one or more layers in between.
```

逐行解释：

- 第 1 行
  - 说明下一个模块是 shortcut，也叫 residual connection。
- 空行
  - 用于分段。
- 第 3 行
  - 解释 residual connection 的作用：给梯度提供一条更短的传播路径。
  - 这样可以减轻深层网络中的梯度消失问题。
- 空行
  - 用于分段。
- 第 5 行
  - 解释实现方式：把较早层的输出加到较晚层的输出上。
  - 本质上就是一种跳连结构。

## Cell 32 | Code

原文：

```python
# Toy Deep Neural Network to demonstrate shortcut connection

class ToyDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x) # compute the output of current layer
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output # apply shortcut
            else:
                x = layer_output
        return x

# Helper function to print values of gradients
def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])
    # Calculate loss based on how close the target and output are
    loss = nn.MSELoss()
    loss = loss(output, target)
    loss.backward() # backprop
    # Print
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
```

逐行解释：

- 第 1 行
  - 注释，说明下面定义一个玩具深层网络来演示 shortcut 的作用。
- 空行
  - 用于分段。
- 第 3 行 `class ToyDeepNeuralNetwork(nn.Module):`
  - 定义一个教学用的深层网络。
- 第 4 行 `def __init__(self, layer_sizes, use_shortcut):`
  - 构造函数接收层大小列表和是否启用 shortcut。
- 第 5 行 `super().__init__()`
  - 初始化父类。
- 第 6 行 `self.use_shortcut = use_shortcut`
  - 保存一个布尔开关，决定 forward 时是否加 shortcut。
- 第 7-13 行 `self.layers = nn.ModuleList([...])`
  - 用 `ModuleList` 保存 5 个顺序模块。
  - 每个顺序模块都是 `Linear + GELU`。
  - `layer_sizes` 决定每层输入输出维度。
- 空行
  - 用于分段。
- 第 15 行 `def forward(self, x):`
  - 定义前向传播。
- 第 16 行 `for layer in self.layers:`
  - 逐层遍历网络。
- 第 17 行 `layer_output = layer(x)`
  - 计算当前层的输出。
- 第 18 行 `if self.use_shortcut and x.shape == layer_output.shape:`
  - 只有在启用了 shortcut 且输入输出 shape 一致时，才允许相加。
  - 这是因为残差相加要求维度匹配。
- 第 19 行 `x = x + layer_output`
  - 应用 shortcut。
- 第 20-21 行 `else: x = layer_output`
  - 如果不能 shortcut，就正常把当前层输出作为下一层输入。
- 第 22 行 `return x`
  - 返回最终输出。
- 空行
  - 用于分段。
- 第 24 行
  - 注释，说明下面定义辅助函数，查看梯度大小。
- 第 25 行 `def print_gradients(model, x):`
  - 定义一个函数，用来比较不同模型的梯度传播情况。
- 第 26 行 `output = model(x)`
  - 先做前向传播。
- 第 27 行 `target = torch.tensor([[0.]])`
  - 构造一个目标输出。
- 第 28 行
  - 注释，说明后面用 loss 比较 output 和 target。
- 第 29 行 `loss = nn.MSELoss()`
  - 创建均方误差损失函数。
- 第 30 行 `loss = loss(output, target)`
  - 计算损失值。
- 第 31 行 `loss.backward()`
  - 反向传播，计算每层参数梯度。
- 第 32 行
  - 注释，说明下面打印梯度。
- 第 33 行 `for name, param in model.named_parameters():`
  - 遍历模型所有参数。
- 第 34 行 `if 'weight' in name:`
  - 只关注权重参数，不打印 bias。
- 第 35 行 `print(...)`
  - 打印每个权重矩阵梯度绝对值的平均值。
  - 这样可以直观看到梯度是否消失。

## Cell 33 | Code

原文：

```python
layer_sizes = [3, 3, 3, 3, 3, 1]

sample_input = torch.tensor([[1., 0., -1.]])

torch.manual_seed(123)
model_without_shortcut = ToyDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)
print_gradients(model_without_shortcut, sample_input)
```

逐行解释：

- 第 1 行 `layer_sizes = [3, 3, 3, 3, 3, 1]`
  - 定义网络每层的维度。
  - 前四层都维持 3 维，最后映射到 1 维输出。
- 空行
  - 用于分段。
- 第 3 行 `sample_input = torch.tensor([[1., 0., -1.]])`
  - 定义一个简单输入样本。
- 空行
  - 用于分段。
- 第 5 行 `torch.manual_seed(123)`
  - 固定随机种子，使模型初始化可复现。
- 第 6-8 行
  - 创建一个不使用 shortcut 的模型。
- 第 9 行 `print_gradients(model_without_shortcut, sample_input)`
  - 打印该模型每层权重的梯度均值。
  - 用于后面和有 shortcut 的情况比较。

## Cell 34 | Code

原文：

```python
torch.manual_seed(123)
model_with_shortcut = ToyDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
)
print_gradients(model_with_shortcut, sample_input)
```

逐行解释：

- 第 1 行 `torch.manual_seed(123)`
  - 使用同一个随机种子，确保两个模型初始化一致。
  - 这样比较才公平。
- 第 2-4 行
  - 创建启用 shortcut 的模型。
- 第 5 行 `print_gradients(model_with_shortcut, sample_input)`
  - 打印有 shortcut 时的梯度均值。
  - 你会看到前层梯度通常更大，说明梯度传播更顺畅。

## Cell 35 | Markdown

原文：

```markdown
As we can see shorcut connections prevent gradients from vanishing.

---

Finally, let's combine `MultiHeadAttention`, `FeedForward`, `LayerNorm` and shortcut together to build a `TransformerBlock` module.

Here, `MultiHeadAttention` is implemented differently from the Toy MHA in order to be compatible.
```

逐行解释：

- 第 1 行
  - 总结上面实验现象：shortcut 有助于缓解梯度消失。
- 空行
  - 用于分段。
- 第 3 行 `---`
  - Markdown 分隔线，表示开始新的部分。
- 空行
  - 用于分段。
- 第 5 行
  - 告诉你接下来要把前面各个组件组合起来，构成真正的 TransformerBlock。
- 空行
  - 用于分段。
- 第 7 行
  - 提醒你：接下来用的 `MultiHeadAttention` 来自 `utils.py`，不是刚才那个教学版 Toy 实现。

## Cell 36 | Code

原文：

```python
from utils import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"], 
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x
```

逐行解释：

- 第 1 行 `from utils import MultiHeadAttention`
  - 从本地 `utils.py` 导入真正可兼容 batch 输入的多头注意力实现。
- 空行
  - 用于分段。
- 第 3 行 `class TransformerBlock(nn.Module):`
  - 定义 Transformer block。
- 第 4 行 `def __init__(self, cfg):`
  - 构造函数，接收配置。
- 第 5 行 `super().__init__()`
  - 初始化父类。
- 第 6-13 行 `self.att = MultiHeadAttention(...)`
  - 创建多头注意力子模块。
  - `d_in` 和 `d_out` 都是 embedding 维度。
  - `context_length` 用于构造因果 mask。
  - `num_heads` 指头数。
  - `dropout` 和 `qkv_bias` 也是标准超参数。
- 第 14 行 `self.ff = FeedForward(cfg)`
  - 创建前馈网络模块。
- 第 15-16 行
  - 创建两个 LayerNorm。
  - 一个用于 attention 前，一个用于 FFN 前。
- 第 17 行 `self.drop_shortcut = nn.Dropout(cfg["drop_rate"])`
  - 创建 dropout，用在 residual 分支前。
- 空行
  - 用于分段。
- 第 19 行 `def forward(self, x):`
  - 定义前向传播。
- 第 20 行注释
  - 说明下面是 attention 分支的残差连接。
- 第 21 行 `shortcut = x`
  - 先保存原输入，后面做残差相加。
- 第 22 行 `x = self.norm1(x)`
  - 先做 LayerNorm。
  - 这属于 pre-norm 结构。
- 第 23 行 `x = self.att(x)`
  - 经过多头注意力。
- 第 24 行 `x = self.drop_shortcut(x)`
  - 对子模块输出做 dropout。
- 第 25 行 `x = x + shortcut`
  - 加回原输入，形成残差连接。
- 空行
  - 用于分段。
- 第 27 行注释
  - 说明下面是 FFN 分支的残差连接。
- 第 28 行 `shortcut = x`
  - 再次保存当前输入。
- 第 29 行 `x = self.norm2(x)`
  - 第二个 LayerNorm。
- 第 30 行 `x = self.ff(x)`
  - 经过前馈网络。
- 第 31 行 `x = self.drop_shortcut(x)`
  - 再做 dropout。
- 第 32 行 `x = x + shortcut`
  - 再次加回残差。
- 空行
  - 用于分段。
- 第 34 行 `return x`
  - 返回 block 输出。

## Cell 37 | Code

原文：

```python
torch.manual_seed(123)

x = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
```

逐行解释：

- 第 1 行 `torch.manual_seed(123)`
  - 固定随机种子。
- 空行
  - 用于分段。
- 第 3 行 `x = torch.rand(2, 4, 768)`
  - 构造 batch 输入。
  - `2` 是 batch size，`4` 是 token 数，`768` 是 embedding 维度。
- 第 4 行 `block = TransformerBlock(GPT_CONFIG_124M)`
  - 实例化一个 Transformer block。
- 第 5 行 `output = block(x)`
  - 执行前向传播。
- 空行
  - 用于分段。
- 第 7-8 行 `print`
  - 打印输入输出 shape。
  - 重点是验证 block 不会改变张量 shape，输入输出都应为 `(2, 4, 768)`。

## Cell 38 | Markdown

原文：

```markdown
## T5. Overview of a Transformer-based GPT-like Language Model

Generative Pretrained Transformer (GPT) is a decoder-only transformer-based architecture for language modeling.

Here we lay out the necessary components of building such a model.

**Notes**:
- In addition to the `TranformerBlock` we just implemented above, a language modeling head submodule `out_head` is added. It maps the embedding outputs from the final layer to logits of `vocab_size`.
- Input token IDs are converted to embeddings by `tok_emb` and added with an absolute positional embedding `pos_emb`.
```

逐行解释：

- 第 1 行
  - 任务 5：概览 GPT 类语言模型的整体结构。
- 空行
  - 用于分段。
- 第 3 行
  - 解释 GPT 是 decoder-only Transformer。
  - 它主要用于语言建模，也就是根据前文预测后文。
- 空行
  - 用于分段。
- 第 5 行
  - 说明下面会把构建 GPT 模型所需的组件串起来。
- 空行
  - 用于分段。
- 第 7 行 `**Notes**:`
  - 开始列出实现备注。
- 第 8 行
  - 说明除了 Transformer blocks 之外，还需要一个输出头 `out_head`。
  - 它负责把最后的隐藏状态映射到词表大小的 logits。
- 第 9 行
  - 说明模型输入不是直接用 token id，而是先通过 `tok_emb` 变成词向量，再加上位置向量 `pos_emb`。

## Cell 39 | Code

原文：

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Use a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        # Use a placeholder for LayerNorm
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

逐行解释：

- 第 1 行 `class GPTModel(nn.Module):`
  - 定义一个简化版 GPT 模型。
- 第 2 行 `def __init__(self, cfg):`
  - 构造函数。
- 第 3 行 `super().__init__()`
  - 初始化父类。
- 第 4 行 `self.tok_emb = nn.Embedding(...)`
  - 创建 token embedding 层。
  - 它把 token id 映射成 dense 向量。
- 第 5 行 `self.pos_emb = nn.Embedding(...)`
  - 创建位置 embedding 层。
  - 它让模型知道每个 token 在序列中的位置。
- 第 6 行 `self.drop_emb = nn.Dropout(cfg["drop_rate"])`
  - 在 embedding 后做 dropout。
- 空行
  - 用于分段。
- 第 8 行注释
  - 说明下面将堆叠多个 Transformer block。
- 第 9-10 行 `self.trf_blocks = nn.Sequential(...)`
  - 用列表推导式创建 `n_layers` 个 block。
  - 再用 `Sequential` 按顺序串起来。
- 空行
  - 用于分段。
- 第 12 行注释
  - 说明后面创建最终 LayerNorm。
- 第 13 行 `self.final_norm = LayerNorm(cfg["emb_dim"])`
  - 在所有 Transformer blocks 后做最终归一化。
- 第 14-16 行 `self.out_head = nn.Linear(...)`
  - 定义输出层。
  - 把每个位置的隐藏向量映射到整个词表大小的 logits。
  - `bias=False` 是 GPT 类模型中常见设置。
- 空行
  - 用于分段。
- 第 18 行 `def forward(self, in_idx):`
  - 定义前向传播，输入是 token id 张量。
- 第 19 行 `batch_size, seq_len = in_idx.shape`
  - 取出 batch 大小和序列长度。
- 第 20 行 `tok_embeds = self.tok_emb(in_idx)`
  - 把 token id 查表变成 token embeddings。
- 第 21 行 `pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))`
  - 生成位置索引 `0,1,...,seq_len-1`。
  - 再查位置 embedding 表，得到位置向量。
- 第 22 行 `x = tok_embeds + pos_embeds`
  - 把 token embedding 和位置 embedding 相加。
- 第 23 行 `x = self.drop_emb(x)`
  - 做 dropout。
- 第 24 行 `x = self.trf_blocks(x)`
  - 送入多个 Transformer blocks。
- 第 25 行 `x = self.final_norm(x)`
  - 做最终 LayerNorm。
- 第 26 行 `logits = self.out_head(x)`
  - 映射到词表维度，得到 logits。
- 第 27 行 `return logits`
  - 返回结果。

## Cell 40 | Markdown

原文：

```markdown
Let's prepare some data and test the model
```

逐行解释：

- 第 1 行
  - 说明下面要准备一小段文本，测试刚刚定义好的 GPTModel。

## Cell 41 | Code

原文：

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)
```

逐行解释：

- 第 1 行 `import tiktoken`
  - 导入 tokenizer 库。
- 空行
  - 用于分段。
- 第 3 行 `tokenizer = tiktoken.get_encoding("gpt2")`
  - 获取 GPT-2 使用的编码器。
- 空行
  - 用于分段。
- 第 5 行 `batch = []`
  - 初始化一个列表，用来收集多个句子的 token 序列。
- 第 6 行 `txt1 = "Every effort moves you"`
  - 定义第一个句子。
- 第 7 行 `txt2 = "Every day holds a"`
  - 定义第二个句子。
- 空行
  - 用于分段。
- 第 9 行 `batch.append(torch.tensor(tokenizer.encode(txt1)))`
  - 先把第一个句子编码成 token id 列表，再转成 tensor，加到 batch 里。
- 第 10 行 `batch.append(torch.tensor(tokenizer.encode(txt2)))`
  - 对第二个句子做同样的事。
- 第 11 行 `batch = torch.stack(batch, dim=0)`
  - 把两个一维 token 序列堆成一个二维 batch。
  - 结果 shape 是 `(2, 4)`。
- 第 12 行 `print(batch)`
  - 打印 token id，确认编码后的内容。

## Cell 42 | Markdown

原文：

```markdown
The model takes the bach of tokenized IDs and produces some logits. Watch the size of the output.
```

逐行解释：

- 第 1 行
  - 说明下一步会把 token id batch 送进模型。
  - 重点观察输出 shape，而不是具体数值。

## Cell 43 | Code

原文：

```python
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)
```

逐行解释：

- 第 1 行 `torch.manual_seed(123)`
  - 固定随机种子，确保模型初始化可复现。
- 第 2 行 `model = GPTModel(GPT_CONFIG_124M)`
  - 用 GPT 配置实例化模型。
- 空行
  - 用于分段。
- 第 4 行 `out = model(batch)`
  - 把 token id batch 输入模型，得到输出 logits。
- 第 5 行 `print("Input batch:\n", batch)`
  - 打印输入 batch，方便和输出 shape 对照。
- 第 6 行 `print("\nOutput shape:", out.shape)`
  - 打印输出 shape。
  - 对 GPT 来说，输出应该是 `(batch_size, seq_len, vocab_size)`。
- 第 7 行 `print(out)`
  - 打印完整 logits 张量。
  - 每个位置都会对应一个长度为词表大小的预测分数向量。

---

## 最后总结

如果把整本 lab 的逻辑压缩成一条主线，可以这样理解：

1. 先在单头场景下手写 `Q/K/V -> score -> softmax -> output`。
2. 再加入 causal mask，让模型不能偷看未来 token。
3. 再把单头扩展成多头，核心难点是 `view/permute/repeat/contiguous`。
4. 接着补齐 Transformer block 的另外几个核心组件：FFN、LayerNorm、Residual。
5. 最后把这些模块堆起来，构成一个 GPT-like language model。

你如果能顺着这份文档把每个 cell 的 shape 变化和每个模块的职责复述出来，这节 lab 就已经理解到位了。

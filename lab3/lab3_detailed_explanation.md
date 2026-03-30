# Lab3 Detailed Explanation

本文档逐段解释 [`lab3_w2v_completed.ipynb`](/D:/OneDrive/Study/2026%20Spring/Natural%20Language%20Processing/lab3/lab3_w2v_completed.ipynb) 和 [`utils.py`](/D:/OneDrive/Study/2026%20Spring/Natural%20Language%20Processing/lab3/utils.py) 的内容。

说明：

- 讲解对象以完成版 notebook 为准，因为它包含了已经补全的 `T1-T4`。
- 原模板 [`lab3_w2v.ipynb`](/D:/OneDrive/Study/2026%20Spring/Natural%20Language%20Processing/lab3/lab3_w2v.ipynb) 的文字内容和结构基本一致，只是待填空没有实现。
- `03-wv.pdf` 是课程讲义，`lunyu_20chapters.txt` 是实验语料，它们在本讲解中只说明作用，不逐页/逐句展开。


## 0. 文件作用总览

- [`lab3_w2v_completed.ipynb`](/D:/OneDrive/Study/2026%20Spring/Natural%20Language%20Processing/lab3/lab3_w2v_completed.ipynb)
  这是实验主文件，按步骤带你实现 skip-gram + negative sampling。
- [`utils.py`](/D:/OneDrive/Study/2026%20Spring/Natural%20Language%20Processing/lab3/utils.py)
  这是辅助类 `CorpusReader`，负责读语料、建词表、建负采样表。
- [`lunyu_20chapters.txt`](/D:/OneDrive/Study/2026%20Spring/Natural%20Language%20Processing/lab3/lunyu_20chapters.txt)
  训练数据，中文按字切分。
- [`03-wv.pdf`](/D:/OneDrive/Study/2026%20Spring/Natural%20Language%20Processing/lab3/03-wv.pdf)
  对应课程讲义，讲词向量、word2vec、negative sampling、LSA、analogy 等概念。


## 1. Notebook 逐 Cell 讲解

### Cell 0 [Markdown]

1. `## CS310 Natural Language Processing`
   这是课程标题，说明这个 notebook 属于 CS310 自然语言处理课程。
2. `## Lab 3: Data preparation for implementing word2vec`
   这是实验标题，说明本次 lab 的主题是“为实现 word2vec 做数据准备”。
3. `In this lab, we will practice ...`
   这句是总目标说明：
   本次实验不是直接调用现成包训练，而是亲手实现 word2vec 所需的关键步骤，使用的模型结构是 `skipgram`，训练技巧是 `negative sampling`。


### Cell 1 [Code]

1. `from typing import List`
   导入 `List` 类型标注，后面函数会写成 `words: List[str]` 这种更清晰的声明。
2. `from pprint import pprint`
   导入更好看的打印函数，用于输出复杂列表或元组时排版更清楚。
3. `from utils import CorpusReader`
   从本目录的 `utils.py` 里导入 `CorpusReader` 类，这是后续所有数据读取和负采样的基础。
4. `import torch`
   导入 PyTorch 主库。
5. `import torch.nn as nn`
   导入神经网络模块，后面会用 `nn.Module`、`nn.Embedding`。
6. `import torch.nn.functional as F`
   导入函数式 API，后面会用 `F.logsigmoid`。
7. `import numpy as np`
   导入 NumPy，用于频率表、负采样表、数组堆叠等操作。


### Cell 2 [Code]

1. `# We set min_count=1 to include all words in the corpus`
   注释说明：这里把 `min_count` 设成 1，表示只要在语料里出现过，就进入词表。
2. `corpus = CorpusReader(inputFileName="lunyu_20chapters.txt", min_count=1, lang="zh")`
   真正创建语料对象：
   - 文件是 `lunyu_20chapters.txt`
   - `min_count=1`
   - `lang="zh"` 表示中文按字切分
3. `# corpus = CorpusReader(inputFileName="shakespeare.txt", min_count=3, lang="en")`
   这行被注释掉，说明同一个框架也可以切到英文语料。
4. `# corpus = CorpusReader(inputFileName="ptb.train.txt", min_count=5, lang="en")`
   另一组英文语料示例。

这一格的核心作用是：初始化整个实验的数据入口。


### Cell 3 [Code]

1. `print(corpus.word2id["子"])`
   打印字 `"子"` 的编号，证明词表已经建立。
2. `print(corpus.id2word[1])`
   打印编号 1 对应的字，证明 `id -> word` 的映射也已建立。
3. `print(len(corpus.id2word))`
   打印词表长度。
4. `print(corpus.vocab_size)`
   再打印一次词表大小，和上一行应该一致。
5. 后面几行被注释掉的 `print(...)`
   是英文语料时的示例查询，拿来查 `man / woman / king / queen` 等词是否进词表。

这一格的作用是做最基础的数据 sanity check。


### Cell 4 [Markdown]

1. `### Efficient way for negative sampling`
   小节标题：高效负采样。
2. `In utils.CorpusReader class, we have implemented ...`
   说明 `utils.py` 中已经实现了 `initTableNegatives`。
3. `It creates a list of words (self.negatives) with a size of 1e8`
   说明负采样表的规模非常大，是 `1e8`。
4. `The purpose is to have a large value so that it scales up ...`
   解释为什么要这么大：为了让抽样近似真实概率分布，也为了大语料时依然有效。
5. `The list contains the integer index ... proportional to ... 0.75`
   说明抽样概率不是按原始词频，而是按词频的 `0.75` 次幂。

这段文字是在给你解释 `utils.py` 中已经帮你实现好的关键优化。


### Cell 5 [Code]

1. `# This is a simulation ...`
   注释说明：下面只是一个小例子，帮助你理解负采样表怎么建，不是正式语料。
2. `word_frequency = {"a": 1, "b": 2, "c": 3, "d": 4}`
   人工构造一个四个词的小词频表。
3. 后面四行关于 `scaled sum` 和 `scaled probability`
   直接把数学计算结果写出来，帮助你对照代码理解：
   - 先把频率做 `0.75` 次幂
   - 再归一化成概率
4. `def initTableNegatives():`
   定义一个玩具版负采样表构造函数。
5. `pow_frequency = np.array(list(word_frequency.values())) ** 0.75`
   对所有频率做 `0.75` 次幂。
6. `words_pow = sum(pow_frequency)`
   把所有加权频率求和，作为归一化分母。
7. `ratio = pow_frequency / words_pow`
   得到每个词的采样概率。
8. `count = np.round(ratio * CorpusReader.NEGATIVE_TABLE_SIZE)`
   把概率乘以表的总长度，得到每个词应该复制多少次。
9. `negatives = []`
   先建空列表。
10. `for wid, c in enumerate(count):`
    逐个词处理，`wid` 是词编号，`c` 是这个词要复制的次数。
11. `negatives += [wid] * int(c)`
    把词编号重复很多次加入列表。
12. `negatives = np.array(negatives)`
    把 Python 列表转成 NumPy 数组，便于后面快速切片。
13. `np.random.shuffle(negatives)`
    打乱顺序，这样后续直接切一段就相当于随机采样。
14. `return negatives`
    返回构好的负采样表。
15. `negatives = initTableNegatives()`
    运行函数，得到结果。


### Cell 6 [Code]

1. `print(len(negatives))`
   看负采样表长度是不是接近目标值。
2. `print(set(negatives))`
   看表里是否包含四个编号 `0,1,2,3`。
3. `print(np.sum(negatives == 0) / len(negatives))`
   统计编号 0 的比例，看是否接近理论概率。
4. 同样的方法统计 1、2、3 的比例。

这格的核心目标是验证：
负采样表中每个词出现的比例，确实和 `freq^0.75` 的归一化概率一致。


### Cell 7 [Markdown]

1. `Next, the getNegatives method returns the negative samples ...`
   说明真正取负样本时，不是每次重新采样，而是从大数组里切一段。
2. `If the segment contains the target word ...`
   如果这段里恰好包含中心词本身，就扔掉重取。

这段文字是在解释 `getNegatives()` 的工作原理。


### Cell 8 [Code]

1. `# Test some examples`
   注释：做个小测试。
2. `corpus.getNegatives(target=1, size=5)`
   从负采样表里取 5 个负样本，并要求它们不等于目标词 `1`。


### Cell 9 [Markdown]

1. `## T1. Generate data for training`
   任务 1 标题：生成训练数据。
2. `Now we are going to implement the sliding window ...`
   说明这部分要你实现滑动窗口。
3. `It takes a list of words as input ...`
   输入是一个词列表。
4. `For each center word, both the left and right window_size words ...`
   对每个中心词，要同时取左右窗口内的上下文词。
5. `Call corpus.getNegatives ...`
   对每个中心词生成负样本。


### Cell 10 [Code]

1. `def generate_data(words: List[str], window_size: int, k: int, corpus: CorpusReader):`
   定义训练样本生成函数。
2. `""" Generate the training data ... """`
   函数说明：
   - `words` 是当前句子的词列表
   - `window_size` 是滑窗大小
   - `k` 是负样本数
   - `corpus` 提供词表和负采样功能
3. `word_ids = []`
   准备把词转成编号。
4. `for word in words:`
   遍历输入的每个词。
5. `if word in corpus.word2id:`
   只保留词表中存在的词，避免 OOV 报错。
6. `word_ids.append(corpus.word2id[word])`
   把词转换成 id。
7. `for center_pos, center_id in enumerate(word_ids):`
   枚举每个位置，把它当作中心词。
8. `left = max(0, center_pos - window_size)`
   左边界不能越界，所以和 0 比较取最大值。
9. `right = min(len(word_ids), center_pos + window_size + 1)`
   右边界不能超过句子长度，所以和总长度比较取最小值。
10. `for outside_pos in range(left, right):`
    遍历窗口中的所有位置。
11. `if outside_pos == center_pos:`
    如果是中心词自己，就跳过。
12. `continue`
    真正执行跳过。
13. `outside_id = word_ids[outside_pos]`
    取当前 outside word 的编号。
14. `negative_ids = corpus.getNegatives(center_id, k)`
    为当前中心词取 `k` 个负样本。
15. `yield center_id, outside_id, negative_ids`
    用生成器逐条返回训练样本。

这一格的本质是把一句话变成很多组：

```text
(center, outside, negatives)
```


### Cell 11 [Code]

1. `# Test generate_data`
   注释：开始测 `generate_data`。
2. `text = "学而时习之"`
   选一个很短的测试字符串。
3. `words = list(text)`
   中文按字切分，所以直接变成字符列表。
4. `print('words:', words)`
   打印切分后的词。
5. `print('word ids:', [corpus.word2id[word] for word in words])`
   打印每个字对应的编号。
6. `# first center word is 学`
   注释：说明先看中心词“学”。
7. `print()`
   打印空行，便于阅读。
8. `print(f'When window size is 3 ...')`
   打印“学”的编号。
9. `print(f'the outside words are: ')`
   提示下面要显示它的上下文词。
10. 后三行 `print(...)`
    手工列出窗口内应该出现的三个 outside words。
11. `print()`
    再空一行。
12. `print('output from generate_data:')`
    提示下面是函数输出。
13. `data = list(generate_data(...))`
    真正调用刚写好的函数，并把生成器全部展开成列表。
14. `pprint(data[:3])`
    打印前三个样本，看看是否符合预期。
15. 最后大段注释
    给出参考输出，方便你对照。


### Cell 12 [Markdown]

1. `The above data are not in batch yet.`
   说明目前样本还是一条条的，还没打包成 batch。
2. `We want all center words are batched into a tensor ...`
   目标是把中心词、上下文词、负样本分别打包成张量。
3. `For example, in "学而时习之"...`
   用具体例子说明 batch 长什么样。
4. `The first tensor contains center words ...`
   解释第一个张量是中心词 id。
5. `The second tensor contains the correponding outside words ...`
   解释第二个张量是对应的 outside word id。
6. `The third tensor contains the negative samples ...`
   解释第三个张量是负样本矩阵。
7. `The data type of the tensors is torch.long.`
   强调索引张量必须是 `torch.long`。


### Cell 13 [Code]

1. `def batchify(data: List, batch_size: int):`
   定义打包函数。
2. `""" Group a stream into batches ... """`
   函数说明：
   输入是一串样本，输出是一个个 batch。
3. `assert batch_size < len(data)`
   保证 batch size 不会比数据总量还大。
4. `for i in range(0, len(data), batch_size):`
   每次跨 `batch_size` 个样本取一块。
5. `batch = data[i:i + batch_size]`
   取当前 batch。
6. `if i > len(data) - batch_size:`
   如果已经到了最后一批，而且最后一批不够长，就补齐。
7. `batch = batch + data[:i + batch_size - len(data)]`
   用开头的一些样本拼到末尾，使最后一批也满足固定大小。
8. `center = torch.tensor([item[0] for item in batch], dtype=torch.long)`
   提取所有中心词 id，形成 `(B,)` 张量。
9. `outside = torch.tensor([item[1] for item in batch], dtype=torch.long)`
   提取所有上下文词 id，形成 `(B,)` 张量。
10. `negative = torch.tensor(np.stack([item[2] for item in batch]), dtype=torch.long)`
    把每条样本中的负样本数组堆成 `(B, k)` 张量。
11. `yield center, outside, negative`
    返回这个 batch。


### Cell 14 [Code]

1. `# Test batchify`
   注释：测试打包函数。
2. `text = "学而时习之"`
   用同一条短句。
3. `words = list(text)`
   仍然按字切分。
4. `data = list(generate_data(...))`
   先得到未打包样本。
5. `batches = list(batchify(data, batch_size=4))`
   打成 batch，这里批大小设成 4。
6. `print(batches[0])`
   打印第一个 batch。
7. 后面注释给出了预期格式。


### Cell 15 [Markdown]

1. `## T2. Implement the SkipGram class`
   任务 2 标题：实现模型类。
2. `SkipGram is a subclass of nn.Module`
   说明它是一个 PyTorch 模型。
3. 关于 `__init__` 的两条说明
   解释为什么有两套 embedding：
   - `emb_v` 给中心词
   - `emb_u` 给 outside / negative words
4. 关于 `forward` 的两条说明
   解释前向传播输入什么，以及 loss 是什么公式。
5. `Hint` 部分
   提醒你可以用 `F.logsigmoid`，也提醒你要做数值稳定处理。


### Cell 16 [Code]

1. `class SkipGram(nn.Module):`
   定义模型类。
2. `def __init__(self, vocab_size, emb_size):`
   构造函数，输入词表大小和 embedding 维度。
3. `super(SkipGram, self).__init__()`
   调用父类初始化。
4. `self.vocab_size = vocab_size`
   保存词表大小。
5. `self.emb_size = emb_size`
   保存 embedding 维度。
6. `self.emb_v = nn.Embedding(vocab_size, emb_size, sparse=True)`
   建中心词 embedding 表。
7. `self.emb_u = nn.Embedding(vocab_size, emb_size, sparse=True)`
   建 outside/negative embedding 表。
8. `initrange = 1.0 / self.emb_size`
   定义初始化范围。
9. `nn.init.uniform_(...)`
   用均匀分布初始化 `emb_v`。
10. `nn.init.constant_(..., 0)`
    把 `emb_u` 初始化成全 0。
11. `def forward(self, center, outside, negative):`
    定义前向传播。
12. 文档字符串三行
    说明三个输入的形状：
    - `center`: `(B,)`
    - `outside`: `(B,)`
    - `negative`: `(B, k)`
13. `v_c = self.emb_v(center)`
    查出中心词向量，形状 `(B, D)`。
14. `u_o = self.emb_u(outside)`
    查出正样本上下文向量，形状 `(B, D)`。
15. `u_n = self.emb_u(negative)`
    查出负样本向量，形状 `(B, K, D)`。
16. `positive_score = torch.sum(v_c * u_o, dim=1)`
    逐元素相乘后按维求和，也就是正样本点积。
17. `positive_score = torch.clamp(...)`
    对正样本分数做裁剪，防止数值过大。
18. `positive_loss = F.logsigmoid(positive_score)`
    计算 `log σ(v_c · u_o)`。
19. `negative_score = torch.bmm(u_n, v_c.unsqueeze(2)).squeeze(2)`
    对每个负样本和中心词做批量点积，得到 `(B, K)`。
20. `negative_score = torch.clamp(...)`
    同样做裁剪。
21. `negative_loss = F.logsigmoid(-negative_score).sum(dim=1)`
    计算所有负样本项的和：
    `Σ log σ(-v_c · u_k)`。
22. `loss = -(positive_loss + negative_loss)`
    前面算的是 log-likelihood，这里加负号变成最小化的 loss。
23. `return loss`
    返回每个样本的 loss 向量。
24. `def save_embedding(self, id2word, file_name):`
    定义导出 embedding 的函数。
25. `embedding = self.emb_v.weight.cpu().data.numpy()`
    取中心词 embedding，并转成 NumPy。
26. `with open(file_name, 'w', encoding='utf8') as f:`
    以 UTF-8 打开输出文件。
27. `f.write('%d %d\n' % (len(id2word), self.emb_size))`
    第一行写词表大小和向量维度。
28. `for wid, w in id2word.items():`
    遍历所有词。
29. `e = ' '.join(map(lambda x: str(x), embedding[wid]))`
    把一个向量的所有维度拼成字符串。
30. `f.write('%s %s\n' % (w, e))`
    写成标准的 word2vec 文本格式。


### Cell 17 [Code]

1. `# Test the model`
   注释：开始验证 `forward` 是否正确。
2. `vacob_size = len(corpus.id2word)`
   取词表大小。
3. `emb_size = 32`
   这里特意设成 32 维。
4. `model = SkipGram(vacob_size, emb_size)`
   创建模型。
5. `weight = torch.empty(vacob_size, emb_size)`
   建一个空张量，准备手工赋值。
6. `start_value = 0.01`
   设置起始值。
7. `for i in range(vacob_size):`
   对每一行 embedding 赋值。
8. `weight[i] = start_value + i * 0.01`
   每一行都用规则递增值填满，这样结果可控。
9. `model.emb_v.weight.data.copy_(weight)`
   把人工权重复制到中心词表。
10. `model.emb_u.weight.data.copy_(weight)`
    把同样的权重复制到上下文词表。
11. `center = torch.tensor([0, 1, 2, 3, 4])`
    构造测试中心词。
12. `outside = torch.tensor([0, 1, 2, 3, 4])`
    构造测试 outside word。
13. `negative = torch.tensor([[0,1,2,3,4], ...])`
    构造固定负样本矩阵。
14. `with torch.no_grad():`
    这里只做前向检查，不需要梯度。
15. `loss = model(center, outside, negative)`
    跑前向传播。
16. `print(loss)`
    查看输出是否与预期一致。
17. 最后注释给出参考结果。

这格的意义是：
在一个可控的数值环境里验证公式和代码完全一致。


### Cell 18 [Markdown]

1. `## T3. Train and save`
   任务 3 标题：训练并保存。
2. `Train the SkipGram model ...`
   要求你把模型、数据、优化器组织起来。
3. `Once training is done ...`
   训练完要把 embedding 保存成兼容 `gensim` 的文本格式。


### Cell 19 [Code]

1. `# Some sample code for training and saving embeddings`
   注释：这一格给训练和保存的完整样例。
2. `def train(model, dataloader, optimizer, epochs):`
   定义训练函数。
3. `dataloader = list(dataloader)`
   把输入转成列表，保证多轮 epoch 时可以重复迭代。
4. `model.train()`
   切到训练模式。
5. `device = next(model.parameters()).device`
   取模型所在设备。
6. `loss_history = []`
   用来记录每轮平均 loss。
7. `for epoch in range(epochs):`
   迭代多个 epoch。
8. `total_loss = 0.0`
   累计总 loss。
9. `total_examples = 0`
   累计样本数。
10. `for center, outside, negative in dataloader:`
    遍历每个 batch。
11. `center = center.to(device)`
    把中心词张量移到设备上。
12. `outside = outside.to(device)`
    同理。
13. `negative = negative.to(device)`
    同理。
14. `optimizer.zero_grad()`
    清空上一轮梯度。
15. `loss = model(center, outside, negative).mean()`
    模型返回每个样本的 loss，这里对 batch 求平均。
16. `loss.backward()`
    反向传播。
17. `optimizer.step()`
    更新参数。
18. `batch_examples = center.size(0)`
    当前 batch 的样本数。
19. `total_loss += loss.item() * batch_examples`
    用“平均 loss × batch 大小”恢复总 loss，便于算 epoch 平均值。
20. `total_examples += batch_examples`
    更新总样本数。
21. `avg_loss = total_loss / total_examples`
    得到当前 epoch 平均 loss。
22. `loss_history.append(avg_loss)`
    记录下来。
23. `print(f'Epoch ...')`
    打印训练进度。
24. `return loss_history`
    返回全部 epoch 的 loss。
25. `window_size = 3`
    超参数：窗口大小。
26. `k = 5`
    超参数：每个正样本配 5 个负样本。
27. `batch_size = 64`
    超参数：批大小。
28. `emb_size = 100`
    超参数：词向量维度。
29. `epochs = 5`
    超参数：训练轮数。
30. `learning_rate = 0.025`
    超参数：学习率。
31. `training_data = []`
    准备收集所有训练样本。
32. `for line in open('lunyu_20chapters.txt', encoding='utf8'):`
    逐行读语料。
33. `words = [word for word in line.strip() if word in corpus.word2id]`
    中文按字取出，并只保留词表中的字。
34. `if len(words) < 2:`
    如果一行太短，没法形成上下文，就跳过。
35. `continue`
    真正跳过。
36. `training_data.extend(generate_data(...))`
    把这一行产生的所有样本加入总训练集。
37. `print(f'Num training pairs: ...')`
    打印样本总数。
38. `batches = list(batchify(training_data, batch_size=batch_size))`
    把所有样本打包。
39. `print(f'Num batches: ...')`
    打印 batch 数。
40. `model = SkipGram(corpus.vocab_size, emb_size)`
    创建模型。
41. `optimizer = torch.optim.SparseAdam(...)`
    使用稀疏版 Adam 优化器，适合 sparse embedding。
42. `loss_history = train(...)`
    真正开始训练。
43. `output_file = 'embeddings.txt'`
    指定导出文件名。
44. `model.save_embedding(corpus.id2word, output_file)`
    导出 embedding。
45. `print(f'Saved embeddings ...')`
    打印保存成功信息。
46. `print('Final epoch loss:', loss_history[-1])`
    打印最后一轮 loss。


### Cell 20 [Markdown]

1. `## T4. Play around with the trained embeddings`
   任务 4 标题：玩一玩训练好的词向量。
2. `Install matplotlib, gensim, and scikit-learn first`
   说明这一部分需要额外依赖。
3. 代码块里的 `pip install ...`
   给出安装命令。
4. `Note: On this machine ...`
   说明本机上 `gensim` 在 Python 3.14 下装不上，所以 T4 用 Python 3.12 验证。
5. `You can load the saved word embeddings ...`
   说明两种玩法：
   - 读你自己训练出来的 embedding
   - 读预训练词向量
6. 最后一行给出 `gensim downloader` 文档链接。


### Cell 21 [Code]

1. `def load_embedding_model():`
   定义加载预训练 embedding 的函数。
2. 文档字符串
   说明返回值是预训练向量对象。
3. `import gensim.downloader as api`
   导入 gensim 的下载器。
4. `wv_from_bin = api.load("glove-wiki-gigaword-200")`
   下载并加载 GloVe 200 维词向量。
5. 注释说明也可以试 `word2vec-google-news-300`
   但更大，下载和加载更慢。
6. `print("Loaded vocab size %i" % ...)`
   打印词表大小。
7. `return wv_from_bin`
   返回词向量对象。
8. 最后两行被注释掉的代码
   是查询 gensim 可下载模型列表的提示。


### Cell 22 [Code]

1. 三行横杠注释
   只是为了让 notebook 看起来更清晰。
2. `wv_from_bin = load_embedding_model()`
   真正加载预训练词向量。


### Cell 23 [Markdown]

1. 说明 `gensim` 提供 `most_similar` 方法。
2. 这个方法的本质是做向量运算，再找最近邻。


### Cell 24 [Code]

1. `# Run this cell to answer ...`
   注释说明要做的 analogy 是：
   `man : grandfather :: woman : ?`
2. `pprint(wv_from_bin.most_similar(...))`
   真正调用 `most_similar`：
   - `positive=['woman', 'grandfather']`
   - `negative=['man']`
   等价于计算：
   `v(woman) + v(grandfather) - v(man)`
   再寻找最相似的词。


### Cell 25 [Markdown]

1. 说明你也可以自己设计 analogy 问题。
2. 给了一个经典例子：
   `China : Beijing = Japan : ?`


### Cell 26 [Code]

1. `analogy_tasks = [ ... ]`
   定义三个 analogy 任务。
2. 第一项 `('china : beijing :: japan : ?', ['japan', 'beijing'], ['china'])`
   对应国家和首都关系。
3. 第二项 `('france : paris :: italy : ?', ...)`
   再做一个欧洲首都关系。
4. 第三项 `('man : king :: woman : ?', ...)`
   做性别 + 王室关系。
5. `for prompt, positive, negative in analogy_tasks:`
   逐个任务执行。
6. `print(prompt)`
   先打印题目。
7. `pprint(wv_from_bin.most_similar(...))`
   打印前 5 个最可能答案。
8. `print()`
   空一行，便于阅读。


### Cell 27 [Markdown]

1. 说明下面要做 SVD 可视化。
2. 这里用的是降维思想：把高维词向量投影到二维平面。


### Cell 28 [Code]

1. `import matplotlib.pyplot as plt`
   导入绘图库。
2. `from sklearn.decomposition import TruncatedSVD`
   导入截断奇异值分解。
3. `plot_words = ['man', 'woman', 'king', 'queen', 'prince', 'princess']`
   选一组语义关系清晰的词做可视化。
4. `plot_vectors = np.vstack([wv_from_bin[word] for word in plot_words])`
   取出这些词的向量，并堆成矩阵。
5. `n_components = 2`
   目标维度设成 2。
6. `svd = TruncatedSVD(n_components=n_components, random_state=42)`
   创建降维器，并固定随机种子。
7. `vectors_2d = svd.fit_transform(plot_vectors)`
   把原始高维向量投影到二维。
8. `print(vectors_2d.shape)`
   打印结果形状，应该是 `(6, 2)`。


### Cell 29 [Markdown]

1. 这句只是承上启下：
   “下面看具体可视化效果。”


### Cell 30 [Code]

1. `words = plot_words`
   复用前面选好的词。
2. `plt.figure(figsize=(7, 7))`
   建一个 7×7 的画布。
3. `for i, word in enumerate(words):`
   逐个词画点。
4. `plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1])`
   把第 `i` 个词的二维坐标画成散点。
5. `plt.text(vectors_2d[i, 0] + 0.01, vectors_2d[i, 1] + 0.01, word)`
   在点旁边加标签，并略微偏移避免压在点上。
6. `plt.title('2D projection of selected pretrained word embeddings')`
   设置图标题。
7. `plt.show()`
   显示图像。


## 2. `utils.py` 逐段讲解

### 文件开头

1. `# Adapted from ...`
   说明这份实现改编自一个 PyTorch word2vec 项目。
2. `import numpy as np`
   导入 NumPy，后面大多数概率和表构造都依赖它。


### `class CorpusReader`

1. `class CorpusReader:`
   定义语料读取类。
2. `NEGATIVE_TABLE_SIZE = 1e8`
   设定负采样表的目标大小。


### `__init__`

1. `def __init__(self, inputFileName, min_count:int = 5, lang="zh") :`
   初始化函数，输入：
   - 语料文件名
   - 最低词频阈值
   - 语言类型
2. `self.negatives = []`
   用来存大负采样表。
3. `self.discards = []`
   用来存 subsampling 概率。
4. `self.negpos = 0`
   当前负采样游标位置。
5. `self.word2id = dict()`
   词到编号的映射。
6. `self.id2word = dict()`
   编号到词的映射。
7. `self.token_count = 0`
   总 token 数。
8. `self.word_frequency = dict()`
   每个词的词频。
9. `self.lang = lang`
   保存语言类型。
10. `self.inputFileName = inputFileName`
    保存语料路径。
11. `self.read_words(min_count)`
    先读语料，建立词表。
12. `self.vocab_size = len(self.word2id)`
    记录词表大小。
13. `self.initTableNegatives()`
    建负采样表。
14. `self.initTableDiscards()`
    建 subsampling 概率表。


### `read_words`

1. `word_frequency = dict()`
   先用局部字典统计原始词频。
2. `for line in open(self.inputFileName, encoding="utf8"):`
   按行读取语料。
3. `if self.lang == "zh":`
   如果是中文。
4. `words = list(line.strip())`
   中文直接按字切分。
5. `elif self.lang == "en":`
   如果是英文。
6. `words = line.split()`
   英文按空格切词。
7. `if len(words) > 0:`
   空行就不处理。
8. `for word in words:`
   遍历每个 token。
9. `self.token_count += 1`
   总 token 数加一。
10. `word_frequency[word] = word_frequency.get(word, 0) + 1`
    更新词频。
11. `if self.token_count % 1000000 == 0:`
    如果读到一百万词的整数倍。
12. `print("Read ...")`
    打印进度。
13. `wid = 0`
    准备开始分配词表编号。
14. `for w, c in sorted(word_frequency.items(), key=lambda x: x[1], reverse=True):`
    按词频从高到低排序。
15. `if c < min_count:`
    如果低于阈值，就丢掉。
16. `continue`
    真正跳过。
17. `self.word2id[w] = wid`
    记录词到编号。
18. `self.id2word[wid] = w`
    记录编号到词。
19. `self.word_frequency[wid] = c`
    记录编号对应的词频。
20. `wid += 1`
    下一个词编号加一。
21. `print("Total vocabulary: ...")`
    输出最终词表大小。


### `initTableDiscards`

1. `t = 0.0001`
   这是 subsampling 常用阈值。
2. `f = np.array(list(self.word_frequency.values())) / self.token_count`
   把词频转成相对频率。
3. `self.discards = np.sqrt(t / f) + (t / f)`
   计算丢弃概率相关值。

注意：
本 lab 并没有真正用到 `self.discards` 去做高频词下采样，但这个结构提前留好了。


### `initTableNegatives`

1. `pow_frequency = np.array(list(self.word_frequency.values())) ** (3/4)`
   对词频做 `0.75` 次幂。
2. `words_pow = sum(pow_frequency)`
   求总和。
3. `ratio = pow_frequency / words_pow`
   归一化成概率。
4. `count = np.round(ratio * CorpusReader.NEGATIVE_TABLE_SIZE)`
   决定每个词在大数组中应复制多少次。
5. `for wid, c in enumerate(count):`
   逐个词填表。
6. `self.negatives += [wid] * int(c)`
   把词编号重复追加很多次。
7. `self.negatives = np.array(self.negatives)`
   转成 NumPy 数组。
8. `np.random.shuffle(self.negatives)`
   打乱顺序。

这个方法的本质是：
用“空间换时间”，预先把抽样概率编码进一个超大的数组里。


### `getNegatives`

1. 文档字符串第一句
   说明这个函数没有检查上下文词是否误入负样本。
2. 文档字符串第二句
   提醒如果你需要更严格的负样本，可以在外部自己做额外检查。
3. `while True:`
   用循环不断尝试抽样，直到满足条件。
4. `response = self.negatives[self.negpos:self.negpos + size]`
   从大数组当前位置切一段。
5. `self.negpos = (self.negpos + size) % len(self.negatives)`
   更新游标，形成环形读取。
6. `if len(response) != size:`
   如果切到数组末尾导致长度不够。
7. `response = np.concatenate((response, self.negatives[0:self.negpos]))`
   从头部补上，保证长度够。
8. `if target in response:`
   如果负样本里误包含中心词自身。
9. `continue`
   丢掉，重新取。
10. `return response`
    返回负样本。


## 3. 其它两个文件为什么重要

### `lunyu_20chapters.txt`

- 这是训练语料。
- 因为 `lang="zh"`，所以每个汉字都被当成一个 token。
- 这意味着你训练出来的是“字向量”，不是“词向量”。


### `03-wv.pdf`

- 这份讲义提供了本 lab 的理论背景。
- 重点包括：
  - one-hot 的问题
  - 什么是词向量
  - skip-gram 和 CBOW
  - softmax 为什么慢
  - negative sampling 为什么更高效
  - analogy 与可视化怎么解释词向量


## 4. 读完这份讲解后你应该掌握什么

如果你已经能跟着上面的逐行解释理解 notebook，那么你应该已经掌握了这节 lab 的四个核心点：

1. 如何把一句话转成 `(center, outside, negatives)` 训练样本。
2. 为什么 negative sampling 可以替代 full softmax。
3. SkipGram 的 forward 里到底在算什么张量和什么公式。
4. 训练好的词向量为什么可以做 analogy 和降维可视化。

如果你还想继续深入，下一步最值得做的是：

- 把 `CorpusReader.discards` 真正用于 subsampling。
- 试试更大的语料。
- 试试英文语料和中文语料的区别。
- 比较自己训练的 embedding 和预训练 GloVe 的差别。


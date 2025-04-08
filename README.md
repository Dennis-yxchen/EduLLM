# ReadMe

## Progress
0403-0406: implement new RAG Method 

0406: finished algorithm
## algorithm design (0409)
构建数据库：
- 将所有dataset的question $Q = {q_1, q_2, ......, q_n}$传入数据库 $MEM$
- 对于每个question $q_i$, 都会提取出对应的知识点 $K_i=\{\[k_i^1,k_i^2,......,k_i^p\]\}$ , 其中 $p$ 为预先定义的超参数
- 对每个知识点 $k \in K_i$ , 都将通过文本嵌入函数 $Embedder$ 转化为向量 $v \in V_i$ , 也就是说: $V_i = Embedder(K_i)$
  - 如果 $|V_i| \ne p$, 则使用 $\vec{0}$ 填充, 直到 $|V_i| = p$
- 随后我们可以对每个 question $q_i$ 得到 $memory_q\\{q_i, V_i, K_i\\}$, 将其存入 $MEM$
- 最后我们能得到: $MEM=\\{memory_1, ....., memory_n\\}$

检索数据库:
- 对每个传入的问题 $q_{example}$ 提取出对应的知识点 ${K_{example}}$
- 通过文本嵌入函数 $Embedder$ 转化为向量 ${V_{example}}$, 如果 $|V_{example}| \ne p$, 则使用 $\vec{0}$ 填充, 直到 $|V_{example}| = p$
- 对每个向量 $v_{example}^i \in V_{example}$, 对所有 $MEM$ 中的向量 $[V_1, V_2, ......, V_n]$ 计算点乘相似度
- 最后对每个 $MEM$中的所有相似度进行求和，得到每个 question $q_i$的相似度向量(pairwise similarity): $sim = pairwise({V_i, V_{example}})$，取平均: $sum(sim) / p^2$
- 返回 $topK$个相似问题以及对应的知识点

生成问题：
- 构建 $system\\_prompt$
- 将 $topK$个相似问题与知识点加入 $system\\_prompt$
- 只加入 $q_{example}$ 的知识点 ${K_{example}}$ 到 $system\\_prompt$, 得到 $final\\_prompt$
- 生成 $q_{output} = LLM($final\\_prompt$)$

from collections import abc
from collections.abc import Callable, Iterable, Sequence
import datetime
import json
import threading

import numpy as np
import pandas as pd
from importance_model import ConstantImportanceModel

_NUM_TO_RETRIEVE_TO_CONTEXTUALIZE_IMPORTANCE = 25
class NaiveAssociativeMemory:
    """实现关联记忆的类"""

    def __init__(
        self,
        sentence_embedder: Callable[[str], np.ndarray],
        importance: Callable[[str, Sequence[tuple[str, float]]], float] | None = None,
        seed: int | None = None,
    ):
        """构造函数

        参数:
          sentence_embedder: 文本嵌入模型
          importance: 将句子映射到 [0, 1] 范围内的重要性模型，如果为None，则使用一个将所有记忆设置为1.0重要性的恒定重要性模型
          seed: 随机数生成器使用的可选种子。如果为None，则使用默认rng。
        """
        self._memory_bank_lock = threading.Lock()
        self._seed = seed
        self._embedder = sentence_embedder
        self._num_to_retrieve_to_contextualize_importance = (
            _NUM_TO_RETRIEVE_TO_CONTEXTUALIZE_IMPORTANCE)
        self._importance = (
            importance or ConstantImportanceModel().importance)

        self._memory_bank = pd.DataFrame(
            columns=['text', 'tags', 'embedding', 'importance']
        )
        self._stored_hashes = set()

    def get_state(self) -> dict:
        """将 NaiveAssociativeMemory 转换为字典"""
        with self._memory_bank_lock:
            output = {
                'seed': self._seed,
                'stored_hashes': list(self._stored_hashes),
                'memory_bank': self._memory_bank.to_json(),
            }
        return output
    
    def set_state(self, state: dict) -> None:
        """从字典设置 NaiveAssociativeMemory"""
        with self._memory_bank_lock:
            self._seed = state['seed']
            self._stored_hashes = set(state['stored_hashes'])
            self._memory_bank = pd.read_json(state['memory_bank'])
    
    def add(
        self,
        text: str,
        *,
        tags: Iterable[str] = (),
        importance: float | None = None,
    ) -> None:
        """添加非重复条目（文本、标签、重要性）到记忆中

        参数:
          text: 添加到记忆中的内容
          tags: 可选标签
          importance: 可选地设置记忆的重要性。
        """
        if importance is None:
            with self._memory_bank_lock:
                memory_size = len(self._memory_bank)
            num_to_retrieve = self._num_to_retrieve_to_contextualize_importance
            if memory_size < num_to_retrieve:
                num_to_retrieve = memory_size
            context = self.retrieve_random_with_importance(k=num_to_retrieve)
            importance = self._importance(text, context)

        # 移除记忆中的所有换行符。
        text = text.replace('\n', ' ')

        contents = {
            'text': text,
            'tags': tuple(tags),
            'importance': importance,
        }
        hashed_contents = hash(tuple(contents.values()))
        derived = {'embedding': self._embedder(text)}
        new_df = pd.Series(contents | derived).to_frame().T.infer_objects()

        with self._memory_bank_lock:
            if hashed_contents in self._stored_hashes:
                return
            self._memory_bank = pd.concat(
                [self._memory_bank, new_df], ignore_index=True
            )
            self._stored_hashes.add(hashed_contents)

    def extend(
        self,
        texts: Iterable[str],
        **kwargs,
    ) -> None:
        """添加文本到记忆中

        参数:
          texts: 要添加到记忆中的字符串列表
          **kwargs: 传递给 .add 的参数
        """
        for text in texts:
            self.add(text, **kwargs)

    def get_data_frame(self) -> pd.DataFrame:
        with self._memory_bank_lock:
            return self._memory_bank.copy()

    def _get_top_k_cosine(self, x: np.ndarray, k: int):
        """返回与输入向量x最相似的前k行

        参数:
          x: 输入向量。
          k: 返回的行数。

        返回:
          按余弦相似度降序排序的行。
        """
        with self._memory_bank_lock:
            cosine_similarities = self._memory_bank['embedding'].apply(
                lambda y: np.dot(x, y)
            )

            # 按余弦相似度降序排序。
            cosine_similarities.sort_values(ascending=False, inplace=True)

            # 返回前k行。
            return self._memory_bank.iloc[cosine_similarities.head(k).index]

    def _get_top_k_similar_rows(
        self, x, k: int, use_importance: bool = True
    ):
        """返回与输入向量x最相似的前k行

        参数:
          x: 输入向量。
          k: 返回的行数。
          use_importance: 如果为true则按重要性加权相似度

        返回:
          按相似度降序排序的行。
        """
        with self._memory_bank_lock:
            cosine_similarities = self._memory_bank['embedding'].apply(
                lambda y: np.dot(x, y)
            )

            similarity_score = cosine_similarities

            if use_importance:
                importance = self._memory_bank['importance']
                similarity_score += importance

            # 按相似度降序排序。
            similarity_score.sort_values(ascending=False, inplace=True)

            # 返回前k行。
            return self._memory_bank.iloc[similarity_score.head(k).index]

    def _pd_to_text(
        self,
        data: pd.DataFrame,
    ) -> Sequence[str]:
        """将数据框格式化为字符串列表

        参数:
          data: 要处理的数据框

        返回:
          字符串列表，每个记忆对应一个字符串
        """
        output = data['text']
        return output.tolist()

    def retrieve_associative(
        self,
        query: str,
        k: int = 1,
        use_importance: bool = True,
    ) -> Sequence[str]:
        """关联检索记忆

        参数:
          query: 用于检索的字符串
          k: 要检索的记忆数量
          use_importance: 是否使用重要性进行检索

        返回:
          对应记忆的字符串列表
        """
        query_embedding = self._embedder(query)

        data = self._get_top_k_similar_rows(
            query_embedding,
            k,
            use_importance=use_importance,
        )

        return self._pd_to_text(data)

    def retrieve_by_regex(
        self,
        regex: str,
    ) -> Sequence[str]:
        """通过正则表达式检索记忆

        参数:
          regex: 匹配的正则表达式

        返回:
          对应记忆的字符串列表
        """
        with self._memory_bank_lock:
            data = self._memory_bank[self._memory_bank['text'].str.contains(regex)]

        return self._pd_to_text(data)

    def retrieve_random(
        self,
        k: int = 1,
    ) -> Sequence[str]:
        """检索随机的记忆

        参数:
          k: 要检索的记忆数量

        返回:
          对应记忆的字符串列表
        """
        with self._memory_bank_lock:
            data = self._memory_bank.sample(k, random_state=self._seed)
        return self._pd_to_text(data)

    def retrieve_random_with_importance(
        self,
        k: int = 1,
    ) -> Sequence[tuple[str, float]]:
        """检索随机的记忆并返回重要性

        参数:
          k: 要检索的记忆数量

        返回:
          对应记忆的字符串列表及其重要性值
        """
        with self._memory_bank_lock:
            data = self._memory_bank.sample(k, random_state=self._seed)
        return tuple(zip(list(data['text']), list(data['importance'])))

    def __len__(self):
        """返回记忆库中的条目数量

        由于记忆不能被删除，长度不会减少，可用于检查记忆库的内容是否发生变化。
        """
        with self._memory_bank_lock:
            return len(self._memory_bank)

    def get_mean_importance(self) -> float:
        """返回记忆库中记忆的重要性的平均值"""
        with self._memory_bank_lock:
            return self._memory_bank['importance'].mean()

    def get_max_importance(self) -> float:
        """返回记忆库中记忆的最大重要性"""
        with self._memory_bank_lock:
            return self._memory_bank['importance'].max()

    def get_min_importance(self) -> float:
        """返回记忆库中记忆的重要性的最小重要性"""
        with self._memory_bank_lock:
            return self._memory_bank['importance'].min()

    def set_num_to_retrieve_to_contextualize_importance(
        self, num_to_retrieve: int) -> None:
        """设置用于上下文化重要性的要检索的记忆数量

        设置为0以禁用重要性的上下文化

        参数:
          num_to_retrieve: 用于上下文化重要性的要检索的记忆数量
        """
        self._num_to_retrieve_to_contextualize_importance = num_to_retrieve

    def get_all_memories_as_text(self) -> Sequence[str]:
        """返回记忆库中的所有记忆作为字符串序列"""
        memories_data_frame = self.get_data_frame()
        texts = self._pd_to_text(memories_data_frame)
        return texts
    
    def save(self, file_path: str) -> None:
        """将记忆库保存到 JSON 文件"""
        state = self.get_state()
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=4)

    def load(self, file_path: str) -> None:
        """从 JSON 文件加载记忆库"""
        with open(file_path, 'r') as f:
            state = json.load(f)
        # 将 embedding 从列表转换回 np.ndarray
        for item in state['memory_bank']:
            item['embedding'] = np.array(item['embedding'])
        self.set_state(state)
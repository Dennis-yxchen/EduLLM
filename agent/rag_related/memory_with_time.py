from collections import abc
from collections.abc import Callable, Iterable, Sequence
import datetime
import threading

# from concordia.associative_memory import importance_function
# from concordia.typing import entity_component
import numpy as np
import pandas as pd
from importance_model import ConstantImportanceModel

_NUM_TO_RETRIEVE_TO_CONTEXTUALIZE_IMPORTANCE = 25



def _check_date_in_range(timestamp: datetime.datetime) -> None:
  if timestamp < pd.Timestamp.min:
    min_date = pd.Timestamp.min
    raise ValueError(f'timestamp {timestamp} < pd.Timestamp.min {min_date}')
  if timestamp > pd.Timestamp.max:
    max_date = pd.Timestamp.max
    raise ValueError(f'timestamp {timestamp} > pd.Timestamp.max {max_date}')


class AssociativeMemory:
    """实现关联记忆的类"""

    def __init__(
        self,
        sentence_embedder: Callable[[str], np.ndarray],
        importance: Callable[[str, Sequence[tuple[str, float]]], float] | None = None,
        clock: Callable[[], datetime.datetime] = datetime.datetime.now,
        clock_step_size: datetime.timedelta | None = None,
        seed: int | None = None,
    ):
        """构造函数

        参数:
          sentence_embedder: 文本嵌入模型
          importance: 将句子映射到 [0, 1] 范围内的重要性模型，如果为None，则使用一个将所有记忆设置为1.0重要性的恒定重要性模型
          clock: 获取添加记忆时的时间的可调用对象
          clock_step_size: 设置时钟的时间步长。如果为None，则假定时钟精度很高
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
            columns=['text', 'time', 'tags', 'embedding', 'importance']
        )
        self._clock_now = clock
        self._interval = clock_step_size
        self._stored_hashes = set()

    # def get_state(self) -> entity_component.ComponentState:
    #     """将 AssociativeMemory 转换为字典"""
    #     with self._memory_bank_lock:
    #         serialized_times = self._memory_bank['time'].apply(
    #             lambda x: x.strftime('[%d-%b-%Y-%H:%M:%S]')
    #         ).tolist()
    #         output = {
    #             'seed': self._seed,
    #             'stored_hashes': list(self._stored_hashes),
    #             'memory_bank': self._memory_bank.to_json(),
    #             'time': serialized_times,
    #         }
    #         if self._interval:
    #             output['interval'] = self._interval.total_seconds()
    #     return output

    # def set_state(self, state: entity_component.ComponentState) -> None:
    #     """从字典设置 AssociativeMemory"""
    #     with self._memory_bank_lock:
    #         self._seed = state['seed']
    #         self._stored_hashes = set(state['stored_hashes'])
    #         self._memory_bank = pd.read_json(state['memory_bank'])
    #         self._memory_bank['time'] = [
    #             datetime.datetime.strptime(t, '[%d-%b-%Y-%H:%M:%S]')
    #             for t in state['time']
    #         ]
    #         if 'interval' in state:
    #             self._interval = datetime.timedelta(seconds=state['interval'])

    def add(
        self,
        text: str,
        *,
        timestamp: datetime.datetime | None = None,
        tags: Iterable[str] = (),
        importance: float | None = None,
    ) -> None:
        """添加非重复条目（时间、文本、标签、重要性）到记忆中

        参数:
          text: 添加到记忆中的内容
          timestamp: 记忆的时间戳
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

        if timestamp is None:
            timestamp = self._clock_now()

        _check_date_in_range(timestamp)

        # 移除记忆中的所有换行符。
        text = text.replace('\n', ' ')

        contents = {
            'text': text,
            'time': timestamp,
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
        self, x, k: int, use_recency: bool = True, use_importance: bool = True
    ):
        """返回与输入向量x最相似的前k行

        参数:
          x: 输入向量。
          k: 返回的行数。
          use_recency: 如果为true则按最近程度加权相似度
          use_importance: 如果为true则按重要性加权相似度

        返回:
          按相似度降序排序的行。
        """
        with self._memory_bank_lock:
            cosine_similarities = self._memory_bank['embedding'].apply(
                lambda y: np.dot(x, y)
            )

            similarity_score = cosine_similarities

            if use_recency:
                max_time = self._memory_bank['time'].max()
                discounted_time = self._memory_bank['time'].apply(
                    lambda y: 0.99 ** ((max_time - y) / datetime.timedelta(minutes=1))
                )
                similarity_score += discounted_time

            if use_importance:
                importance = self._memory_bank['importance']
                similarity_score += importance

            # 按相似度降序排序。
            similarity_score.sort_values(ascending=False, inplace=True)

            # 返回前k行。
            return self._memory_bank.iloc[similarity_score.head(k).index]

    def _get_k_recent(self, k: int):
        with self._memory_bank_lock:
            recency = self._memory_bank['time'].sort_values(ascending=False)
            return self._memory_bank.iloc[recency.head(k).index]

    def _pd_to_text(
        self,
        data: pd.DataFrame,
        add_time: bool = False,
        sort_by_time: bool = True,
    ) -> Sequence[str]:
        """将数据框格式化为字符串列表

        参数:
          data: 要处理的数据框
          add_time: 是否添加时间
          sort_by_time: 是否按时间排序

        返回:
          字符串列表，每个记忆对应一个字符串
        """
        if sort_by_time:
            data = data.sort_values('time', ascending=True)

        if add_time and not data.empty:
            if self._interval:
                this_time = data['time']
                next_time = data['time'] + self._interval

                interval = this_time.dt.strftime(
                    '%d %b %Y [%H:%M:%S  '
                ) + next_time.dt.strftime('- %H:%M:%S]: ')
                output = interval + data['text']
            else:
                output = data['time'].dt.strftime('[%d %b %Y %H:%M:%S] ') + data['text']
        else:
            output = data['text']

        return output.tolist()

    def retrieve_associative(
        self,
        query: str,
        k: int = 1,
        use_recency: bool = True,
        use_importance: bool = True,
        add_time: bool = True,
        sort_by_time: bool = True,
    ) -> Sequence[str]:
        """关联检索记忆

        参数:
          query: 用于检索的字符串
          k: 要检索的记忆数量
          use_recency: 是否使用时间戳按最近程度加权
          use_importance: 是否使用重要性进行检索
          add_time: 是否向输出添加时间戳
          sort_by_time: 是否按时间对结果排序

        返回:
          对应记忆的字符串列表
        """
        query_embedding = self._embedder(query)

        data = self._get_top_k_similar_rows(
            query_embedding,
            k,
            use_recency=use_recency,
            use_importance=use_importance,
        )

        return self._pd_to_text(data, add_time=add_time, sort_by_time=sort_by_time)

    def retrieve_by_regex(
        self,
        regex: str,
        add_time: bool = True,
        sort_by_time: bool = True,
    ) -> Sequence[str]:
        """通过正则表达式检索记忆

        参数:
          regex: 匹配的正则表达式
          add_time: 是否向输出添加时间戳
          sort_by_time: 是否按时间对结果排序

        返回:
          对应记忆的字符串列表
        """
        with self._memory_bank_lock:
            data = self._memory_bank[self._memory_bank['text'].str.contains(regex)]

        return self._pd_to_text(data, add_time=add_time, sort_by_time=sort_by_time)

    def retrieve_time_interval(
        self,
        time_from: datetime.datetime,
        time_until: datetime.datetime,
        add_time: bool = False,
    ) -> Sequence[str]:
        """检索时间间隔内的记忆

        参数:
          time_from: 时间间隔的开始时间
          time_until: 时间间隔的结束时间
          add_time: 是否向输出添加时间戳

        返回:
          对应记忆的字符串列表
        """
        with self._memory_bank_lock:
            data = self._memory_bank[
                (self._memory_bank['time'] >= time_from)
                & (self._memory_bank['time'] <= time_until)
            ]

        return self._pd_to_text(data, add_time=add_time, sort_by_time=True)

    def retrieve_recent(
        self,
        k: int = 1,
        add_time: bool = False,
    ) -> Sequence[str]:
        """检索最近的记忆

        参数:
          k: 要检索的记忆数量
          add_time: 是否向输出添加时间戳

        返回:
          对应记忆的字符串列表
        """
        with self._memory_bank_lock:
            recent_memories = self._get_k_recent(k)
        return self._pd_to_text(recent_memories, add_time=add_time, sort_by_time=True)


    def retrieve_recent_with_importance(
        self,
        k: int = 1,
        add_time: bool = False,
    ) -> tuple[Sequence[str], Sequence[float]]:
        """检索最近的记忆并返回重要性

        参数:
          k: 要检索的记忆数量
          add_time: 是否向输出添加时间戳

        返回:
          对应记忆的字符串列表和它们的重要性值
        """
        data = self._get_k_recent(k)

        return (
            self._pd_to_text(data, add_time=add_time, sort_by_time=True),
            list(data['importance']),
        )

    def retrieve_random(
        self,
        k: int = 1,
        add_time: bool = False,
    ) -> Sequence[str]:
        """检索随机的记忆

        参数:
          k: 要检索的记忆数量
          add_time: 是否向输出添加时间戳

        返回:
          对应记忆的字符串列表
        """
        with self._memory_bank_lock:
            data = self._memory_bank.sample(k, random_state=self._seed)
        return self._pd_to_text(data, add_time=add_time, sort_by_time=True)

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
        """返回记忆库中记忆的最小重要性"""
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

    def get_all_memories_as_text(
        self,
        add_time: bool = True,
        sort_by_time: bool = True,
    ) -> Sequence[str]:
        """返回记忆库中的所有记忆作为字符串序列"""
        memories_data_frame = self.get_data_frame()
        texts = self._pd_to_text(memories_data_frame,
                                 add_time=add_time,
                                 sort_by_time=sort_by_time)
        return texts
import os
import json

class DatasetReader:
    def __init__(self, data_path, file_name='comp2501.json', preprocess_func=None):
        # 获取当前脚本的目录
        current_dir = os.path.dirname(__file__)
        # 构建数据文件的路径
        self.data_path = os.path.join(data_path, file_name)
        # 预处理函数
        self.preprocess_func = preprocess_func
    
    def read_data(self):
        # 读取JSON文件
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def preprocess(self, data):
        # 使用提供的预处理函数或默认处理逻辑
        # print(self.preprocess_func)
        if self.preprocess_func:
            return self.preprocess_func(data)
        else:
            # 默认的处理逻辑，返回原始数据
            return data
    
    def get_data(self):
        # 读取并预处理数据
        data = self.read_data()
        processed_data = self.preprocess(data)
        # print(f"processed_data: {processed_data}")
        return processed_data


if __name__ == '__main__':
    # 自定义预处理函数示例
    def uppercase_values(data):
        result = []
        for i in data:
            tmp = str(i).upper()
            result.append(tmp)
        return result
    # 测试案例：使用自定义预处理函数
    reader = DatasetReader(preprocess_func=uppercase_values)
    data = reader.get_data()
    print(data)
    print(f"end of uppercase_values test")
    # 测试案例：不使用自定义预处理函数
    reader = DatasetReader()
    data = reader.get_data()
    print(data)
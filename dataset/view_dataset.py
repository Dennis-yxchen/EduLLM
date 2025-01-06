import json

# 定义读取 JSON 文件的函数
def read_and_print_json(file_path):
    try:
        # 打开 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            # 加载 JSON 数据
            data = json.load(file)
            # 打印 JSON 数据
            for i in data:
                print(i)
                print('-------------------')
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except json.JSONDecodeError:
        print(f"文件 {file_path} 不是有效的 JSON 格式。")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

# 调用函数，传入 JSON 文件路径
json_file_path = 'comp2501.json'  # 替换为你的 JSON 文件路径
read_and_print_json(json_file_path)
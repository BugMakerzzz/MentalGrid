import yaml

# 1. 准备数据：使用列表和字典的组合
# train_data = [f'ProcTHOR-Train-{i}' for i in range(10000)]
train_data = [f'FloorPlan_Train{i+1}_{j+1}' for i in range(12) for j in range(5)]
test_data = [f'FloorPlan_Val{i+1}_{j+1}' for i in range(3) for j in range(5)]
data = {
    'test': test_data,
    'train': train_data
}
# 2. 写入 YAML 文件
with open('/mnt/userdata/ljc/data/ai2thor-hab/ai2thor-hab/configs/scenes/RoboTHOR/scene_splits.yml', 'w', encoding='utf-8') as f:
    # sort_keys=False 可以保持字典原有的顺序
    # default_flow_style=False 确保生成的是“块状结构”而不是大括号格式
    yaml.dump(data, f, sort_keys=False, default_flow_style=False)

print("YAML 文件已成功生成！")
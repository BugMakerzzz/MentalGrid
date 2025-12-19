import matplotlib.pyplot as plt
import numpy as np

S = 0.5

EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
    (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
    (0, 4), (1, 5), (2, 6), (3, 7)   # 侧边
]

def get_cube_vertices(center_point, s=S):
    """根据中心点计算立方体的八个顶点坐标"""
    x_c, y_c, z_c = center_point
    
    # 计算八个顶点 (Vertices)
    v = np.array([
        [x_c - s, y_c - s, z_c - s],  # 0
        [x_c + s, y_c - s, z_c - s],  # 1
        [x_c + s, y_c + s, z_c - s],  # 2
        [x_c - s, y_c + s, z_c - s],  # 3
        [x_c - s, y_c - s, z_c + s],  # 4
        [x_c + s, y_c - s, z_c + s],  # 5
        [x_c + s, y_c + s, z_c + s],  # 6
        [x_c - s, y_c + s, z_c + s]   # 7
    ])
    return v

def plot_multiple_cubes(center_points_list):
    
    # 1. 创建 3D 图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 用于计算总体的轴范围
    all_x, all_y, all_z = [], [], []

    # 2. 循环绘制每个立方体
    for i, center_point in enumerate(center_points_list):
        # 计算当前立方体的顶点
        vertices = get_cube_vertices(center_point)
        x_c, y_c, z_c = center_point
        
        # 收集所有顶点的坐标，以便稍后设置合适的视图范围
        all_x.extend(vertices[:, 0])
        all_y.extend(vertices[:, 1])
        all_z.extend(vertices[:, 2])

        # 确定颜色（可以使用索引 i 来变化颜色）
        color = plt.cm.viridis(i / len(center_points_list)) # 使用颜色映射生成不同颜色
        
        # 绘制12条边 (线框图)
        for i, j in EDGES:
            x = [vertices[i, 0], vertices[j, 0]]
            y = [vertices[i, 1], vertices[j, 1]]
            z = [vertices[i, 2], vertices[j, 2]]
            ax.plot(x, y, z, color=color, alpha=0.8, linewidth=1.5)

        # 绘制中心点
        ax.scatter([x_c], [y_c], [z_c], color='black', marker='o', s=50)

    # 3. 设置图的属性，确保所有立方体都在视野内
    if all_x:
        min_x, max_x = min(all_x) - S, max(all_x) + S
        min_y, max_y = min(all_y) - S, max(all_y) + S
        min_z, max_z = min(all_z) - S, max(all_z) + S
        
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])
        ax.set_zlim([min_z, max_z])

    ax.set_title(f'Plotting {len(center_points_list)} Cubes')
    plt.savefig('test.png')

# --- 示例运行 ---

# 定义多个中心点列表
centers = [
    [0, 0, 0],       # 原点
    [3, 0, 0],       # 沿着X轴
    [0, 4, 1],       # 另一个位置
    [-2, -3, 5],     # 第四个位置
]

# 调用函数生成并展示图像
plot_multiple_cubes(centers)
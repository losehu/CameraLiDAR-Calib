import numpy as np

# 原始外参矩阵
extrinsic = np.array([
    [0.705727, -0.707989, -0.0264904, 50.5759],
    [0.708389, 0.705752, 0.00997664, 69.2468],
    [0.0116323, -0.0258063, 0.999599, -285.353],
    [0, 0, 0, 1]
])

# 计算旋转角度：1970/5188*360度
angle_deg = (1970 / 5188) * 360

# 定义绕三个轴旋转的旋转矩阵
def rotation_x(angle_deg):
    """绕X轴旋转"""
    angle = np.radians(angle_deg)
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle), -np.sin(angle), 0],
        [0, np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1]
    ])

def rotation_y(angle_deg):
    """绕Y轴旋转"""
    angle = np.radians(angle_deg)
    return np.array([
        [np.cos(angle), 0, np.sin(angle), 0],
        [0, 1, 0, 0],
        [-np.sin(angle), 0, np.cos(angle), 0],
        [0, 0, 0, 1]
    ])

def rotation_z(angle_deg):
    """绕Z轴旋转"""
    angle = np.radians(angle_deg)
    return np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def save_matrix(matrix, filename):
    """保存矩阵到文件，格式与原始文件一致"""
    with open(filename, 'w') as f:
        f.write('extrinsic\n')
        for i in range(4):
            row_str = '  '.join([f'{matrix[i, j]:.6g}' if j < 3 else f'{matrix[i, j]:.6g}' 
                                for j in range(4)])
            f.write(row_str + '\n')

# 生成6个旋转后的外参矩阵
rotations = [
    ('x_pos', rotation_x(angle_deg)),
    ('x_neg', rotation_x(-angle_deg)),
    ('y_pos', rotation_y(angle_deg)),
    ('y_neg', rotation_y(-angle_deg)),
    ('z_pos', rotation_z(angle_deg)),
    ('z_neg', rotation_z(-angle_deg))
]

print("原始外参矩阵:")
print(extrinsic)
print(f"\n旋转角度: {angle_deg:.6f}度 (1970/5188*360)")
print("="*60 + "\n")

for name, rot_matrix in rotations:
    # 应用旋转：新矩阵 = 旋转矩阵 × 原矩阵
    rotated = rot_matrix @ extrinsic
    
    filename = f'extrinsic_911_{name}.txt'
    save_matrix(rotated, filename)
    
    axis = name.split('_')[0].upper()
    direction = '+' if 'pos' in name else '-'
    print(f"{name} (绕{axis}轴旋转{direction}{angle_deg:.6f}度):")
    print(rotated)
    print(f"已保存到: {filename}\n")

print("所有旋转外参矩阵已生成完成！")

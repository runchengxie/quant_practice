import numpy as np

# 已知数据
mu = np.array([0.02, 0.07, 0.15, 0.20])   # 期望收益率向量
sigma = np.array([0.05, 0.12, 0.17, 0.25]) # 标准差向量
R = np.array([
    [1.0, 0.3, 0.3, 0.3],
    [0.3, 1.0, 0.6, 0.6],
    [0.3, 0.6, 1.0, 0.6],
    [0.3, 0.6, 0.6, 1.0]
])

m = 0.045  # 目标收益率4.5%

# 构建协方差矩阵Σ = D R D
D = np.diag(sigma)         # 对角矩阵D
Sigma = D @ R @ D          # 矩阵乘法构建Σ

# 求Σ⁻¹
Sigma_inv = np.linalg.inv(Sigma)

# 定义向量1
ones = np.ones(len(mu))

# 计算A、B、C
A = ones @ Sigma_inv @ ones
B = ones @ Sigma_inv @ mu
C = mu @ Sigma_inv @ mu

# 根据公式计算w*
w_star = ((C - B*m)/(A*C - B**2)) * (Sigma_inv @ ones) + ((A*m - B)/(A*C - B**2)) * (Sigma_inv @ mu)

# 输出结果
print("Optimal weights w* for m=4.5%:")
for i, w in enumerate(w_star, start=1):
    print(f"Asset {i}: {w:.6f}")

# 计算组合风险:
portfolio_variance = w_star @ Sigma @ w_star
portfolio_risk = np.sqrt(portfolio_variance)
print(f"Portfolio risk σ_Π: {portfolio_risk:.6f}")

# 检查答案

# 使用优化后的权重进行验证

# 计算组合的预期收益率
portfolio_return = mu @ w_star
print(f"组合的预期收益率: {portfolio_return:.6f}")

# 检查是否接近期望的目标收益率
if np.isclose(portfolio_return, m):
    print("验证通过：组合的预期收益率接近目标收益率。")
else:
    print("验证失败：组合的预期收益率与目标收益率不符。")
    
# 检查权重之和是否为1
total_weight = np.sum(w_star)
if np.isclose(total_weight, 1.0):
    print("验证通过：组合的权重之和为1。")
else:
    print("验证失败：组合的权重之和不为1。")
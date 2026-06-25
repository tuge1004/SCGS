import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import os

# ==================== 配置 ====================
INPUT_FILE = r"C:\Users\hyy46\PycharmProjects\数据挖掘\AI就业_全量数据_含圈层标签_筛选后.csv"
OUTPUT_DIR = r"C:\Users\hyy46\PycharmProjects\数据挖掘"

TOP_K = 5                     # 每个圈层选取的领袖数量
CHUNK_SIZE = 50000            # 分块读取大小
K_SAMPLE = 1000               # 介数中心性采样节点数

WEIGHT_DEGREE = 0.4
WEIGHT_BETWEENNESS = 0.3
WEIGHT_FANS = 0.3

print("=" * 60)
print("基于筛选后数据重建转发网络并识别意见领袖")

# ==================== 第一步：分块聚合转发边及粉丝数 ====================
print("第一步：分块聚合转发边及粉丝数...")
edges_dict = defaultdict(int)
user_fans = defaultdict(int)

# 先获取文件总行数（用于百分比进度条）
# 方法：遍历一次只读一列并计数，速度快
print("正在统计文件总行数...")
total_rows = 0
for chunk in pd.read_csv(INPUT_FILE, usecols=['作者ID'], chunksize=CHUNK_SIZE, encoding='utf-8-sig'):
    total_rows += len(chunk)
print(f"文件总行数: {total_rows}")

# 重新读取并处理，带进度条
reader = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False, encoding='utf-8-sig')
with tqdm(total=total_rows, desc="聚合数据", unit="行") as pbar:
    for chunk in reader:
        chunk['作者ID'] = chunk['作者ID'].astype(str).str.strip()
        if '根微博作者' in chunk.columns:
            chunk['根微博作者'] = chunk['根微博作者'].astype(str).str.strip()

        # 转发边
        if '原创/转发' in chunk.columns and '根微博作者' in chunk.columns:
            retweet = chunk[(chunk['原创/转发'] == '转发') & (chunk['根微博作者'] != '')]
            for _, row in retweet.iterrows():
                edges_dict[(row['作者ID'], row['根微博作者'])] += 1

        # 粉丝数
        if '粉丝数' in chunk.columns:
            chunk['粉丝数'] = pd.to_numeric(chunk['粉丝数'], errors='coerce').fillna(0)
            fans = chunk.groupby('作者ID')['粉丝数'].max()
            for uid, f in fans.items():
                if f > user_fans.get(uid, 0):
                    user_fans[uid] = f
        pbar.update(len(chunk))

print(f"聚合后转发边数量: {len(edges_dict)}")

# ==================== 第二步：构建用户转发有向图 ====================
print("第二步：构建有向图...")
G = nx.DiGraph()
for (src, dst), w in tqdm(edges_dict.items(), desc="添加边到网络", unit="条边"):
    G.add_edge(src, dst, weight=w)
print(f"网络节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

del edges_dict  # 释放内存

# ==================== 第三步：计算中心性 ====================
print("第三步：计算中心性指标...")

# 度中心性（很快，无进度条）
print("计算度中心性...")
deg_cent = nx.degree_centrality(G)

# 介数中心性（采样）
print(f"计算近似介数中心性（采样 {K_SAMPLE} 节点）...")
bet_cent = nx.betweenness_centrality(G, k=K_SAMPLE, weight='weight')

# 特征向量中心性跳过（图不连通）
print("跳过特征向量中心性（图不连通）")
eig_cent = {node: 0.0 for node in G.nodes()}

# ==================== 第四步：计算综合影响力 ====================
print("第四步：计算综合影响力...")

def normalize(d):
    arr = np.array(list(d.values()))
    if arr.std() == 0:
        return {k: 0.5 for k in d}
    minv, maxv = arr.min(), arr.max()
    return {k: (v - minv) / (maxv - minv) for k, v in d.items()}

n_deg = normalize(deg_cent)
n_bet = normalize(bet_cent)
n_fans = normalize(user_fans)

influence = {}
for node in tqdm(G.nodes(), desc="计算影响力", unit="用户"):
    influence[node] = (WEIGHT_DEGREE * n_deg.get(node, 0) +
                       WEIGHT_BETWEENNESS * n_bet.get(node, 0) +
                       WEIGHT_FANS * n_fans.get(node, 0))

# ==================== 第五步：获取圈层-用户映射 ====================
print("第五步：获取圈层-用户映射...")
comm_users = defaultdict(set)
reader = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False, encoding='utf-8-sig')
for chunk in tqdm(reader, desc="读取圈层信息"):
    chunk['作者ID'] = chunk['作者ID'].astype(str).str.strip()
    if '圈层ID' not in chunk.columns:
        print("错误：全量数据中缺少 '圈层ID' 列！")
        exit(1)
    for _, row in chunk[['作者ID', '圈层ID']].drop_duplicates().iterrows():
        comm_users[row['圈层ID']].add(row['作者ID'])
print(f"圈层总数: {len(comm_users)}")

# ==================== 第六步：为每个圈层选择意见领袖 ====================
print("第六步：选择意见领袖...")
comm_leaders = {}
comm_leader_set = defaultdict(set)

for cid, users in tqdm(comm_users.items(), desc="圈层领袖选择"):
    valid = {u for u in users if u in influence}
    if not valid:
        comm_leaders[cid] = []
        continue
    sorted_users = sorted(valid, key=lambda u: influence[u], reverse=True)
    top_users = sorted_users[:TOP_K]
    comm_leaders[cid] = top_users
    comm_leader_set[cid] = set(top_users)

leader_count = sum(1 for v in comm_leaders.values() if v)
print(f"成功选出领袖的圈层数: {leader_count}/{len(comm_users)}")

# ==================== 第七步：将意见领袖标记添加到全量数据 ====================
print("第七步：标记全量数据...")
output_file = os.path.join(OUTPUT_DIR, "AI就业_全量数据_含意见领袖标记.csv")
reader = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False, encoding='utf-8-sig')
first_chunk = True
with tqdm(total=total_rows, desc="写入标记文件", unit="行") as pbar:
    for chunk in reader:
        chunk['作者ID'] = chunk['作者ID'].astype(str).str.strip()
        chunk['是否意见领袖'] = chunk.apply(
            lambda row: 1 if row['作者ID'] in comm_leader_set.get(row['圈层ID'], set()) else 0,
            axis=1
        )
        if first_chunk:
            chunk.to_csv(output_file, index=False, encoding='utf-8-sig', mode='w')
            first_chunk = False
        else:
            chunk.to_csv(output_file, index=False, encoding='utf-8-sig', mode='a', header=False)
        pbar.update(len(chunk))
print(f"标记后文件已保存至: {output_file}")

# ==================== 第八步：输出圈层意见领袖列表 ====================
print("第八步：保存圈层意见领袖列表...")
leader_records = []
for cid, leaders in comm_leaders.items():
    for rank, user in enumerate(leaders, 1):
        leader_records.append({
            '圈层ID': cid,
            '意见领袖ID': user,
            '排名': rank,
            '综合影响力': influence.get(user, 0),
            '度中心性': deg_cent.get(user, 0),
            '介数中心性': bet_cent.get(user, 0),
            '粉丝数': user_fans.get(user, 0)
        })
leader_df = pd.DataFrame(leader_records)
leader_output = os.path.join(OUTPUT_DIR, "圈层意见领袖列表.csv")
leader_df.to_csv(leader_output, index=False, encoding='utf-8-sig')
print(f"意见领袖列表已保存至: {leader_output}")
print("全部完成！")
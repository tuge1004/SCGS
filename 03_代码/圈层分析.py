import os
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
from scipy.stats import beta as beta_dist
import jieba
import emoji
from snownlp import SnowNLP

# ==================== 配置 ====================
INPUT_DIR = r"C:\Users\hyy46\PycharmProjects\数据挖掘"
OUTPUT_DIR = r"C:\Users\hyy46\PycharmProjects\数据挖掘"
FILTERED_DATA = os.path.join(INPUT_DIR, "AI就业_全量数据_含圈层标签_筛选后.csv")

CHUNK_SIZE = 50000
LANG_SAMPLE_SIZE = 500          # 每圈层语言分析最大文本数
CENTRALITY_SAMPLE = 500         # 介数中心性采样节点数
MAX_COMMS_OVERLAP = 500         # 计算重合率矩阵的最大圈层数（按用户数排序）
MIN_NODES_IN_COMM = 10          # 用于指标计算的圈层最小用户数（筛选后文件已保证帖子数≥10，这里二次过滤）

# ==================== 1. 分块聚合数据 ====================
print("分块读取筛选后数据并聚合...")
comm_users = defaultdict(set)           # 圈层 -> 用户集合
user_first_time = {}                    # (用户ID, 圈层ID) -> 最早日期
total_interactions = defaultdict(int)   # 圈层 -> 总互动量
user_count = defaultdict(set)           # 圈层 -> 用户数（用于活跃强度）
min_date, max_date = None, None
text_samples = defaultdict(list)        # 圈层 -> 文本列表（用于语言特征）
edges_list = []                         # 转发边（作者ID, 根微博作者）

reader = pd.read_csv(FILTERED_DATA, chunksize=CHUNK_SIZE, low_memory=False, encoding='utf-8-sig')
with tqdm(desc="分块处理", unit="块") as pbar:
    for chunk in reader:
        # 基本清洗
        chunk['作者ID'] = chunk['作者ID'].astype(str).str.strip()
        if '根微博作者' in chunk.columns:
            chunk['根微博作者'] = chunk['根微博作者'].astype(str).str.strip()

        # ---- 用户-圈层映射 ----
        for _, row in chunk[['作者ID', '圈层ID']].drop_duplicates().iterrows():
            comm_users[row['圈层ID']].add(row['作者ID'])

        # ---- 首次参与时间 ----
        if '日期' in chunk.columns:
            chunk['日期'] = pd.to_datetime(chunk['日期'], errors='coerce')
            date_chunk = chunk.dropna(subset=['日期'])
            for (uid, cid), grp in date_chunk.groupby(['作者ID', '圈层ID']):
                t = grp['日期'].min()
                key = (uid, cid)
                if key not in user_first_time or t < user_first_time[key]:
                    user_first_time[key] = t
            if not date_chunk.empty:
                chunk_min = date_chunk['日期'].min()
                chunk_max = date_chunk['日期'].max()
                if min_date is None or chunk_min < min_date:
                    min_date = chunk_min
                if max_date is None or chunk_max > max_date:
                    max_date = chunk_max

        # ---- 互动量 ----
        for col in ['转发数', '评论数', '点赞数']:
            if col not in chunk.columns:
                chunk[col] = 0
        chunk['互动量'] = chunk[['转发数', '评论数', '点赞数']].fillna(0).sum(axis=1)
        inter_agg = chunk.groupby('圈层ID')['互动量'].sum()
        for cid, val in inter_agg.items():
            total_interactions[cid] += val
        for cid, users in chunk.groupby('圈层ID')['作者ID'].apply(set).items():
            user_count[cid].update(users)

        # ---- 文本采样 ----
        if '全文内容' in chunk.columns:
            text_chunk = chunk[chunk['全文内容'].notna()][['圈层ID', '全文内容']]
            for cid, grp in text_chunk.groupby('圈层ID'):
                texts = grp['全文内容'].tolist()
                if len(text_samples[cid]) < LANG_SAMPLE_SIZE * 2:  # 多收一些
                    text_samples[cid].extend(texts)

        # ---- 转发边 ----
        if '原创/转发' in chunk.columns and '根微博作者' in chunk.columns:
            retweet = chunk[(chunk['原创/转发'] == '转发') & (chunk['根微博作者'] != '')]
            edges_list.append(retweet[['作者ID', '根微博作者']])

        pbar.update(1)

# 时间跨度
time_span = 1
if min_date is not None and max_date is not None:
    time_span = (max_date - min_date).days
    if time_span == 0:
        time_span = 1
print(f"数据时间跨度: {time_span} 天")

# ==================== 2. 圈层重合率矩阵（仅计算前 MAX_COMMS_OVERLAP 大圈层） ====================
print("计算圈层重合率矩阵（仅取前{}大圈层）...".format(MAX_COMMS_OVERLAP))
# 按用户数降序排列圈层
sorted_comms = sorted(comm_users.items(), key=lambda x: len(x[1]), reverse=True)
top_comm_ids = [cid for cid, _ in sorted_comms[:MAX_COMMS_OVERLAP] if len(comm_users[cid]) >= MIN_NODES_IN_COMM]
n = len(top_comm_ids)
overlap_mat = np.zeros((n, n))
for i, cid_a in enumerate(tqdm(top_comm_ids, desc="重合率计算")):
    set_a = comm_users[cid_a]
    for j, cid_b in enumerate(top_comm_ids):
        if i <= j:
            set_b = comm_users[cid_b]
            union = len(set_a | set_b)
            inter = len(set_a & set_b)
            val = inter / union if union > 0 else 0.0
            overlap_mat[i, j] = val
            overlap_mat[j, i] = val

overlap_df = pd.DataFrame(overlap_mat, index=top_comm_ids, columns=top_comm_ids)
overlap_df.to_csv(os.path.join(OUTPUT_DIR, "圈层重合率矩阵_前{}大.csv".format(n)), encoding='utf-8-sig')
print("重合率矩阵已保存。")
upper_tri = overlap_mat[np.triu_indices(n, k=1)]
if len(upper_tri) > 0:
    print(f"平均重合率: {np.mean(upper_tri):.4f}")
    print(f"中位重合率: {np.median(upper_tri):.4f}")
    print(f"最大重合率: {np.max(upper_tri):.4f}")

# ==================== 3. 圈层感染阈值 ====================
print("估计圈层感染阈值...")
comm_threshold = {}
for cid, users_in_comm in tqdm(sorted_comms, desc="拟合阈值分布"):
    if len(users_in_comm) < MIN_NODES_IN_COMM:
        continue
    # 收集该圈层内所有用户的首次参与时间
    user_times = [t for (uid, c), t in user_first_time.items() if c == cid]
    if not user_times:
        comm_threshold[cid] = 0.3
        continue
    user_times.sort()
    n_users = len(user_times)
    thresholds = [i / n_users for i in range(n_users)]  # 比例序列
    vals = np.array(thresholds)
    vals = vals[(vals > 1e-6) & (vals < 1 - 1e-6)]
    if len(vals) < 5:
        comm_threshold[cid] = np.mean(thresholds)
    else:
        try:
            a, b, loc, scale = beta_dist.fit(vals, floc=0, fscale=1)
            comm_threshold[cid] = beta_dist.mean(a, b, loc, scale)
        except:
            comm_threshold[cid] = np.mean(vals)
print("感染阈值估计完成。")

# ==================== 4. 圈层活跃强度 ====================
print("计算圈层活跃强度...")
activity = {}
for cid, users_in_comm in tqdm(sorted_comms, desc="活跃强度"):
    if len(users_in_comm) < MIN_NODES_IN_COMM:
        continue
    total_inter = total_interactions.get(cid, 0)
    u_cnt = len(users_in_comm)  # 用集合大小
    if u_cnt > 0 and time_span > 0:
        activity[cid] = total_inter / (u_cnt * time_span)
    else:
        activity[cid] = 0.0

# ==================== 5. 语言风格特征 ====================
print("提取语言风格特征...")
# 随机采样每个圈层的文本
sampled_texts = []
np.random.seed(42)
for cid, users_in_comm in sorted_comms:
    texts = text_samples.get(cid, [])
    if len(texts) > LANG_SAMPLE_SIZE:
        chosen = np.random.choice(texts, LANG_SAMPLE_SIZE, replace=False)
    else:
        chosen = texts
    for t in chosen:
        sampled_texts.append({'圈层ID': cid, '全文内容': t})
sampled = pd.DataFrame(sampled_texts)
print(f"采样文本数量: {len(sampled)}")

# ----- 情感分析（SnowNLP） -----
def get_sentiment(text):
    try:
        s = SnowNLP(str(text))
        return s.sentiments  # 0~1，越大越正面
    except:
        return 0.5

sent_scores = []
for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc="情感分析"):
    sent_scores.append({'圈层ID': row['圈层ID'], '情感得分': get_sentiment(row['全文内容'])})
sent_df = pd.DataFrame(sent_scores)
comm_sent = sent_df.groupby('圈层ID').agg(
    正面比例=('情感得分', lambda x: (x > 0.6).mean()),   # 粗略正面
    负面比例=('情感得分', lambda x: (x < 0.4).mean()),
    情感平均得分=('情感得分', 'mean')
).reset_index()

# ----- 认知复杂度 -----
def calc_complexity(texts):
    all_words = []
    char_lens = []
    for t in texts:
        words = list(jieba.cut(str(t)))
        all_words.append(words)
        char_lens.append(len(str(t)))
    total_tokens = sum(len(w) for w in all_words)
    total_types = len(set(word for words in all_words for word in words))
    ttr = total_types / total_tokens if total_tokens > 0 else 0
    avg_sent_len = np.mean(char_lens) if char_lens else 0
    func_set = set('的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这'.split())
    func_cnt = 0
    total_words = 0
    for words in all_words:
        for w in words:
            if w in func_set:
                func_cnt += 1
            total_words += 1
    func_ratio = func_cnt / total_words if total_words > 0 else 0
    return ttr, avg_sent_len, func_ratio

comp = sampled.groupby('圈层ID')['全文内容'].apply(lambda x: calc_complexity(list(x))).reset_index()
comp[['型例比', '平均句长', '功能词比例']] = pd.DataFrame(comp['全文内容'].tolist(), index=comp.index)
comp.drop('全文内容', axis=1, inplace=True)

# ----- 语域特征 -----
net_slang_set = set(['yyds','绝绝子','破防','躺平','内卷','emo','xswl','u1s1','awsl','凡尔赛','集美','芭比Q','社死','摆烂'])
def calc_register(texts):
    total_chars = 0
    emo_cnt = 0
    slang_cnt = 0
    for t in texts:
        s = str(t)
        total_chars += len(s)
        emo_cnt += len(emoji.emoji_list(s))
        for word in net_slang_set:
            slang_cnt += s.lower().count(word)
    emo_density = emo_cnt / total_chars if total_chars > 0 else 0
    slang_density = slang_cnt / total_chars if total_chars > 0 else 0
    return emo_density, slang_density

reg = sampled.groupby('圈层ID')['全文内容'].apply(lambda x: calc_register(list(x))).reset_index()
reg[['emoji密度', '网络用语密度']] = pd.DataFrame(reg['全文内容'].tolist(), index=reg.index)
reg.drop('全文内容', axis=1, inplace=True)

# 合并语言特征
lang_feat = comm_sent.merge(comp, on='圈层ID', how='outer').merge(reg, on='圈层ID', how='outer')
print("语言特征提取完成。")

# ==================== 6. 意见领袖识别 ====================
print("识别意见领袖...")
all_edges = pd.concat(edges_list, ignore_index=True)
edges_weighted = all_edges.groupby(['作者ID', '根微博作者']).size().reset_index(name='weight')
G_net = nx.DiGraph()
for _, row in tqdm(edges_weighted.iterrows(), total=len(edges_weighted), desc="构建转发网络"):
    G_net.add_edge(row['作者ID'], row['根微博作者'], weight=row['weight'])

# 中心性计算
deg_cent = nx.degree_centrality(G_net)
if len(G_net) > CENTRALITY_SAMPLE:
    bet_cent = nx.betweenness_centrality(G_net, k=CENTRALITY_SAMPLE, weight='weight')
else:
    bet_cent = nx.betweenness_centrality(G_net, weight='weight')
try:
    eig_cent = nx.eigenvector_centrality_numpy(G_net, weight='weight')
except:
    eig_cent = {node: 0.0 for node in G_net.nodes()}

# 粉丝数（分块再次读取）
user_fans = defaultdict(int)
reader2 = pd.read_csv(FILTERED_DATA, chunksize=CHUNK_SIZE, low_memory=False, encoding='utf-8-sig')
for chunk in reader2:
    if '粉丝数' in chunk.columns:
        chunk['作者ID'] = chunk['作者ID'].astype(str).str.strip()
        fans = chunk.groupby('作者ID')['粉丝数'].max()
        for uid, f in fans.items():
            if f > user_fans.get(uid, 0):
                user_fans[uid] = f

# 归一化并计算综合影响力
def normalize(d):
    arr = np.array(list(d.values()))
    if arr.std() == 0:
        return {k: 0.5 for k in d}
    return {k: (v - arr.min()) / (arr.max() - arr.min()) for k, v in d.items()}

n_deg = normalize(deg_cent)
n_bet = normalize(bet_cent)
n_eig = normalize(eig_cent)
n_fans = normalize(user_fans)

influence = {}
for node in G_net.nodes():
    influence[node] = 0.25*n_deg.get(node,0) + 0.25*n_bet.get(node,0) + 0.25*n_eig.get(node,0) + 0.25*n_fans.get(node,0)

# 每个圈层前5名意见领袖
comm_leaders = {}
for cid in sorted_comms:
    cid = cid[0] if isinstance(cid, tuple) else cid
    users = [u for u in comm_users[cid] if u in influence]
    top = sorted(users, key=lambda u: influence[u], reverse=True)[:5] if users else []
    comm_leaders[cid] = top

# ==================== 7. 输出综合指标 ====================
print("生成综合指标表...")
records = []
for cid, users_in_comm in sorted_comms:
    if len(users_in_comm) < MIN_NODES_IN_COMM:
        continue
    rec = {'圈层ID': cid}
    rec['用户数'] = len(users_in_comm)
    rec['感染阈值'] = comm_threshold.get(cid, np.nan)
    rec['活跃强度'] = activity.get(cid, np.nan)
    lf = lang_feat[lang_feat['圈层ID'] == cid]
    if not lf.empty:
        lf = lf.iloc[0]
        rec['正面比例'] = lf['正面比例']
        rec['负面比例'] = lf['负面比例']
        rec['情感平均得分'] = lf['情感平均得分']
        rec['型例比'] = lf['型例比']
        rec['平均句长'] = lf['平均句长']
        rec['功能词比例'] = lf['功能词比例']
        rec['emoji密度'] = lf['emoji密度']
        rec['网络用语密度'] = lf['网络用语密度']
    else:
        for k in ['正面比例','负面比例','情感平均得分','型例比','平均句长','功能词比例','emoji密度','网络用语密度']:
            rec[k] = np.nan
    rec['意见领袖'] = ','.join(comm_leaders.get(cid, []))
    records.append(rec)

final_df = pd.DataFrame(records)
final_output_path = os.path.join(OUTPUT_DIR, "圈层综合指标.csv")
final_df.to_csv(final_output_path, index=False, encoding='utf-8-sig')
print(f"综合指标已保存至: {final_output_path}")
print("全部完成！")
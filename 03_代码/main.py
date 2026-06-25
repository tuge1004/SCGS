import os
import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
from collections import defaultdict, Counter
from tqdm import tqdm
from scipy.stats import beta as beta_dist
import jieba
import emoji
from transformers import pipeline

# ==================== 配置 ====================
INPUT_DIR = r"C:\Users\hyy46\Downloads\AI就业（数智员工）\w9yqtsjdboss122251219114222133"
OUTPUT_DIR = r"C:\Users\hyy46\PycharmProjects\数据挖掘"
os.makedirs(OUTPUT_DIR, exist_ok=True)

file_paths = [
    os.path.join(INPUT_DIR, "108111011.csv"),
    os.path.join(INPUT_DIR, "108110997.csv"),
    os.path.join(INPUT_DIR, "108110857.csv"),
    os.path.join(INPUT_DIR, "108111039.csv"),
    os.path.join(INPUT_DIR, "108111074.csv"),
    os.path.join(INPUT_DIR, "108110948.csv"),
    os.path.join(INPUT_DIR, "108110962.csv"),
    os.path.join(INPUT_DIR, "108110885.csv"),
    os.path.join(INPUT_DIR, "108110955.csv"),
    os.path.join(INPUT_DIR, "108111025.csv"),
    os.path.join(INPUT_DIR, "108110920.csv"),
    os.path.join(INPUT_DIR, "108110878.csv"),
    os.path.join(INPUT_DIR, "108110969.csv"),
    os.path.join(INPUT_DIR, "108111053.csv"),
    os.path.join(INPUT_DIR, "108110892.csv"),
    os.path.join(INPUT_DIR, "108110906.csv"),
    os.path.join(INPUT_DIR, "108110927.csv"),
    os.path.join(INPUT_DIR, "108111067.csv"),
    os.path.join(INPUT_DIR, "108110941.csv"),
    os.path.join(INPUT_DIR, "108110871.csv"),
    os.path.join(INPUT_DIR, "108110864.csv"),
    os.path.join(INPUT_DIR, "108110990.csv"),
    os.path.join(INPUT_DIR, "108111018.csv"),
    os.path.join(INPUT_DIR, "108110850.csv"),
    os.path.join(INPUT_DIR, "108110934.csv"),
    os.path.join(INPUT_DIR, "108110843.csv"),
    os.path.join(INPUT_DIR, "108111004.csv"),
    os.path.join(INPUT_DIR, "108111060.csv"),
    os.path.join(INPUT_DIR, "108110976.csv"),
    os.path.join(INPUT_DIR, "108110983.csv"),
    os.path.join(INPUT_DIR, "108111032.csv"),
    os.path.join(INPUT_DIR, "108110913.csv"),
    os.path.join(INPUT_DIR, "108111046.csv"),
    os.path.join(INPUT_DIR, "108110899.csv"),
]

MIN_NODES = 10           # 帖子数阈值
LANG_SAMPLE_SIZE = 500
CENTRALITY_SAMPLE = 500
CHUNK_SIZE = 50000

# ==================== 第一步：读取原始数据 ====================
print("=" * 60)
print("第一步：读取原始数据")
df_list = []
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'utf-16']
for file in tqdm(file_paths, desc="读取文件"):
    if not os.path.exists(file):
        print(f"⚠️ 文件不存在: {file}")
        continue
    success = False
    for enc in encodings:
        try:
            df = pd.read_csv(file, encoding=enc, low_memory=False)
            df_list.append(df)
            success = True
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            pass
    if not success:
        print(f"✗ 无法读取: {file}")

if not df_list:
    raise ValueError("没有成功读取任何文件。")

full_df = pd.concat(df_list, ignore_index=True)
print(f"合并后总记录数: {len(full_df)}")

# ==================== 第二步：构建帖子转发网络 ====================
print("=" * 60)
print("第二步：构建帖子-帖子转发网络")

# 确保关键字段存在
assert 'id' in full_df.columns, "数据缺少 'id' 字段"
# 使用 MD5-根微博mid 作为目标节点（被转发的帖子哈希）
target_col = 'MD5-根微博mid'
if target_col not in full_df.columns:
    # 尝试其他可能的列名
    if '根微博mid' in full_df.columns:
        target_col = '根微博mid'
    elif '根微博链接' in full_df.columns:
        target_col = '根微博链接'
    else:
        raise KeyError("找不到根微博mid字段")

print(f"使用帖子ID (id) 作为源节点，{target_col} 作为目标节点")

# 提取转发边
edge_df = full_df[full_df['原创/转发'] == '转发'].copy()
edge_df['source'] = edge_df['id'].astype(str).str.strip()
edge_df['target'] = edge_df[target_col].astype(str).str.strip()
# 丢弃目标为空的边
edge_df = edge_df[edge_df['target'] != '']
edge_df = edge_df[edge_df['target'] != 'nan']
print(f"有效转发边数: {len(edge_df)}")

# 统计边权重（相同 source-target 对聚合）
edge_weights = edge_df.groupby(['source', 'target']).size().reset_index(name='weight')
print(f"去重后边数: {len(edge_weights)}")

# 构建有向图（使用 NetworkX，注意内存）
G = nx.DiGraph()
for _, row in tqdm(edge_weights.iterrows(), total=len(edge_weights), desc="添加边到有向图"):
    G.add_edge(row['source'], row['target'], weight=row['weight'])
print(f"有向图节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

# 转为无向加权图（聚合无向边）
print("转为无向加权图...")
edge_weights_undir = defaultdict(float)
for u, v, data in tqdm(G.edges(data=True), total=G.number_of_edges(), desc="聚合无向边"):
    key = tuple(sorted((u, v)))
    edge_weights_undir[key] += data['weight']

G_undir = nx.Graph()
G_undir.add_nodes_from(G.nodes())
for (u, v), w in tqdm(edge_weights_undir.items(), desc="构建无向图"):
    G_undir.add_edge(u, v, weight=w)
print(f"无向图节点数: {G_undir.number_of_nodes()}, 边数: {G_undir.number_of_edges()}")

# ==================== 第三步：Louvain 社区发现 ====================
print("=" * 60)
print("第三步：Louvain 社区发现（帖子网络）")
# 大规模网络社区发现，可能较慢，但可接受
partition = community_louvain.best_partition(G_undir, weight='weight')

communities = defaultdict(list)
for node, comm_id in partition.items():
    communities[comm_id].append(node)

print(f"识别出圈层数量: {len(communities)}")
for cid in sorted(communities, key=lambda x: len(communities[x]), reverse=True)[:10]:
    print(f"圈层 {cid}: {len(communities[cid])} 个帖子")

# ==================== 第四步：将帖子圈层标签合并到全量数据 ====================
print("=" * 60)
print("第四步：添加圈层标签到每一条帖子")
# 创建帖子ID -> 圈层ID的映射
post_comm_df = pd.DataFrame({
    'id': list(partition.keys()),
    '圈层ID': list(partition.values())
})
# 确保 full_df 中的 id 也是字符串
full_df['id'] = full_df['id'].astype(str).str.strip()

# 左连接
full_df = full_df.merge(post_comm_df, on='id', how='left')
full_df['圈层ID'] = full_df['圈层ID'].fillna(-1).astype(int)

# 输出全量数据（含圈层标签）
full_output_path = os.path.join(OUTPUT_DIR, "AI就业_全量数据_含圈层标签.csv")
full_df.to_csv(full_output_path, index=False, encoding='utf-8-sig')
print(f"全量数据已保存: {full_output_path}")

# ==================== 第五步：导出 Gephi 文件（可选，基于帖子网络） ====================
print("导出 Gephi 网络文件（帖子网络）...")
# 为 Gephi 添加节点属性：社区和度中心性
degree_cent = nx.degree_centrality(G)
for node, comm_id in partition.items():
    G.nodes[node]['community'] = comm_id
    G.nodes[node]['degree_cent'] = degree_cent.get(node, 0.0)
gexf_path = os.path.join(OUTPUT_DIR, "AI就业_帖子转发网络.gexf")
nx.write_gexf(G, gexf_path)
print(f"Gephi 文件已保存: {gexf_path}")

# ==================== 第六步：计算所有圈层的统计指标（帖子数、边数等） ====================
print("=" * 60)
print("第六步：计算圈层统计指标（帖子层级）")
comm_stats = []
for cid, nodes in tqdm(sorted(communities.items(), key=lambda x: len(x[1])), desc="统计圈层"):
    subgraph = G_undir.subgraph(nodes)
    n_posts = subgraph.number_of_nodes()
    n_edges = subgraph.number_of_edges()
    density = nx.density(subgraph) if n_posts > 1 else 0.0

    if n_posts > 1:
        degrees = dict(subgraph.degree())
        avg_deg = np.mean(list(degrees.values()))
        deg_cent_list = [d / (n_posts - 1) for d in degrees.values()]
        avg_deg_cent = np.mean(deg_cent_list)
    else:
        avg_deg = 0.0
        avg_deg_cent = 0.0

    try:
        avg_clust = nx.average_clustering(subgraph)
    except:
        avg_clust = 0.0

    comm_stats.append({
        '圈层ID': cid,
        '帖子数': n_posts,
        '边数': n_edges,
        '密度': density,
        '平均度': avg_deg,
        '平均度中心性': avg_deg_cent,
        '平均聚类系数': avg_clust,
        '平均介数中心性': 0.0
    })

stats_df = pd.DataFrame(comm_stats).sort_values('帖子数', ascending=False)
stats_output_path = os.path.join(OUTPUT_DIR, "AI就业_圈层统计指标.csv")
stats_df.to_csv(stats_output_path, index=False, encoding='utf-8-sig')
print(f"圈层统计指标已保存: {stats_output_path}")

# ==================== 第七步：筛选帖子数 ≥ MIN_NODES 的圈层 ====================
print("=" * 60)
print("第七步：筛选圈层并输出筛选后数据")
valid_cids = stats_df[stats_df['帖子数'] >= MIN_NODES]['圈层ID'].tolist()
valid_cids_set = set(valid_cids)
print(f"筛选出 {len(valid_cids)} 个圈层（帖子数≥{MIN_NODES}）")

# 保存筛选后的统计指标
stats_filtered = stats_df[stats_df['圈层ID'].isin(valid_cids)].copy()
stats_filtered.to_csv(os.path.join(OUTPUT_DIR, "AI就业_圈层统计指标_筛选后.csv"), index=False, encoding='utf-8-sig')

# 分块读取全量数据，筛选有效圈层
print("分块筛选全量数据...")
reader = pd.read_csv(full_output_path, chunksize=CHUNK_SIZE, low_memory=False, encoding='utf-8-sig')
filtered_chunks = []
with tqdm(desc="筛选数据行") as pbar:
    for chunk in reader:
        chunk_filtered = chunk[chunk['圈层ID'].isin(valid_cids_set)].copy()
        if not chunk_filtered.empty:
            filtered_chunks.append(chunk_filtered)
        pbar.update(len(chunk))

full_filtered = pd.concat(filtered_chunks, ignore_index=True)
filtered_output_path = os.path.join(OUTPUT_DIR, "AI就业_全量数据_含圈层标签_筛选后.csv")
full_filtered.to_csv(filtered_output_path, index=False, encoding='utf-8-sig')
print(f"筛选后全量数据已保存: {filtered_output_path}")

# ==================== 第八步：基于筛选后数据计算高级指标 ====================
print("=" * 60)
print("第八步：计算重合率、感染阈值、活跃强度、语言特征、意见领袖")

data_source = filtered_output_path

# 聚合容器
comm_users = defaultdict(set)        # 圈层 -> 用户集合
user_first_time = {}
total_interactions = defaultdict(int)
user_count = defaultdict(set)
min_date, max_date = None, None
text_samples = defaultdict(list)
edges_list = []

reader = pd.read_csv(data_source, chunksize=CHUNK_SIZE, low_memory=False, encoding='utf-8-sig')
with tqdm(desc="分块聚合数据", unit="块") as pbar:
    for chunk in reader:
        chunk['作者ID'] = chunk['作者ID'].astype(str).str.strip()
        if '根微博作者' in chunk.columns:
            chunk['根微博作者'] = chunk['根微博作者'].astype(str).str.strip()

        # 用户-圈层映射（现在一个用户可属于多个圈层）
        for _, row in chunk[['作者ID', '圈层ID']].drop_duplicates().iterrows():
            comm_users[row['圈层ID']].add(row['作者ID'])

        # 首次参与时间
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
                min_date = chunk_min if min_date is None else min(min_date, chunk_min)
                max_date = chunk_max if max_date is None else max(max_date, chunk_max)

        # 互动量
        for col in ['转发数', '评论数', '点赞数']:
            if col not in chunk.columns:
                chunk[col] = 0
        chunk['互动量'] = chunk[['转发数', '评论数', '点赞数']].fillna(0).sum(axis=1)
        inter_agg = chunk.groupby('圈层ID')['互动量'].sum()
        for cid, val in inter_agg.items():
            total_interactions[cid] += val
        for cid, users in chunk.groupby('圈层ID')['作者ID'].apply(set).items():
            user_count[cid].update(users)

        # 文本采样
        if '全文内容' in chunk.columns:
            text_chunk = chunk[chunk['全文内容'].notna()][['圈层ID', '全文内容']]
            for cid, grp in text_chunk.groupby('圈层ID'):
                texts = grp['全文内容'].tolist()
                if len(text_samples[cid]) < LANG_SAMPLE_SIZE * 2:
                    text_samples[cid].extend(texts)

        # 转发边（用于意见领袖识别）
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

# 8.1 圈层重合率矩阵（基于用户集合）
print("计算圈层重合率矩阵...")
comm_ids = sorted(comm_users.keys())
n = len(comm_ids)
overlap_mat = np.zeros((n, n))
for i, cid_a in enumerate(tqdm(comm_ids, desc="重合率计算")):
    set_a = comm_users[cid_a]
    for j, cid_b in enumerate(comm_ids):
        if i <= j:
            set_b = comm_users[cid_b]
            union = len(set_a | set_b)
            inter = len(set_a & set_b)
            val = inter / union if union > 0 else 0.0
            overlap_mat[i, j] = val
            overlap_mat[j, i] = val

overlap_df = pd.DataFrame(overlap_mat, index=comm_ids, columns=comm_ids)
overlap_df.to_csv(os.path.join(OUTPUT_DIR, "圈层重合率矩阵.csv"), encoding='utf-8-sig')
print("重合率矩阵已保存。")
upper_tri = overlap_mat[np.triu_indices(n, k=1)]
if len(upper_tri) > 0:
    print(f"平均重合率: {np.mean(upper_tri):.4f}")
    print(f"中位重合率: {np.median(upper_tri):.4f}")
    print(f"最大重合率: {np.max(upper_tri):.4f}")

# 8.2 圈层感染阈值
print("估计圈层感染阈值...")
comm_threshold = {}
for cid in tqdm(comm_ids, desc="拟合阈值分布"):
    user_times = [t for (uid, c), t in user_first_time.items() if c == cid]
    if not user_times:
        comm_threshold[cid] = 0.3
        continue
    user_times.sort()
    n_users = len(user_times)
    thresholds = [i / n_users for i in range(n_users)]
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

# 8.3 活跃强度（这里基于互动量/用户数/天数）
activity = {}
for cid in comm_ids:
    total_inter = total_interactions.get(cid, 0)
    u_cnt = len(user_count.get(cid, set()))
    if u_cnt > 0 and time_span > 0:
        activity[cid] = total_inter / (u_cnt * time_span)
    else:
        activity[cid] = 0.0

# 8.4 语言风格特征（与之前相同）
sampled_texts = []
np.random.seed(42)
for cid in comm_ids:
    texts = text_samples.get(cid, [])
    if len(texts) > LANG_SAMPLE_SIZE:
        chosen = np.random.choice(texts, LANG_SAMPLE_SIZE, replace=False)
    else:
        chosen = texts
    for t in chosen:
        sampled_texts.append({'圈层ID': cid, '全文内容': t})
sampled = pd.DataFrame(sampled_texts)
print(f"采样文本数量: {len(sampled)}")

sentiment_pipeline = pipeline("sentiment-analysis",
    model="uer/roberta-base-finetuned-jd-binary-chinese",
    tokenizer="uer/roberta-base-finetuned-jd-binary-chinese",
    device=-1, truncation=True, max_length=512)

def get_sentiment(text):
    try:
        res = sentiment_pipeline(str(text)[:512])[0]
        return res['label'], res['score']
    except:
        return 'neutral', 0.5

sent_records = []
for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc="情感分析"):
    label, score = get_sentiment(row['全文内容'])
    sent_records.append({'圈层ID': row['圈层ID'], 'label': label, 'score': score})
sent_df = pd.DataFrame(sent_records)
comm_sent = sent_df.groupby('圈层ID').agg(
    正面比例=('label', lambda x: (x == 'positive').mean()),
    负面比例=('label', lambda x: (x == 'negative').mean()),
    情感平均得分=('score', 'mean')
).reset_index()

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

lang_feat = comm_sent.merge(comp, on='圈层ID', how='outer').merge(reg, on='圈层ID', how='outer')
print("语言特征提取完成。")

# 8.5 意见领袖识别（基于用户转发网络）
all_edges = pd.concat(edges_list, ignore_index=True)
edges_weighted = all_edges.groupby(['作者ID', '根微博作者']).size().reset_index(name='weight')
G_net = nx.DiGraph()
for _, row in tqdm(edges_weighted.iterrows(), total=len(edges_weighted), desc="构建转发网络"):
    G_net.add_edge(row['作者ID'], row['根微博作者'], weight=row['weight'])

deg_cent = nx.degree_centrality(G_net)
if len(G_net) > CENTRALITY_SAMPLE:
    bet_cent = nx.betweenness_centrality(G_net, k=CENTRALITY_SAMPLE, weight='weight')
else:
    bet_cent = nx.betweenness_centrality(G_net, weight='weight')
try:
    eig_cent = nx.eigenvector_centrality_numpy(G_net, weight='weight')
except:
    eig_cent = {node: 0.0 for node in G_net.nodes()}

user_fans = defaultdict(int)
reader2 = pd.read_csv(data_source, chunksize=CHUNK_SIZE, low_memory=False, encoding='utf-8-sig')
for chunk in reader2:
    if '粉丝数' in chunk.columns:
        chunk['作者ID'] = chunk['作者ID'].astype(str).str.strip()
        fans = chunk.groupby('作者ID')['粉丝数'].max()
        for uid, f in fans.items():
            user_fans[uid] = max(user_fans.get(uid, 0), f)

def normalize(d):
    arr = np.array(list(d.values()))
    if arr.std() == 0:
        return {k: 0.5 for k in d}
    return {k: (v - arr.min()) / (arr.max() - arr.min()) for k, v in d.items()}

n_deg = normalize(deg_cent)
n_bet = normalize(bet_cent)
n_eig = normalize(eig_cent)
n_fans = normalize(user_fans)
influence = {n: 0.25*n_deg.get(n,0) + 0.25*n_bet.get(n,0) + 0.25*n_eig.get(n,0) + 0.25*n_fans.get(n,0) for n in G_net.nodes()}

comm_leaders = {}
for cid in comm_ids:
    users = [u for u in comm_users[cid] if u in influence]
    top = sorted(users, key=lambda u: influence[u], reverse=True)[:5]
    comm_leaders[cid] = top

# ==================== 第九步：综合指标输出 ====================
print("生成综合指标表...")
records = []
for cid in comm_ids:
    rec = {'圈层ID': cid}
    rec['用户数'] = len(comm_users[cid])
    rec['感染阈值'] = comm_threshold.get(cid, np.nan)
    rec['活跃强度'] = activity.get(cid, np.nan)
    lf = lang_feat[lang_feat['圈层ID']==cid]
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
print("全部处理完成！")zz
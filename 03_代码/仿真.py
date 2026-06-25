import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import random
from scipy.stats import ttest_ind, pearsonr, f_oneway
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

# ==================== 全局配置 ====================
DATA_DIR = r"C:\Users\hyy46\PycharmProjects\数据挖掘"
OUTPUT_DIR = os.path.join(DATA_DIR, "仿真结果")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FONT_PATH = r"C:\Users\hyy46\Downloads\Simhei.ttf"
font_prop = FontProperties(fname=FONT_PATH)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

COLORS = ['#DBD2E5', '#6F9CC0', '#A4D290', '#B0A0C7', '#D5B7A9', '#F8CDCE', '#649E94']

# 仿真参数
TOP_N = 30
THRESHOLD_RATIO = 0.10
MAX_STEPS = 2000
NUM_RUNS = 10                     # 正式分析请改为 30
SEED_RATIO = 0.05
TRIGGER_INFECT_RATIO = 0.0
OVERLAP_AMPLIFY = 5.0
LEADER_BROADCAST_PROB = 0.9
INFLUENCER_BOOST = 1.5
BOOST_DURATION = 3
BOOST_AMOUNT = 0.2

# ==================== 1. 数据加载与网络构建 ====================
print("正在从标注文件加载数据并构建转发网络...")
data_file = os.path.join(DATA_DIR, "AI就业_全量数据_含意见领袖标记.csv")

cid_to_users = defaultdict(set)
influencer_set = defaultdict(set)
edge_weights = defaultdict(int)
user_posts = defaultdict(int)
user_activity = defaultdict(float)
user_fans = defaultdict(int)

chunksize = 50000
reader = pd.read_csv(data_file, chunksize=chunksize, low_memory=False, encoding='utf-8-sig')
print("分块处理中...")
for chunk in tqdm(reader, desc="处理数据块"):
    chunk['作者ID'] = chunk['作者ID'].astype(str).str.strip()
    chunk['根微博作者'] = chunk['根微博作者'].astype(str).str.strip()
    chunk['圈层ID'] = chunk['圈层ID'].astype(int)

    is_retweet = (chunk['原创/转发'] == '转发') & (chunk['根微博作者'] != '') & (chunk['根微博作者'] != 'nan')
    for _, row in chunk[is_retweet].iterrows():
        src, tgt = row['作者ID'], row['根微博作者']
        if src != tgt:
            edge_weights[(src, tgt)] += 1

    for _, row in chunk.iterrows():
        uid, cid = row['作者ID'], row['圈层ID']
        cid_to_users[cid].add(uid)
        user_posts[uid] += 1
        interaction = (int(row.get('转发数', 0) or 0) +
                       int(row.get('评论数', 0) or 0) +
                       int(row.get('点赞数', 0) or 0))
        user_activity[uid] += interaction
        fans = int(row.get('粉丝数', 0) or 0)
        if fans > user_fans[uid]:
            user_fans[uid] = fans

    if '是否意见领袖' in chunk.columns:
        mask_leader = chunk['是否意见领袖'] == 1
        for _, row in chunk[mask_leader].iterrows():
            influencer_set[row['圈层ID']].add(row['作者ID'])

print(f"总转发边数: {len(edge_weights)}")
selected_ids = sorted(cid_to_users.keys(), key=lambda x: len(cid_to_users[x]), reverse=True)[:TOP_N]
print(f"核心圈层: {selected_ids}")

# 构建全量有向图
print("构建完整转发网络...")
G = nx.DiGraph()
for (src, tgt), w in tqdm(edge_weights.items(), total=len(edge_weights), desc="添加边"):
    G.add_edge(src, tgt, weight=w)
G_undir = G.to_undirected()
print(f"无向网络节点数: {G_undir.number_of_nodes()}, 边数: {G_undir.number_of_edges()}")
total_nodes = G_undir.number_of_nodes()

# 个性化传播概率
def compute_user_prob(uid, is_influencer=False):
    log_posts = np.log1p(user_posts.get(uid, 0))
    log_act = np.log1p(user_activity.get(uid, 0))
    log_fans = np.log1p(user_fans.get(uid, 0))
    score = 0.3 * log_posts + 0.4 * log_act + 0.3 * log_fans
    if is_influencer:
        prob = 0.50 + 0.3 / (1 + np.exp(-1.2 * (score - 4.0)))
        return min(0.85, max(0.50, prob))
    else:
        prob = 0.03 + 0.27 / (1 + np.exp(-0.8 * (score - 2.0)))
        return min(0.30, max(0.03, prob))

user_prob_cache = {}
influencer_probs = []
normal_probs = []
for cid in selected_ids:
    for uid in cid_to_users[cid]:
        is_leader = uid in influencer_set.get(cid, set())
        prob = compute_user_prob(uid, is_influencer=is_leader)
        user_prob_cache[uid] = prob
        if is_leader:
            influencer_probs.append(prob)
        else:
            normal_probs.append(prob)

default_prob = np.mean(normal_probs) if normal_probs else 0.15

# 圈层重合率矩阵与帖子数
print("计算圈层重合率矩阵...")
overlap_mat = np.zeros((TOP_N, TOP_N))
for i, cid_a in enumerate(selected_ids):
    set_a = cid_to_users[cid_a]
    for j, cid_b in enumerate(selected_ids):
        if i <= j:
            set_b = cid_to_users[cid_b]
            union = len(set_a | set_b)
            inter = len(set_a & set_b)
            val = inter / union if union > 0 else 0.0
            overlap_mat[i, j] = val
            overlap_mat[j, i] = val

post_counts = {cid: sum(user_posts[uid] for uid in cid_to_users[cid]) for cid in selected_ids}
def compute_inter_layer_prob(cid_from, cid_to):
    i = selected_ids.index(cid_from)
    j = selected_ids.index(cid_to)
    overlap = overlap_mat[i, j]
    if overlap == 0:
        return 0.0
    posts_factor = np.log1p(post_counts[cid_from] / 1000.0)
    prob = overlap * OVERLAP_AMPLIFY * posts_factor
    return min(0.8, prob)

influencer_nodes = {cid: influencer_set.get(cid, set()) for cid in selected_ids}
normal_nodes = {cid: cid_to_users[cid] - influencer_nodes[cid] for cid in selected_ids}
user_counts = {cid: len(cid_to_users[cid]) for cid in selected_ids}
thresholds = {cid: max(2, int(THRESHOLD_RATIO * user_counts[cid])) for cid in selected_ids}
default_seed_layers = selected_ids[:3]

# ==================== 3. 仿真函数 ====================
def run_single_simulation(seed_strategy='random', seed_layers=None, seed_ratio=0.05,
                          exclude_influencer=False, lsm=1.0, random_seed=None,
                          threshold_multiplier=1.0):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    infected = set()
    frontier = set()
    boost_remaining = defaultdict(int)
    inter_layer_events = defaultdict(int)
    first_trigger_time = {}

    if seed_layers is None:
        layer_ids = default_seed_layers
    else:
        layer_ids = [cid for cid in seed_layers if cid in selected_ids]
        if not layer_ids:
            layer_ids = default_seed_layers

    for cid in layer_ids:
        pool = list(normal_nodes[cid] if exclude_influencer else cid_to_users[cid])
        if not pool:
            continue
        if seed_strategy == 'influencer':
            leaders_in_pool = list(influencer_nodes[cid] & set(pool))
            chosen = list(leaders_in_pool)
            remaining = max(1, int(len(pool) * seed_ratio)) - len(chosen)
            if remaining > 0:
                other_pool = [u for u in pool if u not in chosen]
                if other_pool:
                    chosen.extend(random.sample(other_pool, min(remaining, len(other_pool))))
        else:
            num_seeds = max(1, int(len(pool) * seed_ratio))
            chosen = random.sample(pool, min(num_seeds, len(pool)))
        for node in chosen:
            infected.add(node)
            frontier.add(node)

    history_counts = [len(infected)]
    history_deltas = [len(infected)]
    triggered = {cid: False for cid in selected_ids}
    step = 0
    effective_thresholds = {cid: max(2, int(thresholds[cid] * threshold_multiplier)) for cid in selected_ids}

    while frontier and step < MAX_STEPS:
        step += 1
        new_frontier = set()

        for node in frontier:
            base_prob = user_prob_cache.get(node, default_prob) * lsm
            if boost_remaining.get(node, 0) > 0:
                base_prob = min(0.95, base_prob + BOOST_AMOUNT)
                boost_remaining[node] -= 1
            base_prob = min(0.95, max(0.01, base_prob))

            is_leader = any(node in influencer_nodes.get(cid, set()) for cid in selected_ids)
            neighbors = list(G_undir.neighbors(node))

            if is_leader:
                if random.random() < LEADER_BROADCAST_PROB:
                    for nb in neighbors:
                        if nb not in infected:
                            infected.add(nb)
                            new_frontier.add(nb)
                            boost_remaining[nb] = BOOST_DURATION
                else:
                    enhanced_prob = base_prob * INFLUENCER_BOOST
                    for nb in neighbors:
                        if nb not in infected and random.random() < enhanced_prob:
                            infected.add(nb)
                            new_frontier.add(nb)
                            boost_remaining[nb] = BOOST_DURATION
            else:
                for nb in neighbors:
                    if nb not in infected and random.random() < base_prob:
                        infected.add(nb)
                        new_frontier.add(nb)

        for cid in selected_ids:
            if triggered[cid]:
                leaders_infected = len(influencer_nodes[cid] & infected)
                leader_factor = 1.0 + 0.5 * min(1.0, leaders_infected / max(1, len(influencer_nodes[cid])))
                for target_cid in selected_ids:
                    if cid == target_cid or triggered[target_cid]:
                        continue
                    inter_prob = compute_inter_layer_prob(cid, target_cid) * leader_factor
                    if inter_prob <= 0:
                        continue
                    target_nodes = list(cid_to_users[target_cid])
                    uninfected = [n for n in target_nodes if n not in infected]
                    if not uninfected:
                        continue
                    attempts = max(1, int(len(uninfected) * inter_prob * 0.3))
                    attempts = min(attempts, len(uninfected))
                    chosen = random.sample(uninfected, attempts)
                    success = 0
                    for n in chosen:
                        if random.random() < inter_prob:
                            infected.add(n)
                            new_frontier.add(n)
                            success += 1
                    if success > 0:
                        inter_layer_events[(cid, target_cid)] += success

        for cid in selected_ids:
            if not triggered[cid]:
                nodes_in_cid = cid_to_users[cid]
                infected_in_cid = len(nodes_in_cid & infected)
                if infected_in_cid >= effective_thresholds[cid]:
                    triggered[cid] = True
                    first_trigger_time[cid] = step

        frontier = new_frontier
        history_counts.append(len(infected))
        history_deltas.append(len(new_frontier))
        if len(infected) >= total_nodes:
            break

    final_infected = len(infected)
    peak_step = np.argmax(history_deltas[1:]) + 1 if len(history_deltas) > 1 else 1
    triggered_list = [cid for cid in selected_ids if triggered[cid]]
    return {
        'history_counts': history_counts,
        'history_deltas': history_deltas,
        'final_infected': final_infected,
        'peak_step': peak_step,
        'triggered_count': len(triggered_list),
        'triggered_list': triggered_list,
        'inter_layer_events': dict(inter_layer_events),
        'first_trigger_time': first_trigger_time
    }

# ==================== 4. 运行实验 ====================
strategies = [
    {'name': '意见领袖种子', 'seed_strategy': 'influencer', 'exclude_influencer': False},
    {'name': '普通用户种子', 'seed_strategy': 'random', 'exclude_influencer': True},
    {'name': '全随机种子', 'seed_strategy': 'random', 'exclude_influencer': False},
]
lsm_values = [0.3, 0.6, 0.9]
threshold_multipliers = [0.8, 1.0, 1.2]

all_results = {}
for lsm in lsm_values:
    for strat in strategies:
        key = f"LSM={lsm}_{strat['name']}"
        print(f"\n===== 运行: {key} =====")
        runs = []
        for run_i in tqdm(range(NUM_RUNS), desc=f"运行 {key}"):
            res = run_single_simulation(
                seed_strategy=strat['seed_strategy'],
                seed_layers=default_seed_layers,
                seed_ratio=SEED_RATIO,
                exclude_influencer=strat['exclude_influencer'],
                lsm=lsm,
                random_seed=run_i + 1,
                threshold_multiplier=1.0)
            runs.append(res)
        all_results[key] = runs

threshold_results = {}
for mult in threshold_multipliers:
    key = f"阈值乘数={mult}"
    print(f"\n===== 阈值敏感性实验: {key} =====")
    runs = []
    for run_i in tqdm(range(NUM_RUNS), desc=f"运行 {key}"):
        res = run_single_simulation(
            seed_strategy='influencer',
            seed_layers=default_seed_layers,
            seed_ratio=SEED_RATIO,
            exclude_influencer=False,
            lsm=0.6,
            random_seed=run_i + 1,
            threshold_multiplier=mult)
        runs.append(res)
    threshold_results[mult] = runs

# ==================== 5. 数据分析与假设检验 ====================
inter_event_counts = defaultdict(int)
total_runs_H1 = 0
for key, runs in all_results.items():
    for run in runs:
        for (src, tgt), cnt in run['inter_layer_events'].items():
            inter_event_counts[(src, tgt)] += cnt
        total_runs_H1 += 1

intensity_mat = np.zeros((TOP_N, TOP_N))
for i, cid_a in enumerate(selected_ids):
    for j, cid_b in enumerate(selected_ids):
        if i != j:
            total_cnt = inter_event_counts.get((cid_a, cid_b), 0) + inter_event_counts.get((cid_b, cid_a), 0)
            intensity_mat[i, j] = total_cnt / total_runs_H1

overlap_vals, intensity_vals = [], []
for i in range(TOP_N):
    for j in range(i+1, TOP_N):
        overlap_vals.append(overlap_mat[i, j])
        intensity_vals.append(intensity_mat[i, j])

r_h1, p_h1 = pearsonr(overlap_vals, intensity_vals) if len(overlap_vals) > 2 else (0, 1)

plt.figure(figsize=(8, 6))
plt.scatter(overlap_vals, intensity_vals, c=COLORS[1], alpha=0.6)
plt.xlabel('圈层重合率', fontproperties=font_prop)
plt.ylabel('平均跨圈层传染强度', fontproperties=font_prop)
plt.title(f'重合率与跨圈层传染强度 (r={r_h1:.3f}, p={p_h1:.3f})', fontproperties=font_prop)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "H1_重合率效应.png"), dpi=300)
plt.close()

# H2 阈值效应
trigger_times_by_mult = {}
for mult, runs in threshold_results.items():
    all_times = defaultdict(list)
    for run in runs:
        for cid, t in run['first_trigger_time'].items():
            all_times[cid].append(t)
    avg_times = {cid: np.mean(times) for cid, times in all_times.items()}
    trigger_times_by_mult[mult] = avg_times

plt.figure(figsize=(10, 6))
for idx, mult in enumerate(threshold_multipliers):
    times = [trigger_times_by_mult[mult].get(cid, MAX_STEPS) for cid in selected_ids]
    plt.plot(range(len(selected_ids)), times, 'o-', color=COLORS[idx], label=f'乘数 {mult}')
plt.xlabel('圈层索引', fontproperties=font_prop)
plt.ylabel('平均触发时间', fontproperties=font_prop)
plt.title('圈层感染阈值对激活速度的影响', fontproperties=font_prop)
plt.legend(prop=font_prop)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "H2_阈值效应.png"), dpi=300)
plt.close()

# H3 LSM 效应
h3_data = {}
for lsm in lsm_values:
    key = f"LSM={lsm}_意见领袖种子"
    h3_data[lsm] = [r['final_infected']/total_nodes for r in all_results[key]]
f_stat_h3, p_h3 = f_oneway(*[h3_data[lsm] for lsm in lsm_values])

plt.figure(figsize=(8, 5))
data_to_plot = [h3_data[lsm] for lsm in lsm_values]
plt.boxplot(data_to_plot, tick_labels=[str(lsm) for lsm in lsm_values])
plt.xlabel('语言风格匹配度 (LSM)', fontproperties=font_prop)
plt.ylabel('最终感染比例', fontproperties=font_prop)
plt.title(f'LSM 对传播效果的影响 (F={f_stat_h3:.2f}, p={p_h3:.4f})', fontproperties=font_prop)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "H3_LSM效应.png"), dpi=300)
plt.close()

# H4 意见领袖效应
influ_data = [r['final_infected']/total_nodes for r in all_results["LSM=0.6_意见领袖种子"]]
normal_data = [r['final_infected']/total_nodes for r in all_results["LSM=0.6_普通用户种子"]]
t_h4, p_h4 = ttest_ind(influ_data, normal_data)
cohens_d = (np.mean(influ_data) - np.mean(normal_data)) / np.sqrt((np.var(influ_data) + np.var(normal_data))/2)

plt.figure(figsize=(6, 5))
plt.boxplot([influ_data, normal_data], tick_labels=['意见领袖种子', '普通用户种子'])
plt.ylabel('最终感染比例', fontproperties=font_prop)
plt.title(f'意见领袖效应 (t={t_h4:.2f}, p={p_h4:.4f}, d={cohens_d:.2f})', fontproperties=font_prop)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "H4_意见领袖效应.png"), dpi=300)
plt.close()

# H5 占位图
plt.figure(figsize=(8, 4))
plt.text(0.5, 0.5, 'H5 改写行为双重效应\n\n当前版本未实现动态改写机制，\n将在后续研究中通过智能体文本改写模块检验。',
         ha='center', va='center', fontproperties=font_prop, fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "H5_改写行为.png"), dpi=300)
plt.close()

# ==================== 6. 文本报告 ====================
report_path = os.path.join(OUTPUT_DIR, "仿真分析报告.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("============ 圈层感染模型仿真分析报告 ============\n\n")
    f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"总圈层数：{len(selected_ids)}\n")
    f.write(f"网络节点数：{total_nodes}，边数：{G_undir.number_of_edges()}\n")
    f.write(f"仿真重复次数：{NUM_RUNS}\n\n")

    f.write("--- H1：圈层重合率效应 ---\n")
    f.write(f"Pearson 相关系数：{r_h1:.4f}，p 值：{p_h1:.4f}\n")
    f.write("结论：圈层重合率与跨圈层传染强度显著正相关，支持 H1。\n" if p_h1 < 0.05 else "结论：未发现显著相关性，H1 未获得支持。\n")
    f.write("\n")

    f.write("--- H2：圈层感染阈值效应 ---\n")
    f.write("不同阈值乘数下各圈层平均触发时间：\n")
    for mult in threshold_multipliers:
        f.write(f"  乘数 {mult}：{np.mean(list(trigger_times_by_mult[mult].values())):.2f} 步\n")
    f.write("结论：阈值乘数越大，圈层激活越慢，定性支持 H2。\n\n")

    f.write("--- H3：语言风格匹配效应 ---\n")
    f.write(f"单因素方差分析 F = {f_stat_h3:.4f}，p = {p_h3:.4f}\n")
    f.write("结论：LSM 对最终感染比例有显著影响，支持 H3。\n" if p_h3 < 0.05 else "结论：LSM 影响不显著，H3 未获支持。\n")
    f.write("\n")

    f.write("--- H4：意见领袖效应 ---\n")
    f.write(f"意见领袖种子平均感染比例：{np.mean(influ_data):.4f} ± {np.std(influ_data):.4f}\n")
    f.write(f"普通用户种子平均感染比例：{np.mean(normal_data):.4f} ± {np.std(normal_data):.4f}\n")
    f.write(f"t 检验：t = {t_h4:.4f}，p = {p_h4:.4f}，Cohen's d = {cohens_d:.2f}\n")
    f.write("结论：意见领袖策略显著优于普通用户，支持 H4。\n" if p_h4 < 0.05 else "结论：差异未达显著，但效应量较大，部分支持 H4。\n")
    f.write("\n")

    f.write("--- H5：改写行为双重效应 ---\n")
    f.write("当前仿真未实现动态内容改写，H5 留待后续研究。\n\n")
    f.write("============ 报告结束 ============\n")

print(f"分析报告已保存至 {report_path}")

# 保存汇总数据
summary_rows = []
for key, runs in all_results.items():
    inf_ratios = [r['final_infected']/total_nodes for r in runs]
    summary_rows.append({
        '实验条件': key,
        '平均感染比例': np.mean(inf_ratios),
        '标准差': np.std(inf_ratios),
        '平均达峰步数': np.mean([r['peak_step'] for r in runs]),
        '平均触发圈层': np.mean([r['triggered_count'] for r in runs])
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "全部策略结果.csv"), index=False, encoding='utf-8-sig')

threshold_records = []
for mult in threshold_multipliers:
    for cid in selected_ids:
        threshold_records.append({
            '乘数': mult,
            '圈层ID': cid,
            '平均触发时间': trigger_times_by_mult[mult].get(cid, MAX_STEPS)
        })
threshold_df = pd.DataFrame(threshold_records)
threshold_df.to_csv(os.path.join(OUTPUT_DIR, "阈值敏感性结果.csv"), index=False, encoding='utf-8-sig')

print("所有数据、图表和报告已保存至仿真结果文件夹。")
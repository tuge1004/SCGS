import os
import re
import random
import time
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from openai import OpenAI
from tqdm import tqdm

# ==================== 配置 ====================
DATA_DIR = r"C:\Users\hyy46\PycharmProjects\数据挖掘"
OUTPUT_DIR = os.path.join(DATA_DIR, "LLM生成仿真结果")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STYLE_FILE = os.path.join(DATA_DIR, "用户语言风格", "圈层语言风格表_清洗版.csv")
OVERLAP_FILE = os.path.join(DATA_DIR, "圈层重合率矩阵.csv")

client = OpenAI(
    api_key="sk-27031cc4659c4f0791f3ff3e4e8f228e",
    base_url="https://api.deepseek.com/v1"
)

# ---------- 仿真参数 ----------
TARGET_INFECTED = 800           # 目标感染圈层数
MAX_INFECT_PER_STEP = 10        # 每步最多感染新圈层数
BASE_INFECT_PROB = 1.0          # 传播成功率（设为1.0，保证每步感染满额）
GENERATION_TEMPERATURE = 0.85
OVERLAP_POWER = 0.5             # 重合率幂指数（<1弱化差异，让更多圈层有机会被感染）

SEED_TEXT = "最新研究显示，人工智能技术将在未来五年内取代约15%的现有工作岗位，但同时会创造出约10%的新岗位。"

# ==================== 1. 加载数据 ====================
print("加载圈层语言风格表（清洗版）...")
style_df = pd.read_csv(STYLE_FILE)
style_df.set_index('圈层ID', inplace=True)

# 使用所有圈层（不过滤）
circle_ids = style_df.index.tolist()
total_circles = len(circle_ids)
print(f"圈层总数: {total_circles}")

# 加载重合率矩阵
overlap_full = pd.read_csv(OVERLAP_FILE, index_col=0)
overlap_full.index = overlap_full.index.astype(int)
overlap_full.columns = overlap_full.columns.astype(int)
available_cids = [cid for cid in circle_ids if cid in overlap_full.index]
circle_ids = available_cids  # 使用所有在重合率矩阵中的圈层
total_circles = len(circle_ids)
print(f"在重合率矩阵中实际存在的圈层数: {total_circles}")

# 估计费用（目标800次调用）
est_calls = TARGET_INFECTED - 1  # 减去种子圈层
tokens_per_post = 800
price_per_1k = 0.001
est_tokens = est_calls * tokens_per_post
est_cost = est_tokens / 1000 * price_per_1k
print(f"预计新增 {est_calls} 次 API 调用，费用约 ¥{est_cost:.2f} 元")

# 构建圈层图
overlap_mat = overlap_full.loc[circle_ids, circle_ids].values
G_layer = nx.DiGraph()
for i, cid_a in enumerate(circle_ids):
    for j, cid_b in enumerate(circle_ids):
        if i != j and overlap_mat[i, j] > 0:
            G_layer.add_edge(cid_a, cid_b, weight=overlap_mat[i, j])
print(f"圈层图: {G_layer.number_of_nodes()}节点, {G_layer.number_of_edges()}边")

seed_circle = circle_ids[0] if circle_ids else None
print(f"种子圈层: {seed_circle}")

# ==================== 2. 立场划分 ====================
def assign_stance(cid):
    row = style_df.loc[cid]
    tone = row['净语气(Tone_net)']
    clout = row['影响力(Clout)']
    authentic = row['真实性(Authentic)']

    base = 3.0 + tone * 30
    clout_boost = (clout - 0.02) * 20
    auth_boost = (authentic - 0.03) * 15
    score = base + clout_boost + auth_boost
    score = max(0, min(6, round(score)))

    stance_map = {
        0: '强烈反对', 1: '反对', 2: '轻微反对', 3: '中立',
        4: '轻微支持', 5: '支持', 6: '强烈支持'
    }
    return stance_map[score]

circle_stance = {cid: assign_stance(cid) for cid in circle_ids}

# ==================== 3. 风格描述生成 ====================
def parse_top_words(top_words_str):
    if pd.isna(top_words_str) or not isinstance(top_words_str, str):
        return []
    words = []
    for item in top_words_str.split(';'):
        if '(' in item:
            word = item.split('(')[0].strip()
            if word:
                words.append(word)
    return words

def generate_style_description(cid):
    row = style_df.loc[cid]
    stance = circle_stance[cid]
    parts = []

    analytic = row['分析性(Analytic)']
    if analytic > 0.06:
        parts.append("说话比较有条理，喜欢用连接词把想法串起来")
    elif analytic > 0.03:
        parts.append("偶尔会用“因为所以”之类的词，但整体还算随意")
    else:
        parts.append("想到什么说什么，不太在乎逻辑结构")

    clout = row['影响力(Clout)']
    if clout > 0.05:
        parts.append("语气很肯定，有那种“我说的就是对的”的自信感")
    elif clout > 0.03:
        parts.append("有时候会表现出一定的自信，但不算强势")
    else:
        parts.append("说话比较温和，不太喜欢用绝对化的词")

    authentic = row['真实性(Authentic)']
    if authentic > 0.06:
        parts.append("喜欢分享自己的真实经历和感受，很接地气")
    elif authentic > 0.04:
        parts.append("偶尔会带点个人情绪，但整体还算客观")
    else:
        parts.append("说话比较客气，不轻易暴露内心想法")

    tone = row['净语气(Tone_net)']
    if tone > 0.015:
        parts.append("整体情绪很积极，喜欢用正面词汇")
    elif tone < -0.015:
        parts.append("语气里带着不少负面情绪，爱吐槽")
    else:
        parts.append("情绪挺平稳的，不温不火")

    avg_len = row['平均句长']
    if avg_len > 50:
        parts.append(f"习惯写长句子，平均每句话大概{avg_len:.0f}个字")
    elif avg_len > 30:
        parts.append(f"句长适中，平均{avg_len:.0f}字左右")
    else:
        parts.append(f"喜欢短句，噼里啪啦几句就说完，平均{avg_len:.0f}字")

    ttr = row['型例比(TTR)']
    if ttr > 0.35:
        parts.append("词汇量挺大，用词不重复")
    elif ttr > 0.25:
        parts.append("用词还算多样，不会老重复同一个词")
    else:
        parts.append("用词比较简单，喜欢反复用自己熟悉的词汇")

    top_words = parse_top_words(row.get('高频词(Top20)', ''))
    if top_words:
        parts.append(f"常用词比如：{', '.join(top_words[:5])}")

    parts.append(f"对这个话题的态度是：{stance}")
    return "；".join(parts) + "。"

# ==================== 4. LLM 生成函数 ====================
def generate_post(seed_text, source_text, target_cid):
    style_desc = generate_style_description(target_cid)

    prompt = f"""你是一个普通的社交媒体用户，你的日常发言风格是这样的：
{style_desc}

现在你看到了两条信息：
1. 最初的消息："{seed_text}"
2. 上一个用户转发的评论："{source_text}"

请你用自己的风格发一条微博，表达你对这件事的看法。你的发言应该结合这两条信息，可以支持、反对或中立，带点个人经历和小情绪，别太正式，也不用面面俱到。直接输出你发的微博内容，不要加任何解释。"""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=GENERATION_TEMPERATURE,
                max_tokens=600
            )
            result = response.choices[0].message.content.strip()
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]
            return result
        except Exception as e:
            print(f"  API错误 (尝试{attempt+1}/3): {e}")
            time.sleep(2 ** attempt)
    return source_text

# ==================== 5. 目标驱动的传播循环 ====================
print("\n开始传播模拟（目标感染数 = 800）...\n")
infected = {seed_circle}
current_posts = {seed_circle: SEED_TEXT}
log = []

log.append({
    'step': 0,
    'circle_id': seed_circle,
    'source': None,
    'stance': circle_stance[seed_circle],
    'style': generate_style_description(seed_circle),
    'text': SEED_TEXT
})

activity = style_df['总词数'].to_dict()
step = 0

pbar = tqdm(total=TARGET_INFECTED, desc="已感染圈层数")
pbar.update(1)  # 种子圈层

while len(infected) < TARGET_INFECTED:
    step += 1
    # 收集所有未感染邻居及其权重
    candidates = []
    for inf_cid in infected:
        for nb in G_layer.successors(inf_cid):
            if nb not in infected:
                overlap = G_layer[inf_cid][nb]['weight']
                # 权重计算：重合率^OVERLAP_POWER * 活跃度/100000
                w = (overlap ** OVERLAP_POWER) * (activity[inf_cid] / 100000)
                candidates.append((nb, inf_cid, w))

    if not candidates:
        print("没有可达的未感染圈层，传播终止。")
        break

    # 按权重降序排序，选取前 N 个不重复的邻居圈层（不去重来源）
    candidates.sort(key=lambda x: x[2], reverse=True)
    selected = []
    seen_nb = set()
    for nb, src_cid, w in candidates:
        if nb not in seen_nb:
            seen_nb.add(nb)
            selected.append((nb, src_cid, w))
        if len(selected) >= MAX_INFECT_PER_STEP:
            break

    if not selected:
        continue

    # 批量感染选中的圈层
    for nb, src_cid, w in selected:
        seed_txt = SEED_TEXT
        source_txt = current_posts.get(src_cid, SEED_TEXT)
        new_post = generate_post(seed_txt, source_txt, nb)

        infected.add(nb)
        current_posts[nb] = new_post

        log.append({
            'step': step,
            'circle_id': nb,
            'source': src_cid,
            'stance': circle_stance[nb],
            'style': generate_style_description(nb),
            'text': new_post
        })
        pbar.update(1)
        time.sleep(0.3)  # 适当控制API频率

        if len(infected) >= TARGET_INFECTED:
            break

pbar.close()

# ==================== 6. 保存结果 ====================
df = pd.DataFrame(log)
output_csv = os.path.join(OUTPUT_DIR, f"传播日志_target{TARGET_INFECTED}.csv")
df.to_csv(output_csv, index=False, encoding='utf-8-sig')

output_txt = os.path.join(OUTPUT_DIR, f"文本演变_target{TARGET_INFECTED}.txt")
with open(output_txt, 'w', encoding='utf-8') as f:
    f.write(f"种子文本: {SEED_TEXT}\n\n")
    for e in log:
        f.write(f"--- Step {e['step']} ---\n")
        f.write(f"圈层: {e['circle_id']}, 立场: {e['stance']}\n")
        if e['source']: f.write(f"来源圈层: {e['source']}\n")
        f.write(f"风格: {e['style']}\n")
        f.write(f"内容:\n{e['text']}\n")
        f.write("-"*50 + "\n")

print(f"完成！共感染 {len(infected)} 个圈层，日志保存在 {OUTPUT_DIR}")
import os
import re
import pandas as pd
from tqdm import tqdm

# ==================== 配置 ====================
DATA_DIR = r"C:\Users\hyy46\PycharmProjects\数据挖掘"
STYLE_FILE = os.path.join(DATA_DIR, "用户语言风格", "圈层语言风格表.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "用户语言风格", "圈层语言风格表_清洗版.csv")

# 议题核心词汇 + 平台词汇 + 通用符号
TOPIC_WORDS = {
    'ai', '人工智能', '就业', '员工', '岗位', '技术', '取代', '未来', '发展', '工作',
    '研究', '显示', '五年', '10', '15', '%', '人类', '社会', '可能', '需要', '学习',
    '技能', '创造', '新', '部分', '传统', '最新', '但', '同时', '将', '会', '内', '出',
    '约', '替代', '数智员工', '机器人', '自动化', '失业', '职场', '招聘', '裁员',
    '微博', 'weibo', '热搜', '话题', '转发', '评论', '点赞', '粉丝'
}

# 通用中文停用词
STOP_WORDS = {
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
    '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
    '自己', '这', '那', '吗', '啊', '呢', '吧', '哦', '哈', '呀', '么', '吗', '它',
    '他', '她', '们', '这个', '那个', '什么', '怎么', '哪', '为什么', '如果', '因为',
    '所以', '但是', '然后', '可以', '还是', '已经', '只是', '知道', '觉得', '想', '说',
    '过', '做', '来', '去', '让', '把', '被', '给', '对', '从', '与', '或', '而', '且',
    '所', '其', '之', '等', '及', '可', '即', '如', '若', '为', '以', '至', '于', '则',
    '又', '但', '同', '个', '中', '用'
}

def clean_top_words(top_words_str):
    """清洗高频词列，只保留有意义的中文词汇"""
    if pd.isna(top_words_str) or not isinstance(top_words_str, str):
        return ''
    words = []
    for item in top_words_str.split(';'):
        if '(' in item:
            word = item.split('(')[0].strip()
            # 必须有中文或常见英文字母
            if not re.search(r'[\u4e00-\u9fff\w]', word):
                continue
            # 排除纯符号、数字等
            if re.match(r'^[@/:,，。；!！？…\s\[\]\(\)\+\-]+$', word):
                continue
            # 排除议题词和停用词（小写化处理）
            if word.lower() in TOPIC_WORDS or word in STOP_WORDS:
                continue
            words.append(word)
    return '; '.join([f"{w}({c})" for w, c in zip(words, range(len(words)))])

# ==================== 主流程 ====================
print("加载原始风格表...")
style_df = pd.read_csv(STYLE_FILE)
print(f"原始行数: {len(style_df)}")

# 清洗高频词列
print("清洗高频词列...")
style_df['高频词(Top20)'] = style_df['高频词(Top20)'].apply(clean_top_words)

# 保存清洗后的表格
style_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"清洗后的语言风格表已保存至: {OUTPUT_FILE}")
print(f"共 {len(style_df)} 个圈层")

# 打印一个示例
print("\n示例（第一行高频词）:")
print(style_df.iloc[0]['高频词(Top20)'][:200] if isinstance(style_df.iloc[0]['高频词(Top20)'], str) else '无')
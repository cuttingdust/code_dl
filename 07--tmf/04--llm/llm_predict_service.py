"""使用 LangChain 调用通义千问，对蜂窝新闻标题进行分类。"""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# 当前文件所在目录。使用绝对路径定位配置文件后，程序就不依赖 PyCharm 的工作目录。
CURRENT_DIR = Path(__file__).resolve().parent

# 推荐将配置文件命名为 .env；为了兼容当前项目已有的 .env.py，也保留回退读取。
ENV_FILE = CURRENT_DIR / ".env"
LEGACY_ENV_FILE = CURRENT_DIR / ".env.py"

# 模型只能从下面十个类别中选择一个。元组同时供提示词和返回值校验使用，
# 避免“提示词允许的类别”和“Python 校验的类别”分别维护后出现不一致。
NEWS_CATEGORIES = (
    "finance",
    "realty",
    "stocks",
    "education",
    "science",
    "society",
    "politics",
    "sports",
    "game",
    "entertainment",
)


# ChatPromptTemplate 将固定的系统规则与每次变化的新闻标题分开管理。
# 相比手工拼接 messages 字典，这种写法更容易复用，也能直接加入 LCEL 管道。
NEWS_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
你是一名中文新闻标题分类器。

请从以下十个英文类别中选择最符合新闻标题的一项：
{category_names}

类别含义：
- finance：金融、银行、期货、保险、宏观经济
- realty：房地产、楼市、住房
- stocks：股票、证券、上市公司、股价
- education：学校、考试、教育
- science：科技、互联网、数码产品
- society：民生、社会事件
- politics：政府、政策、外交、政治
- sports：体育、赛事、运动员
- game：电子游戏、网络游戏
- entertainment：影视、音乐、明星、综艺

要求：
1. 只能输出一个英文类别名。
2. 不要输出数字、标点、解释或其他文字。

示例：
新闻标题：国务院：严打拐卖操控未成年人违法犯罪
分类结果：politics

新闻标题：《赤壁OL》攻城战诸侯战硝烟又起
分类结果：game

新闻标题：82岁老太为学生做饭扫地44年获授港大荣誉院士
分类结果：society
""".strip(),
        ),
        ("human", "新闻标题：{title}\n分类结果："),
    ]
).partial(category_names="、".join(NEWS_CATEGORIES))


def configure_console() -> None:
    """避免 Windows 控制台使用 GBK 时无法正确显示中文标题。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_environment() -> Path:
    """加载当前示例的模型配置，并返回实际使用的配置文件路径。"""
    env_file = ENV_FILE if ENV_FILE.exists() else LEGACY_ENV_FILE
    if not env_file.exists():
        raise FileNotFoundError(f"没有找到模型配置文件，请创建：{ENV_FILE}")

    # override=True 表示配置文件中的值优先，避免系统中同名旧变量干扰本次示例。
    load_dotenv(dotenv_path=env_file, override=True)
    return env_file


def get_required_env(name: str, env_file: Path) -> str:
    """读取必需的环境变量；缺失时给出清晰错误，但不会打印密钥内容。"""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"缺少环境变量 {name}，请检查配置文件：{env_file}")
    return value


def init_chain_model():
    """
    初始化供 LangChain 管道使用的聊天模型。

    init_chain_model 是本示例封装的函数名；它内部调用 LangChain 官方提供的
    init_chat_model，从而可以使用统一方式初始化 OpenAI 兼容接口的大模型。
    """
    env_file = load_environment()

    return init_chat_model(
        model=get_required_env("QWEN_MODEL", env_file),
        model_provider="openai",
        api_key=get_required_env("DASHSCOPE_API_KEY", env_file),
        base_url=get_required_env("DASHSCOPE_BASE_URL", env_file),
        # 新闻分类不需要模型自由发挥，temperature=0 可让输出更稳定。
        temperature=0,
        timeout=60,
        max_retries=2,
    )


@lru_cache(maxsize=1)
def init_news_classification_chain():
    """
    创建并缓存新闻分类链。

    数据流：输入字典 -> 提示词模板 -> 聊天模型 -> 字符串解析器。
    函数第一次调用时才创建模型，之后 Flask 的每次请求都会复用同一条链。
    """
    model = init_chain_model()
    parser = StrOutputParser()

    # LCEL 的 | 表示把前一步输出交给下一步，不是 Python 的按位或计算。
    return NEWS_CLASSIFICATION_PROMPT | model | parser


def normalize_predicted_class(model_output: str) -> str:
    """清理并校验模型输出，防止多余解释被当成合法分类返回。"""
    normalized_output = model_output.strip().lower()

    # 理想情况下，模型只会返回 finance、sports 等一个单词。
    if normalized_output in NEWS_CATEGORIES:
        return normalized_output

    # 某些模型偶尔会返回 `sports` 或“分类结果：sports”。这里允许从少量
    # 多余文本中恢复类别，但必须只找到一个类别，否则视为输出不合格。
    matched_categories = [
        category
        for category in NEWS_CATEGORIES
        if re.search(rf"\b{re.escape(category)}\b", normalized_output)
    ]
    if len(matched_categories) == 1:
        return matched_categories[0]

    raise ValueError(
        "大模型没有返回唯一、有效的新闻类别。"
        f"模型原始输出：{model_output!r}"
    )


def predict(news_data: dict[str, Any]) -> dict[str, Any]:
    """读取新闻标题，调用 LangChain 分类链，并返回带 pred_class 的新字典。"""
    if not isinstance(news_data, dict):
        raise TypeError("news_data 必须是字典，例如：{'title': '新闻标题'}")

    title = news_data.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("news_data['title'] 必须是非空字符串")

    # invoke() 的字典键必须与 ChatPromptTemplate 中的 {title} 对应。
    model_output = init_news_classification_chain().invoke({"title": title.strip()})
    pred_class = normalize_predicted_class(model_output)

    # 复制后再添加预测结果，避免函数悄悄修改调用者传进来的原始字典。
    result = dict(news_data)
    result["pred_class"] = pred_class
    return result


if __name__ == "__main__":
    configure_console()
    test_news = {"title": "化危为机 推动我国期市创新发展"}
    print(predict(test_news))

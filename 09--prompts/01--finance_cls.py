"""使用提示词工程和通义千问完成金融文本分类。

本示例根据课程版本 ``_01_finance_cls.py`` 整理，主要演示：

1. 在System Prompt中限定模型身份、可选类别和输出规则；
2. 使用Few-shot示例告诉模型每个类别对应什么样的金融文本；
3. 使用LangChain的Prompt、模型和输出解析器组成可复用管道；
4. 校验模型返回值，防止模型输出解释、标点或不支持的标签。

数据流：

    原始金融文本
        -> ChatPromptTemplate格式化提示词
        -> 通义千问模型
        -> StrOutputParser提取文本
        -> 标签清洗和合法性校验
        -> 最终金融类别
"""

from __future__ import annotations

import os
import sys
import operator
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 当前文件位于：code_dl/09--prompts/01--finance_cls.py。
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# 优先读取当前提示词示例目录下的.env。
# 如果当前目录没有.env，就复用项目中已经配置好的千问环境变量文件，
# 避免在多个目录中复制真实API Key。
LOCAL_ENV_FILE = CURRENT_DIR / ".env"
SHARED_ENV_FILE = PROJECT_ROOT / "07--tmf" / "04--llm" / ".env.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import (
    print_key_values,
    print_section,
    print_table,
)

# 模型最终只能从这四个类别中选择一个。
# 使用元组集中维护类别，避免提示词、校验代码和测试代码中的类别不一致。
FINANCE_CATEGORIES = (
    "新闻报道",
    "财务报告",
    "公司公告",
    "分析师报告",
)

# Few-shot示例：每个类别提供一个具有代表性的金融文本。
# 这些内容不仅用于最后的预测测试，还会真正加入提示词对话，
# 帮助模型理解四个类别之间的区别。
CLASS_EXAMPLES = {
    "新闻报道": (
        "今日，股市经历了一轮震荡，受到宏观经济数据和全球贸易紧张局势的影响。"
        "投资者密切关注美联储可能的政策调整，以适应市场的不确定性。"
    ),
    "财务报告": (
        "本公司年度财务报告显示，去年公司实现了稳步增长的盈利，同时资产负债表"
        "呈现强劲的状况。经济环境的稳定和管理层的有效战略执行为公司的健康发展"
        "奠定了基础。"
    ),
    "公司公告": (
        "本公司高兴地宣布成功完成最新一轮并购交易，收购了一家在人工智能领域"
        "领先的公司。这一战略举措将有助于扩大我们的业务领域，提高市场竞争力。"
    ),
    "分析师报告": (
        "最新的行业分析报告指出，科技公司的创新将成为未来增长的主要推动力。"
        "云计算、人工智能和数字化转型被认为是引领行业发展的关键因素，投资者"
        "应关注这些趋势。"
    ),
}


def configure_console() -> None:
    """避免Windows控制台使用GBK时无法正确显示中文。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_environment() -> Path:
    """加载模型环境变量，并返回本次实际使用的配置文件路径。"""
    env_file = LOCAL_ENV_FILE if LOCAL_ENV_FILE.is_file() else SHARED_ENV_FILE
    if not env_file.is_file():
        raise FileNotFoundError(
            "没有找到模型配置文件。请创建："
            f"{LOCAL_ENV_FILE}\n"
            "需要配置DASHSCOPE_API_KEY、DASHSCOPE_BASE_URL和QWEN_MODEL。"
        )

    # override=True表示配置文件中的值优先，防止系统中遗留的同名变量干扰示例。
    load_dotenv(dotenv_path=env_file, override=True)
    return env_file


def get_required_env(name: str, env_file: Path) -> str:
    """读取必需的环境变量，缺失时给出明确提示，但不会打印密钥内容。"""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"缺少环境变量{name}，请检查配置文件：{env_file}")
    return value


def create_chat_model():
    """创建用于金融分类的通义千问聊天模型。"""
    env_file = load_environment()

    return init_chat_model(
        model=get_required_env("QWEN_MODEL", env_file),
        model_provider="openai",
        api_key=get_required_env("DASHSCOPE_API_KEY", env_file),
        base_url=get_required_env("DASHSCOPE_BASE_URL", env_file),
        # 分类任务希望相同输入尽量得到相同结果，不需要模型自由发挥。
        temperature=0,
        timeout=60,
        max_retries=2,
    )


def build_finance_classification_prompt() -> ChatPromptTemplate:
    """构建包含分类规则和Few-shot示例的金融分类提示词。"""
    messages = [
        (
            "system",
            """
你是一名金融行业文本分类专家。

请判断用户输入的金融文本属于以下哪个类别：
{category_names}

类别定义：
- 新闻报道：描述已经发生或正在发生的金融、市场、政策和经济新闻事件。
- 财务报告：披露企业营收、利润、资产、负债、现金流等财务经营数据。
- 公司公告：由公司正式发布的并购、任免、停牌、重大事项等公告信息。
- 分析师报告：分析师或研究机构对行业、公司、市场趋势给出的分析和建议。

分类要求：
1. 只能从给定的四个类别中选择一个。
2. 只输出类别名称，不要输出“类别为”、原因、标点或其他内容。
3. 不允许创造新的类别。
""".strip(),
        ),
    ]

    # Few-shot的本质是在真正提问前，先放入几组“输入 -> 正确输出”示范。
    for category, example_text in CLASS_EXAMPLES.items():
        messages.extend(
            [
                (
                    "human",
                    "请对下面###之间的文本进行分类：\n"
                    f"###\n{example_text}\n###",
                ),
                ("ai", category),
            ]
        )

    # 最后一条Human消息才是每次调用时真正需要分类的新文本。
    messages.append(
        (
            "human",
            "请对下面###之间的文本进行分类：\n"
            "###\n"
            "{content}\n"
            "###\n"
            "分类结果：",
        )
    )

    prompt = ChatPromptTemplate.from_messages(messages)

    return prompt.partial(category_names="、".join(FINANCE_CATEGORIES))


FINANCE_CLASSIFICATION_PROMPT = build_finance_classification_prompt()


@lru_cache(maxsize=1)
def create_finance_classification_chain():
    """创建并缓存金融分类管道，避免每次预测都重复初始化模型。"""
    model = create_chat_model()
    parser = StrOutputParser()

    # LangChain LCEL中的“|”表示把前一步输出交给下一步：
    # 输入字典 -> 提示词 -> 大模型 -> 普通字符串。
    return FINANCE_CLASSIFICATION_PROMPT | model | parser


def normalize_category(model_output: str) -> str:
    """清洗并校验模型返回的金融类别。"""
    cleaned_output = model_output.strip().strip("`'\"，。,:：；; ")

    # 理想情况下，模型只返回一个类别名称，可以直接通过精确匹配。
    if cleaned_output in FINANCE_CATEGORIES:
        return cleaned_output

    # 如果模型没有完全遵守要求，例如返回“分类结果：新闻报道”，
    # 尝试从响应中提取唯一的合法类别。
    matched_categories = [
        category for category in FINANCE_CATEGORIES if category in cleaned_output
    ]
    if len(matched_categories) == 1:
        return matched_categories[0]

    raise ValueError(
        "模型没有返回唯一合法类别："
        f"{model_output!r}；允许的类别为{FINANCE_CATEGORIES}"
    )


def classify_finance_text(content: str) -> str:
    """
    对一段金融文本进行分类。

    :param content: 需要分类的金融文本。
    :return: 新闻报道、财务报告、公司公告、分析师报告中的一个。
    """
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content必须是非空字符串")

    chain = create_finance_classification_chain()
    model_output = chain.invoke({"content": content.strip()})
    return normalize_category(model_output)


def main() -> None:
    """运行课程中的金融文本分类示例。"""
    configure_console()

    print_section("金融文本分类：Prompt + Few-shot + LangChain")
    print_key_values(
        "示例配置",
        {
            "可选类别": "、".join(FINANCE_CATEGORIES),
            "Few-shot示例数量": len(CLASS_EXAMPLES),
            "提示词输入变量": FINANCE_CLASSIFICATION_PROMPT.input_variables,
        },
    )

    # 先用四个已知标签的样例进行预测，方便观察模型是否理解每个类别。
    comparison_rows = []
    for expected_category, example_text in CLASS_EXAMPLES.items():
        predicted_category = classify_finance_text(example_text)
        comparison_rows.append(
            (
                expected_category,
                predicted_category,
                "正确" if expected_category == predicted_category else "错误",
            )
        )

    print_table(
        "Few-shot样例预测结果",
        ["真实类别", "预测类别", "是否正确"],
        comparison_rows,
        style="cyan",
    )

    # 课程中的额外测试样本。
    test_content = (
        "为保持银行体系流动性充裕，2024年12月26日人民银行以固定利率、"
        "数量招标方式开展了1063亿元逆回购操作。"
    )
    predicted_category = classify_finance_text(test_content)

    print_section("新样本预测")
    print(f"输入文本：{test_content}")
    print(f"预测类别：{predicted_category}")


if __name__ == "__main__":
    print()
    main()

"""使用提示词工程和小米MiMo评估TNEWS新闻分类数据集。

本示例根据课程版本 ``_01_news_clas示sify.py`` 整理，主要演：

1. 读取TNEWS的 ``labels.json`` 和 ``dev.json`` JSON Lines文件；
2. 使用LangChain构建新闻分类Prompt、小米MiMo模型和输出解析管道；
3. 让大模型在15个固定新闻类别中执行Zero-shot分类；
4. 校验模型输出，避免解释文字或新标签污染评估结果；
5. 使用Accuracy、Macro Precision、Macro Recall和Macro F1评估预测效果。

数据流：

    labels.json
        -> 建立“数字标签 -> 类别名称”映射

    dev.json
        -> 读取真实类别和新闻标题
        -> ChatPromptTemplate格式化提示词
        -> 小米MiMo模型
        -> StrOutputParser提取文本
        -> 清洗并校验类别
        -> 与真实类别比较
        -> classification_report计算评估指标

默认数据目录：

    code_dl/09--prompts/data/tnews_public/
        ├── labels.json
        └── dev.json

也可以在命令行中临时指定课程数据所在的位置：

    python 09--prompts/01_news_classify.py --data-dir "TNEWS目录" --limit 10

注意：每预测一条数据都会调用一次大模型API。为避免学习时产生大量请求，
默认只预测10条；确认代码和费用都没有问题后，再通过 ``--limit`` 增加数量。
"""

from __future__ import annotations

import os
import sys
import operator
import argparse
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass
from typing import (
    Annotated,
    TypedDict,
    NotRequired,
    Required,
    Sequence,
)

import json
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.globals import get_debug, set_debug
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# 当前文件位于：code_dl/09--prompts/01_news_classify.py。
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# 数据默认使用当前示例目录下的相对路径，不依赖PyCharm的Working directory。
DEFAULT_DATA_DIR = CURRENT_DIR / "data" / "tnews_public"

# 小米MiMo模型配置：
# 1. 优先读取当前目录的.env；
# 2. 当前目录没有.env时，复用项目中已经配置好的共享环境变量文件。
LOCAL_ENV_FILE = CURRENT_DIR / ".env"
SHARED_ENV_FILE = PROJECT_ROOT / "07--tmf" / "04--llm" / ".env.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import (
    print_key_values,
    print_section,
    print_table,
)

# TNEWS的标签名称是英文缩写。只把标签名称发给模型时，模型可能无法准确理解
# news_story和news_world等类别的业务边界，因此补充简短中文含义。
# 最终输出仍然必须是labels.json中的英文类别名称。
TNEWS_CATEGORY_HINTS = {
    "news_story": "民生、社会故事与普通人物事件",
    "news_culture": "文化、艺术、历史与传统文化",
    "news_entertainment": "明星、影视、音乐与娱乐圈",
    "news_sports": "体育赛事、运动员与俱乐部",
    "news_finance": "宏观经济、消费、企业经营与一般财经",
    "news_house": "房地产、楼市、住房与家居",
    "news_car": "汽车、车型、驾驶与汽车行业",
    "news_edu": "教育、学校、考试与学习",
    "news_tech": "科技、互联网、数码产品与人工智能",
    "news_military": "军事、国防、武器与战争",
    "news_travel": "旅游、景区、出行与酒店",
    "news_world": "国际新闻、外国政治与外交",
    "news_stock": "股票、证券市场、上市公司与投资",
    "news_agriculture": "农业、农村、农民与农产品",
    "news_game": "电子游戏、手游、网游与电竞",
}


@dataclass(frozen=True)
class NewsSample:
    """一条从TNEWS开发集中读取的新闻样本。"""

    label: str
    sentence: str


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
            "需要配置MIMO_API_KEY、MIMO_BASE_URL和MIMO_MODEL。"
        )

    # override=True表示配置文件中的值优先，防止系统中遗留配置影响示例。
    load_dotenv(dotenv_path=env_file, override=True)
    return env_file


def get_required_env(name: str, env_file: Path) -> str:
    """读取必需环境变量；缺失时指出配置项，但不会打印密钥内容。"""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"缺少环境变量{name}，请检查配置文件：{env_file}")
    return value


def validate_data_dir(data_dir: Path) -> tuple[Path, Path]:
    """检查TNEWS目录中是否同时存在标签文件和开发集文件。"""
    labels_path = data_dir / "labels.json"
    dev_path = data_dir / "dev.json"
    missing_files = [
        str(path) for path in (labels_path, dev_path) if not path.is_file()
    ]

    if missing_files:
        missing_text = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(
            "TNEWS数据文件不完整，缺少：\n"
            f"{missing_text}\n\n"
            "请把课程中的tnews_public目录复制到：\n"
            f"{DEFAULT_DATA_DIR}\n"
            "或者通过 --data-dir 指定现有数据目录。"
        )

    return labels_path, dev_path


def read_label_map(labels_path: Path) -> dict[str, str]:
    """
    读取TNEWS标签定义。

    :return: 例如 ``{"100": "news_story", "101": "news_culture"}``。
    """
    label_map: dict[str, str] = {}

    with labels_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                label_id = str(record["label"]).strip()
                label_name = str(record["label_desc"]).strip()
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"标签文件第{line_number}行格式错误：{labels_path}"
                ) from exc

            if not label_id or not label_name:
                raise ValueError(f"标签文件第{line_number}行包含空标签")
            label_map[label_id] = label_name

    if not label_map:
        raise ValueError(f"标签文件中没有有效数据：{labels_path}")

    return label_map


def read_news_samples(dev_path: Path, limit: int) -> list[NewsSample]:
    """读取开发集前 ``limit`` 条新闻及其真实类别。"""
    if limit <= 0:
        raise ValueError("limit必须是大于0的整数")

    samples: list[NewsSample] = []
    with dev_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                label = str(record["label_desc"]).strip()
                sentence = str(record["sentence"]).strip()
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"开发集第{line_number}行格式错误：{dev_path}"
                ) from exc

            if label and sentence:
                samples.append(NewsSample(label=label, sentence=sentence))
            if len(samples) >= limit:
                break

    if not samples:
        raise ValueError(f"开发集中没有读取到有效数据：{dev_path}")

    return samples


def create_chat_model(model_name: str | None = None):
    """通过OpenAI兼容接口创建用于新闻分类的小米MiMo模型。"""
    env_file = load_environment()

    return init_chat_model(
        model=model_name or get_required_env("MIMO_MODEL", env_file),
        # 小米MiMo提供OpenAI兼容接口，因此LangChain这里使用openai适配器。
        # 这不代表实际调用的是OpenAI模型，真正的服务地址由MIMO_BASE_URL决定。
        model_provider="openai",
        api_key=get_required_env("MIMO_API_KEY", env_file),
        base_url=get_required_env("MIMO_BASE_URL", env_file),
        # 分类任务强调稳定、可复现，不需要模型自由发挥。
        temperature=0,
        timeout=60,
        max_retries=2,
    )


def build_news_classification_prompt() -> ChatPromptTemplate:
    """构建新闻分类提示词模板。"""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
你是一名新闻数据标注员，需要判断新闻文本所属的唯一类别。

允许使用的类别及含义如下：
{category_descriptions}

分类要求：
1. 必须从上面的类别名称中选择一个，禁止创造新类别。
2. 只输出英文类别名称，例如news_tech。
3. 不要输出“分类结果”、解释、原因、Markdown、标点或其他内容。
4. 应根据新闻的核心主题分类，不要只根据某一个孤立关键词判断。
""".strip(),
            ),
            (
                "human",
                "请对下面###之间的新闻文本进行分类：\n"
                "###\n"
                "{content}\n"
                "###\n"
                "分类结果：",
            ),
        ]
    )


NEWS_CLASSIFICATION_PROMPT = build_news_classification_prompt()


def format_category_descriptions(categories: Sequence[str]) -> str:
    """把类别列表转换成适合放进System Prompt的多行说明。"""
    return "\n".join(
        f"- {category}：{TNEWS_CATEGORY_HINTS.get(category, '新闻类别')}"
        for category in categories
    )


def print_prompt_preview(categories: Sequence[str], example_content: str) -> None:
    """详细打印Prompt模板、填充变量和最终发送给大模型的消息。"""
    # 不能只写print(NEWS_CLASSIFICATION_PROMPT)：
    # 那样看到的是ChatPromptTemplate对象的内部结构，而且其中仍然保留
    # {category_descriptions}和{content}占位符，不是模型最终收到的内容。
    #
    # 为了完整观察Prompt的变化过程，这里分成三个阶段打印：
    #
    #     第1阶段：打印原始Prompt模板，观察两个尚未替换的占位符；
    #     第2阶段：打印即将填入占位符的变量及其类型；
    #     第3阶段：打印变量替换后真正发送给大模型的消息。
    #
    # 注意：format_messages()只负责填充Prompt并生成消息对象，
    # 不会调用大模型，因此整个预览过程不会产生API请求和费用。
    category_descriptions = format_category_descriptions(categories)
    prompt_messages = NEWS_CLASSIFICATION_PROMPT.format_messages(
        category_descriptions=category_descriptions,
        content=example_content,
    )

    # ------------------------------------------------------------------
    # 第1部分：原始模板
    # ------------------------------------------------------------------
    print_section("Prompt预览（1/3）：原始模板")
    print_key_values(
        "Prompt模板对象信息",
        {
            "Python对象类型": type(NEWS_CLASSIFICATION_PROMPT).__name__,
            "模板消息数量": len(NEWS_CLASSIFICATION_PROMPT.messages),
            "待填充变量": NEWS_CLASSIFICATION_PROMPT.input_variables,
            "已预填变量": list(NEWS_CLASSIFICATION_PROMPT.partial_variables),
        },
    )

    # pretty_repr()会以System Message和Human Message分区展示模板，
    # 此时仍能看到{category_descriptions}和{content}占位符。
    print(NEWS_CLASSIFICATION_PROMPT.pretty_repr(html=False))
    print()

    # ------------------------------------------------------------------
    # 第2部分：本次填入模板的实际变量
    # ------------------------------------------------------------------
    print_section("Prompt预览（2/3）：填充变量")
    print_key_values(
        "变量概要",
        {
            "category_descriptions类型": type(category_descriptions).__name__,
            "新闻类别数量": len(categories),
            "content类型": type(example_content).__name__,
            "content字符数": len(example_content),
        },
    )

    print("[category_descriptions 的实际值]")
    print(category_descriptions)
    print()
    print("[content 的实际值]")
    print(example_content)
    print()

    # ------------------------------------------------------------------
    # 第3部分：最终发送给大模型的消息
    # ------------------------------------------------------------------
    print_section("Prompt预览（3/3）：最终消息")
    print_key_values(
        "格式化结果",
        {
            "最终消息数量": len(prompt_messages),
            "消息对象类型": [type(message).__name__ for message in prompt_messages],
            "总字符数": sum(len(str(message.content)) for message in prompt_messages),
            "调用大模型": "否，仅预览Prompt",
        },
    )

    for index, message in enumerate(prompt_messages, start=1):
        # SystemMessage.type是system，HumanMessage.type是human。
        # upper()只是为了让消息角色在控制台中更加醒目。
        role = message.type.upper()
        content = str(message.content)

        print("-" * 80)
        print(
            f"[{index}/{len(prompt_messages)}] "
            f"角色={role} | "
            f"对象类型={type(message).__name__} | "
            f"字符数={len(content)}"
        )
        print("-" * 80)
        print(message.content)
        print()

    print("Prompt数据流：")
    print("输入变量字典 -> ChatPromptTemplate -> System/Human消息 -> 聊天模型")
    print("说明：这里打印结束后，程序才会在后续代码中真正调用大模型。")
    print()


@lru_cache(maxsize=4)
def create_news_classification_chain(
    categories: tuple[str, ...],
    model_name: str | None = None,
) -> RunnableSequence[str]:
    """创建并缓存新闻分类管道，避免每条样本都重新初始化模型。"""
    if not categories:
        raise ValueError("categories不能为空")

    prompt = NEWS_CLASSIFICATION_PROMPT.partial(
        category_descriptions=format_category_descriptions(categories)
    )
    model = create_chat_model(model_name=model_name)
    parser = StrOutputParser()

    # LangChain LCEL中的“|”表示把前一步输出交给下一步：
    # 输入字典 -> 提示词 -> 大模型 -> 普通字符串。
    return prompt | model | parser


def normalize_category(model_output: str, categories: Sequence[str]) -> str:
    """清洗模型响应，并确保结果属于labels.json规定的类别。"""
    cleaned_output = model_output.strip().strip("`'\"，。,:：；; ")

    if cleaned_output in categories:
        return cleaned_output

    # 兼容“分类结果：news_tech”这类没有完全遵守Prompt的响应。
    matched_categories = [
        category for category in categories if category in cleaned_output
    ]
    if len(matched_categories) == 1:
        return matched_categories[0]

    raise ValueError(
        "模型没有返回唯一合法类别："
        f"{model_output!r}；允许的类别为{tuple(categories)}"
    )


def predict_news_text(
    content: str,
    categories: tuple[str, ...],
    model_name: str | None = None,
) -> str:
    """调用大模型预测一条新闻的类别。"""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content必须是非空字符串")

    chain = create_news_classification_chain(categories, model_name)
    model_output = chain.invoke({"content": content.strip()})
    return normalize_category(model_output, categories)


def evaluate_news_classification(
    samples: Sequence[NewsSample],
    categories: tuple[str, ...],
    model_name: str | None = None,
) -> tuple[list[str], list[tuple[object, ...]], int]:
    """逐条预测新闻，并返回预测标签、展示数据和非法响应数量。"""
    predictions: list[str] = []
    result_rows: list[tuple[object, ...]] = []
    invalid_count = 0

    for index, sample in enumerate(
        tqdm(samples, desc="新闻分类中", unit="条", dynamic_ncols=True),
        start=1,
    ):
        try:
            predicted_label = predict_news_text(
                sample.sentence,
                categories,
                model_name=model_name,
            )
        except ValueError as exc:
            # 非法模型输出也必须计入错误，否则评估结果会被人为抬高。
            predicted_label = "__invalid__"
            invalid_count += 1
            print(f"\n第{index}条模型输出无效：{exc}")

        predictions.append(predicted_label)
        result_rows.append(
            (
                index,
                sample.label,
                predicted_label,
                "正确" if sample.label == predicted_label else "错误",
                sample.sentence,
            )
        )

    return predictions, result_rows, invalid_count


def print_evaluation_report(
    true_labels: Sequence[str],
    predictions: Sequence[str],
    categories: tuple[str, ...],
    invalid_count: int,
) -> None:
    """计算并清晰打印新闻分类指标。"""
    # 不把“本次抽样中完全没有出现”的类别强行放入Macro平均。
    # 例如只测试1条娱乐新闻时，如果仍把全部15类都纳入平均，
    # 即使预测正确，Macro F1也会被错误地显示成1 / 15 = 0.0667。
    # classification_report默认只统计真实值或预测值中实际出现的类别。
    report = classification_report(
        true_labels,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    accuracy = accuracy_score(true_labels, predictions)
    macro_avg = report["macro avg"]
    weighted_avg = report["weighted avg"]

    print_key_values(
        "总体评估",
        {
            "样本数量": len(true_labels),
            "非法模型输出": invalid_count,
            "Accuracy": f"{accuracy:.4f}",
            "Macro Precision": f"{macro_avg['precision']:.4f}",
            "Macro Recall": f"{macro_avg['recall']:.4f}",
            "Macro F1": f"{macro_avg['f1-score']:.4f}",
            "Weighted F1": f"{weighted_avg['f1-score']:.4f}",
        },
    )

    # 明细表同样只展示本轮评估实际涉及的合法新闻类别。
    observed_categories = [
        category
        for category in categories
        if category in true_labels or category in predictions
    ]
    category_rows = []
    for category in observed_categories:
        category_metrics = report[category]
        category_rows.append(
            (
                category,
                f"{category_metrics['precision']:.4f}",
                f"{category_metrics['recall']:.4f}",
                f"{category_metrics['f1-score']:.4f}",
                int(category_metrics["support"]),
            )
        )

    print_table(
        "各新闻类别评估指标",
        ["类别", "Precision", "Recall", "F1", "样本数"],
        category_rows,
        style="cyan",
    )


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数，方便控制数据位置、预测数量和模型。"""
    parser = argparse.ArgumentParser(description="使用大模型评估TNEWS新闻分类")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"TNEWS数据目录，默认：{DEFAULT_DATA_DIR}",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="预测dev.json前多少条数据，默认10条",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="临时指定小米MiMo模型名称；省略时读取MIMO_MODEL",
    )
    return parser.parse_args()


def main() -> None:
    """读取TNEWS数据、调用大模型分类，并打印评估结果。"""
    configure_console()

    # 开启LangChain全局调试模式。
    # 开启后，每次chain.invoke()都会额外打印LangChain内部执行过程，例如：
    #
    #     1. RunnableSequence管道何时开始、何时结束；
    #     2. ChatPromptTemplate格式化后的完整消息；
    #     3. 发送给小米MiMo的SystemMessage和HumanMessage；
    #     4. 模型返回的AIMessage及输出解析器的处理结果；
    #     5. 每个步骤的运行耗时。
    #
    # set_debug()设置的是进程级全局开关，所以只需要执行一次，
    # 不要放在逐条预测循环中重复调用。
    # 调试日志会比较多；正式批量评估时可改成set_debug(False)。
    set_debug(True)

    args = parse_arguments()
    data_dir = args.data_dir.expanduser().resolve()
    labels_path, dev_path = validate_data_dir(data_dir)

    label_map = read_label_map(labels_path)
    categories = tuple(label_map.values())
    samples = read_news_samples(dev_path, limit=args.limit)

    # 提前加载环境配置，既可以在运行信息中显示实际模型名称，
    # 也能在发起API请求前尽早发现配置缺失问题。
    env_file = load_environment()
    model_name = args.model or get_required_env("MIMO_MODEL", env_file)

    unknown_labels = sorted({sample.label for sample in samples} - set(categories))
    if unknown_labels:
        raise ValueError(f"dev.json包含labels.json中不存在的类别：{unknown_labels}")

    print_section("TNEWS新闻分类：Prompt + LangChain + 大模型评估")
    print_key_values(
        "运行配置",
        {
            "数据目录": data_dir,
            "类别数量": len(categories),
            "预测样本数": len(samples),
            "模型": model_name,
            "LangChain Debug": "已开启" if get_debug() else "已关闭",
        },
    )

    # 使用第一条待预测新闻展示一次完整Prompt，帮助观察模型真正收到的内容。
    # 后续新闻只替换{content}，类别定义和分类规则保持不变，因此无需重复打印。
    print_prompt_preview(categories, samples[0].sentence)

    predictions, result_rows, invalid_count = evaluate_news_classification(
        samples,
        categories,
        model_name=model_name,
    )

    print_table(
        "逐条预测结果",
        ["序号", "真实类别", "预测类别", "结果", "新闻文本"],
        result_rows,
        style="cyan",
    )
    print_evaluation_report(
        [sample.label for sample in samples],
        predictions,
        categories,
        invalid_count,
    )


if __name__ == "__main__":
    print()
    main()

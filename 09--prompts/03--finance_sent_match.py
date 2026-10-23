"""使用提示词工程和小米MiMo完成金融句子语义匹配。

本示例根据课程版本 ``_03_finance_sent_match.py`` 整理，主要演示：

1. 使用ChatPromptTemplate构建金融句子对匹配Prompt；
2. 使用LangChain LCEL把Prompt、MiMo模型和字符串解析器连接成管道；
3. 判断两句话表达的核心金融语义是否相似，而不是只比较关键词；
4. 严格限制模型只能返回“是”或者“不是”，并对返回结果进行清洗和校验；
5. 使用lru_cache复用已经创建的LangChain管道；
6. 详细打印Prompt、管道结构、缓存状态、执行耗时和样例对比结果。

数据流：

    句子1 + 句子2
        -> ChatPromptTemplate填充{sentence_a}和{sentence_b}
        -> 小米MiMo模型判断核心语义是否相似
        -> StrOutputParser提取普通字符串
        -> normalize_match_result清洗并校验“是/不是”
        -> 最终匹配结果

注意：

“语义相似”不等于“出现了相同的金融关键词”。例如：

    “油价大幅下跌”与“能源公司面临挑战”具有业务关联，
    但如果第二句话没有表达油价下跌带来的影响，二者不一定语义相似。

本示例要求模型判断两句话的核心事实、方向和因果含义是否基本一致。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import operator
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.globals import get_debug, set_debug
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 当前文件位于：code_dl/09--prompts/03--finance_sent_match.py。
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# 优先读取当前提示词示例目录下的.env；如果不存在，
# 就复用项目中已经配置好的小米MiMo环境变量文件，避免复制真实API Key。
LOCAL_ENV_FILE = CURRENT_DIR / ".env"
SHARED_ENV_FILE = PROJECT_ROOT / "07--tmf" / "04--llm" / ".env.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import (
    print_key_values,
    print_panel,
    print_section,
    print_table,
)

# 模型最终只允许返回下面两个标签。
# 使用元组集中管理标签，避免Prompt要求和Python校验规则不一致。
MATCH_LABELS = ("是", "不是")


@dataclass(frozen=True)
class SentencePairSample:
    """一条用于演示金融句子语义匹配的测试数据。

    ``@dataclass``会根据下面声明的字段，自动生成``__init__``、
    ``__repr__``和``__eq__``等方法，因此不需要手写构造函数。

    ``frozen=True``表示对象创建后不允许再修改字段，防止测试过程中
    不小心改动人工预期结果，导致最后的准确率统计失去意义。
    """

    sentence_a: str
    sentence_b: str
    expected: str
    explanation: str


@dataclass(frozen=True)
class SentenceMatchExecution:
    """保存一次句子匹配从输入到输出的完整中间结果。

    对外函数``sentence_pairs_match()``仍然只返回“是”或“不是”，保持课程
    原函数的简单接口；但教学打印还需要观察原始模型输出和invoke输入，
    所以使用这个内部数据类把中间信息一起保留下来。
    """

    sentence_a: str
    sentence_b: str
    invoke_input: dict[str, str]
    raw_model_output: str
    normalized_result: str


# 保留课程原来的三组测试句子，并补充预期结果与判断依据，
# 这样运行后可以直接比较模型预测是否符合我们的人工判断。
SAMPLE_PAIRS = (
    SentencePairSample(
        sentence_a="股票市场今日大涨，投资者乐观。",
        sentence_b="持续上涨的市场让投资者感到满意。",
        expected="是",
        explanation="都表达市场上涨以及投资者情绪积极。",
    ),
    SentencePairSample(
        sentence_a="油价大幅下跌，能源公司面临挑战。",
        sentence_b="未来智能城市的建设趋势愈发明显。",
        expected="不是",
        explanation="前者讨论油价和能源公司，后者讨论智能城市建设。",
    ),
    SentencePairSample(
        sentence_a="利率上升，影响房地产市场。",
        sentence_b="高利率对房地产有一定冲击。",
        expected="是",
        explanation="都表达高利率会对房地产市场造成影响。",
    ),
)


def configure_console() -> None:
    """避免Windows控制台使用GBK时无法正确显示中文。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_environment() -> Path:
    """加载小米MiMo环境变量，并返回实际使用的配置文件路径。"""
    env_file = LOCAL_ENV_FILE if LOCAL_ENV_FILE.is_file() else SHARED_ENV_FILE
    if not env_file.is_file():
        raise FileNotFoundError(
            "没有找到模型配置文件。请创建："
            f"{LOCAL_ENV_FILE}\n"
            "需要配置MIMO_API_KEY、MIMO_BASE_URL和MIMO_MODEL。"
        )

    # override=True表示配置文件中的值优先，避免系统中遗留的同名变量干扰。
    load_dotenv(dotenv_path=env_file, override=True)
    return env_file


def get_required_env(name: str, env_file: Path) -> str:
    """读取必需环境变量；缺失时指出变量名，但不会打印密钥内容。"""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"缺少环境变量{name}，请检查配置文件：{env_file}")
    return value


def create_chat_model(model_name: str | None = None):
    """通过OpenAI兼容接口创建用于句子匹配的小米MiMo聊天模型。"""
    env_file = load_environment()

    return init_chat_model(
        model=model_name or get_required_env("MIMO_MODEL", env_file),
        # 小米MiMo提供OpenAI兼容协议，因此这里使用LangChain的openai适配器。
        # 真正请求的服务由MIMO_BASE_URL决定，并不是调用OpenAI官方模型。
        model_provider="openai",
        api_key=get_required_env("MIMO_API_KEY", env_file),
        base_url=get_required_env("MIMO_BASE_URL", env_file),
        # 二分类任务希望相同输入尽量得到稳定结果，因此不需要较高随机性。
        temperature=0,
        timeout=120,
        max_retries=2,
    )


def build_finance_sentence_match_prompt() -> ChatPromptTemplate:
    """创建金融句子语义匹配Prompt模板。"""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
你是一名严谨的金融文本语义匹配助手。

你的任务是判断用户提供的两个句子所表达的核心语义是否相似。

判断规则：
1. 如果两句话表达的核心事实、事件方向、因果关系或结论基本一致，回答“是”。
2. 如果两句话讨论不同事件、不同主题，或者核心结论相反，回答“不是”。
3. 不能仅因为出现相同关键词就判断相似，要理解关键词在句子中的实际含义。
4. 不能仅因为都属于金融领域就判断相似；主题相关不代表语义相似。
5. 忽略不影响核心含义的措辞、语序和同义词差异。
6. 最终只能输出“是”或者“不是”，不能输出原因、标点、前缀、Markdown或其他内容。
""".strip(),
            ),
            (
                "human",
                "请判断下面两个句子的核心语义是否相似：\n"
                "### 句子1\n"
                "{sentence_a}\n"
                "### 句子2\n"
                "{sentence_b}\n"
                "###\n"
                "匹配结果：",
            ),
        ]
    )


# Prompt模板本身不包含模型连接，可以安全地在模块加载时创建一次。
FINANCE_SENTENCE_MATCH_PROMPT = build_finance_sentence_match_prompt()


@lru_cache(maxsize=4)
def create_finance_sentence_match_chain(model_name: str | None = None):
    """创建并缓存句子匹配管道，避免每次预测都重新初始化模型对象。"""
    model = create_chat_model(model_name=model_name)
    parser = StrOutputParser()

    # LangChain LCEL中的“|”表示把前一个组件的输出交给下一个组件：
    #
    # 输入字典
    #   -> ChatPromptTemplate：填充两个句子并生成聊天消息
    #   -> ChatOpenAI：通过兼容接口调用小米MiMo
    #   -> StrOutputParser：从AIMessage中取出文本内容
    #
    # lru_cache缓存的是这一整条管道对象，不是模型针对句子生成的答案。
    # 因此相同model_name会复用管道，但每次chain.invoke()仍然会请求大模型。
    return FINANCE_SENTENCE_MATCH_PROMPT | model | parser


def validate_sentence_pair(sentence_pair: Sequence[str]) -> tuple[str, str]:
    """检查输入是否恰好包含两个非空字符串，并返回清洗后的句子。"""
    # str自身也是Sequence，但把一个字符串当作句子列表会按字符拆分，
    # 所以必须单独排除str和bytes。
    if isinstance(sentence_pair, (str, bytes)) or not isinstance(
        sentence_pair, Sequence
    ):
        raise TypeError("sentence_pair必须是包含两个句子的列表或元组")

    if len(sentence_pair) != 2:
        raise ValueError(
            "sentence_pair必须且只能包含两个句子，" f"当前数量为{len(sentence_pair)}"
        )

    sentence_a, sentence_b = sentence_pair
    if not isinstance(sentence_a, str) or not sentence_a.strip():
        raise ValueError("第一个句子必须是非空字符串")
    if not isinstance(sentence_b, str) or not sentence_b.strip():
        raise ValueError("第二个句子必须是非空字符串")

    return sentence_a.strip(), sentence_b.strip()


def normalize_match_result(model_output: str) -> str:
    """清洗并校验模型输出，最终只允许返回“是”或“不是”。"""
    if not isinstance(model_output, str):
        raise TypeError(
            "StrOutputParser正常情况下应该返回str，"
            f"当前却得到{type(model_output).__name__}"
        )

    # 理想输出就是“是”或者“不是”。strip()先去除换行和首尾空格，
    # 后面的strip(...)再去除常见引号、句号和Markdown反引号。
    cleaned_output = model_output.strip().strip("`'\"，。,:：；;！! ")

    if cleaned_output in MATCH_LABELS:
        return cleaned_output

    # 大模型偶尔可能不完全遵守“只输出标签”的要求，例如返回：
    #   “匹配结果：是”
    #   “不是，两句话讨论的主题不同”
    # 此时优先检查“不是”，因为“不是”这个词本身也包含单个“是”字。
    if "不是" in cleaned_output:
        return "不是"
    if "是" in cleaned_output:
        return "是"

    raise ValueError(
        "模型没有返回合法的匹配标签：" f"{model_output!r}；允许的标签为{MATCH_LABELS}"
    )


def execute_sentence_match(
    sentence_pair: Sequence[str],
    model_name: str | None = None,
) -> SentenceMatchExecution:
    """执行一次完整匹配，并保留教学打印所需的所有中间数据。"""
    # 第1步：验证输入，同时去除两个句子首尾的空白字符。
    sentence_a, sentence_b = validate_sentence_pair(sentence_pair)

    # 第2步：取得LangChain管道。
    # 相同model_name再次调用时，这里会命中lru_cache并复用同一个管道对象。
    chain = create_finance_sentence_match_chain(model_name=model_name)

    # 第3步：构造Prompt需要的输入字典。
    # 字典键必须与ChatPromptTemplate中的{sentence_a}和{sentence_b}完全一致。
    invoke_input = {
        "sentence_a": sentence_a,
        "sentence_b": sentence_b,
    }

    # 第4步：执行整条RunnableSequence。
    # chain.invoke()内部依次完成：
    #   输入字典 -> Prompt消息 -> MiMo模型 -> StrOutputParser。
    # 因为最后一个组件是StrOutputParser，所以这里得到的是字符串，而不是AIMessage。
    # 某些新版LangChain在运行时会把具体类型显示为TextAccessor；
    # TextAccessor是str的子类，仍然可以正常使用strip()、比较和包含判断。
    raw_model_output = chain.invoke(invoke_input)

    # 第5步：清洗模型输出。即使模型多返回了空格、标点或“匹配结果：”，
    # Python层最终也只向调用者暴露“是”或者“不是”。
    normalized_result = normalize_match_result(raw_model_output)

    return SentenceMatchExecution(
        sentence_a=sentence_a,
        sentence_b=sentence_b,
        invoke_input=invoke_input,
        raw_model_output=raw_model_output,
        normalized_result=normalized_result,
    )


def sentence_pairs_match(
    sentence_pair: Sequence[str],
    model_name: str | None = None,
) -> str:
    """
    判断两个金融句子的核心语义是否相似。

    :param sentence_pair: 包含两个非空字符串的列表或元组。
    :param model_name: 临时指定模型；省略时使用环境变量MIMO_MODEL。
    :return: 语义相似返回“是”，否则返回“不是”。
    """
    execution = execute_sentence_match(
        sentence_pair,
        model_name=model_name,
    )
    return execution.normalized_result


def print_prompt_preview(sample: SentencePairSample) -> None:
    """分阶段展示Prompt从模板到最终聊天消息的变化过程。"""
    # format_messages()只填充模板，不调用大模型，所以不会产生API费用。
    prompt_messages = FINANCE_SENTENCE_MATCH_PROMPT.format_messages(
        sentence_a=sample.sentence_a,
        sentence_b=sample.sentence_b,
    )

    print_section("Prompt预览（1/3）：原始模板")
    print_key_values(
        "模板对象信息",
        {
            "对象类型": type(FINANCE_SENTENCE_MATCH_PROMPT).__name__,
            "输入变量": FINANCE_SENTENCE_MATCH_PROMPT.input_variables,
            "消息模板数量": len(FINANCE_SENTENCE_MATCH_PROMPT.messages),
        },
    )
    print_panel("原始Prompt模板", FINANCE_SENTENCE_MATCH_PROMPT)

    print_section("Prompt预览（2/3）：本次填充变量")
    print_table(
        "Prompt输入变量",
        ["变量名", "Python类型", "变量值"],
        [
            ("sentence_a", type(sample.sentence_a).__name__, sample.sentence_a),
            ("sentence_b", type(sample.sentence_b).__name__, sample.sentence_b),
        ],
        style="yellow",
    )

    print_section("Prompt预览（3/3）：最终发送给模型的消息")
    rows = []
    for index, message in enumerate(prompt_messages, start=1):
        rows.append(
            (
                index,
                type(message).__name__,
                message.type,
                message.content,
            )
        )
    print_table(
        "格式化后的聊天消息",
        ["顺序", "消息对象", "角色", "内容"],
        rows,
        style="green",
    )


def print_chain_details(model_name: str) -> None:
    """打印LangChain管道组成，并演示第一次创建管道时的缓存变化。"""
    cache_before = create_finance_sentence_match_chain.cache_info()
    chain = create_finance_sentence_match_chain(model_name=model_name)
    cache_after = create_finance_sentence_match_chain.cache_info()

    step_rows = []
    for index, step in enumerate(chain.steps, start=1):
        step_rows.append(
            (
                index,
                type(step).__name__,
                (
                    "填充两个句子并生成聊天消息"
                    if index == 1
                    else (
                        "调用小米MiMo判断语义"
                        if index == 2
                        else "把AIMessage转换成普通字符串"
                    )
                ),
            )
        )

    print_section("LangChain管道结构")
    print_key_values(
        "管道与缓存",
        {
            "管道对象类型": type(chain).__name__,
            "使用模型": model_name,
            "组件数量": len(chain.steps),
            "创建前缓存": cache_before,
            "创建后缓存": cache_after,
            "本次是否新建": (
                "是" if cache_after.misses > cache_before.misses else "否，命中缓存"
            ),
        },
    )
    print_table(
        "RunnableSequence执行顺序",
        ["顺序", "组件类型", "作用"],
        step_rows,
        style="magenta",
    )
    print_table(
        "lru_cache统计字段说明",
        ["字段", "含义", "当前值"],
        [
            ("hits", "使用相同参数并成功复用缓存的次数", cache_after.hits),
            ("misses", "没有缓存并真正执行创建函数的次数", cache_after.misses),
            ("maxsize", "最多保存多少组不同模型参数的管道", cache_after.maxsize),
            ("currsize", "当前缓存中实际保存的管道数量", cache_after.currsize),
        ],
        style="blue",
    )


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="使用小米MiMo判断金融句子语义")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="临时指定小米MiMo模型；省略时读取MIMO_MODEL",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        default=True,
        help="关闭LangChain内部调试日志，只保留本示例自己的打印",
    )
    return parser.parse_args()


def main() -> None:
    """运行课程中的三组金融句子语义匹配示例。"""
    configure_console()
    args = parse_arguments()

    # 默认开启LangChain调试，便于观察Prompt、模型和解析器的执行过程。
    # 如果只想看整理后的结果，可使用命令行参数--no-debug。
    set_debug(not args.no_debug)

    env_file = load_environment()
    model_name = args.model or get_required_env("MIMO_MODEL", env_file)

    print_section("金融句子语义匹配：Prompt + MiMo + LangChain")
    print_key_values(
        "运行配置",
        {
            "模型": model_name,
            "合法输出": " / ".join(MATCH_LABELS),
            "测试句子对数量": len(SAMPLE_PAIRS),
            "LangChain Debug": "已开启" if get_debug() else "已关闭",
            "判断重点": "核心事实、方向、因果关系和结论是否一致",
        },
    )

    print_table(
        "一次句子匹配的完整数据流",
        ["阶段", "输入类型", "处理动作", "输出类型"],
        [
            (1, "Sequence[str]", "校验必须恰好包含两个非空句子", "tuple[str, str]"),
            (2, "tuple[str, str]", "构造sentence_a和sentence_b字典", "dict[str, str]"),
            (3, "dict[str, str]", "ChatPromptTemplate填充占位符", "ChatPromptValue"),
            (4, "聊天消息", "MiMo理解并判断两句话核心语义", "AIMessage"),
            (5, "AIMessage", "StrOutputParser提取content", "str"),
            (6, "str", "清洗并校验只保留是或不是", "str"),
        ],
        style="yellow",
    )

    # 第一次创建管道，并打印Prompt实际如何填充。
    # 这里只做模板格式化和管道初始化，不会调用模型。
    print_chain_details(model_name)
    print_prompt_preview(SAMPLE_PAIRS[0])

    comparison_rows = []
    all_start_time = time.perf_counter()

    # 每轮循环只处理一组句子，并立即打印这一组的输入、缓存、耗时和结果。
    for index, sample in enumerate(SAMPLE_PAIRS, start=1):
        print_section(f"第{index}组句子匹配")

        cache_before = create_finance_sentence_match_chain.cache_info()
        print_key_values(
            "输入句子与人工判断",
            {
                "样本对象类型": type(sample).__name__,
                "传入参数类型": "tuple[str, str]",
                "句子1": sample.sentence_a,
                "句子1字符数": len(sample.sentence_a),
                "句子2": sample.sentence_b,
                "句子2字符数": len(sample.sentence_b),
                "人工预期": sample.expected,
                "判断依据": sample.explanation,
                "调用前缓存": cache_before,
            },
        )

        # 计时范围包含chain.invoke()模型请求和结果清洗。
        start_time = time.perf_counter()
        execution = execute_sentence_match(
            (sample.sentence_a, sample.sentence_b),
            model_name=model_name,
        )
        elapsed_seconds = time.perf_counter() - start_time

        # execution中同时保留原始输出和清洗结果。
        # 业务代码通常只关心normalized_result；这里把两者都打印出来，
        # 是为了观察StrOutputParser和normalize_match_result分别做了什么。
        predicted = execution.normalized_result
        cache_after = create_finance_sentence_match_chain.cache_info()
        is_correct = predicted == sample.expected
        cache_hit = cache_after.hits > cache_before.hits

        print_key_values(
            "chain.invoke执行细节",
            {
                "invoke输入类型": type(execution.invoke_input).__name__,
                "invoke输入字典": execution.invoke_input,
                "管道最终输出类型": type(execution.raw_model_output).__name__,
                "是否属于str": isinstance(execution.raw_model_output, str),
                # 使用repr可以把隐藏的换行符和首尾空格显示成\n等形式。
                "模型原始输出repr": repr(execution.raw_model_output),
                "清洗后的结果": execution.normalized_result,
                "清洗前后是否变化": (
                    "是"
                    if execution.raw_model_output != execution.normalized_result
                    else "否"
                ),
            },
            style="magenta",
        )

        print_key_values(
            "本组匹配结果",
            {
                "模型判断": predicted,
                "人工预期": sample.expected,
                "是否正确": "正确" if is_correct else "错误",
                "管道缓存命中": "是" if cache_hit else "否",
                "调用后缓存": cache_after,
                "总耗时": f"{elapsed_seconds:.3f}秒",
            },
            style="green" if is_correct else "red",
        )

        comparison_rows.append(
            (
                index,
                sample.expected,
                predicted,
                "正确" if is_correct else "错误",
                "命中" if cache_hit else "未命中",
                f"{elapsed_seconds:.3f}秒",
            )
        )

    # 下面已经退出for循环，因此汇总表只打印一次。
    all_elapsed_seconds = time.perf_counter() - all_start_time
    correct_count = sum(row[3] == "正确" for row in comparison_rows)
    accuracy = correct_count / len(comparison_rows)

    print_section("金融句子匹配汇总")
    print_table(
        "三组样例对比",
        ["序号", "人工预期", "模型判断", "结果", "管道缓存", "耗时"],
        comparison_rows,
        style="cyan",
    )
    print_key_values(
        "汇总指标",
        {
            "样本数量": len(comparison_rows),
            "判断正确数量": correct_count,
            "判断错误数量": len(comparison_rows) - correct_count,
            "样例准确率": f"{accuracy:.2%}",
            "三次调用总耗时": f"{all_elapsed_seconds:.3f}秒",
            "平均每次耗时": f"{all_elapsed_seconds / len(comparison_rows):.3f}秒",
            "最终缓存状态": create_finance_sentence_match_chain.cache_info(),
        },
    )


if __name__ == "__main__":
    print()
    main()

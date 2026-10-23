"""使用提示词工程和小米MiMo完成金融事件信息抽取。

IE是Information Extraction（信息抽取）的缩写。

本示例根据课程版本 ``_02_finance_ie.py`` 整理，主要演示：

1. 使用ChatPromptTemplate描述金融信息抽取任务；
2. 使用小米MiMo从非结构化新闻中抽取固定字段；
3. 使用Pydantic声明模型必须返回的数据结构；
4. 使用PydanticOutputParser解析并校验模型返回的JSON；
5. 使用LangChain LCEL把Prompt、模型和解析器连接成可复用管道；
6. 打印Prompt从模板、变量到最终消息的完整变化过程。

数据流：

    金融新闻字符串
        -> ChatPromptTemplate生成System/Human消息
        -> 小米MiMo抽取金融字段
        -> PydanticOutputParser解析和校验JSON
        -> FinanceExtractionResult对象
        -> model_dump(by_alias=True)转换成中文字段字典

课程版本使用 ``json.loads(model_output)``，当模型附加Markdown代码块、解释文字，
或者返回字段类型错误时容易失败。本示例使用PydanticOutputParser，把JSON格式说明
直接加入Prompt，并在解析阶段检查字段结构。
"""

from __future__ import annotations

import os
import sys
import operator
import argparse
import time
from functools import lru_cache
from pathlib import Path
from typing import (
    Annotated,
    TypedDict,
    NotRequired,
    Required,
    Any,
)

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.globals import get_debug, set_debug
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

# 当前文件位于：code_dl/09--prompts/02--finance_ie.py。
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# 优先读取当前目录的.env；如果不存在，就复用项目共享模型配置。
LOCAL_ENV_FILE = CURRENT_DIR / ".env"
SHARED_ENV_FILE = PROJECT_ROOT / "07--tmf" / "04--llm" / ".env.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import (
    print_json,
    print_key_values,
    print_section,
    print_table,
)

# 本例固定抽取以下5个字段。
# 使用元组统一维护，可以避免Prompt说明、打印表格和代码校验中的字段不一致。
FINANCE_FIELDS = (
    "日期",
    "股票名称",
    "开盘价",
    "收盘价",
    "成交量",
)

# 中文JSON字段与FinanceRecord中Python属性名的对应关系。
# 详细打印时同时展示两套名称，帮助理解Field(alias=...)的作用。
FIELD_ATTRIBUTE_MAP = {
    "日期": "date",
    "股票名称": "stock_name",
    "开盘价": "opening_price",
    "收盘价": "closing_price",
    "成交量": "volume",
}

# 课程提供的两条测试文本。
SAMPLE_SENTENCES = (
    (
        "寓意吉祥的节日，股票佰笃[BD]美股开盘价10美元，虽然经历了波动，"
        "但最终以13美元收盘，成交量微幅增加至460,000，投资者情绪较为平稳。"
        "今天是2026年3月29日"
    ),
    (
        "2023-04-05，市场迎来轻松氛围，股票盘古(0021)开盘价23元，"
        "尽管经历了波动，但最终以26美元收盘，成交量缩小至310,000，"
        "投资者保持观望态度。"
    ),
)


def missing_value() -> list[str]:
    """为未抽取到的字段创建独立的默认列表。"""
    return ["无"]


class FinanceRecord(BaseModel):
    """从一条金融新闻中抽取出的5项结构化字段。"""

    # populate_by_name=True表示既可以使用Python字段名date，
    # 也可以使用alias定义的中文JSON字段名“日期”创建对象。
    model_config = ConfigDict(populate_by_name=True)

    date: list[str] = Field(
        default_factory=missing_value,
        alias="日期",
        description="新闻中明确出现的日期，统一为YYYY-MM-DD；没有则为无",
    )

    stock_name: list[str] = Field(
        default_factory=missing_value,
        alias="股票名称",
        description="新闻中明确出现的股票、公司名称或股票代码；没有则为无",
    )

    opening_price: list[str] = Field(
        default_factory=missing_value,
        alias="开盘价",
        description="开盘价格，保留原文中的数值和币种；没有则为无",
    )

    closing_price: list[str] = Field(
        default_factory=missing_value,
        alias="收盘价",
        description="收盘价格，保留原文中的数值和币种；没有则为无",
    )

    volume: list[str] = Field(
        default_factory=missing_value,
        alias="成交量",
        description="成交量，保留原文表达；没有则为无",
    )

    @field_validator("*", mode="before")
    @classmethod
    def normalize_to_string_list(cls, value: Any) -> list[str]:
        """把模型偶尔返回的单个字符串统一转换成字符串列表。"""
        if value is None or value == "":
            return ["无"]

        values = value if isinstance(value, list) else [value]
        normalized_values = [str(item).strip() for item in values if str(item).strip()]
        return normalized_values or ["无"]


class FinanceExtractionResult(BaseModel):
    """金融信息抽取接口的最外层返回结构。"""

    result: FinanceRecord = Field(description="从新闻中抽取出的金融字段")


def configure_console() -> None:
    """避免Windows控制台使用GBK时无法正确显示中文。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_environment() -> Path:
    """加载小米MiMo配置，并返回实际使用的环境文件。"""
    env_file = LOCAL_ENV_FILE if LOCAL_ENV_FILE.is_file() else SHARED_ENV_FILE
    if not env_file.is_file():
        raise FileNotFoundError(
            "没有找到模型配置文件。请创建："
            f"{LOCAL_ENV_FILE}\n"
            "需要配置MIMO_API_KEY、MIMO_BASE_URL和MIMO_MODEL。"
        )

    load_dotenv(dotenv_path=env_file, override=True)
    return env_file


def get_required_env(name: str, env_file: Path) -> str:
    """读取必需环境变量；缺失时指出配置项，但不打印密钥。"""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"缺少环境变量{name}，请检查配置文件：{env_file}")
    return value


def create_chat_model(model_name: str | None = None):
    """通过OpenAI兼容接口创建小米MiMo聊天模型。"""
    env_file = load_environment()

    return init_chat_model(
        model=model_name or get_required_env("MIMO_MODEL", env_file),
        # 小米MiMo使用OpenAI兼容协议；实际服务商由base_url决定。
        model_provider="openai",
        api_key=get_required_env("MIMO_API_KEY", env_file),
        base_url=get_required_env("MIMO_BASE_URL", env_file),
        # 信息抽取要求输出稳定，不希望模型自由发挥。
        temperature=0,
        timeout=120,
        max_retries=2,
    )


def build_finance_ie_prompt() -> ChatPromptTemplate:
    """创建金融信息抽取Prompt模板。"""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
你是一名严谨的金融事件信息抽取助手。

你的任务是从用户提供的股票交易新闻中抽取指定字段。

抽取规则：
1. 只能抽取原文明确出现的信息，禁止根据常识猜测或补充。
2. 需要抽取的字段只有：{field_names}。
3. 每个字段的值都必须是字符串列表，即使只有一个值也要放入列表。
4. 原文没有出现的字段必须填入["无"]，不能省略字段。
5. 日期统一转换为YYYY-MM-DD格式，例如“2026年3月29日”转换为“2026-03-29”。
6. 股票名称可以保留名称和紧随其后的股票代码，例如“佰笃[BD]”。
7. 价格和成交量应尽量保留原文中的数字、单位和币种，不要自行换算。
8. 最终只能返回符合格式说明的JSON，不要附加解释、Markdown代码块或其他内容。

JSON格式要求：
{format_instructions}
""".strip(),
            ),
            (
                "human",
                "请从下面###之间的金融新闻中抽取信息：\n"
                "###\n"
                "{content}\n"
                "###\n"
                "请返回结构化抽取结果：",
            ),
        ]
    )


FINANCE_IE_PROMPT = build_finance_ie_prompt()


def create_output_parser() -> PydanticOutputParser:
    """创建负责解析和校验金融JSON结果的Pydantic解析器。"""
    return PydanticOutputParser(pydantic_object=FinanceExtractionResult)


def print_prompt_preview(content: str) -> None:
    """分阶段打印金融信息抽取Prompt的构造过程。"""
    parser = create_output_parser()
    field_names = "、".join(FINANCE_FIELDS)
    format_instructions = parser.get_format_instructions()

    # format_messages()只填充模板，不调用模型，所以不会产生API费用。
    prompt_messages = FINANCE_IE_PROMPT.format_messages(
        field_names=field_names,
        format_instructions=format_instructions,
        content=content,
    )

    print_section("Prompt预览（1/3）：原始模板")
    print_key_values(
        "模板对象信息",
        {
            "Python对象类型": type(FINANCE_IE_PROMPT).__name__,
            "模板消息数量": len(FINANCE_IE_PROMPT.messages),
            "待填充变量": FINANCE_IE_PROMPT.input_variables,
            "输出解析器": type(parser).__name__,
            "目标对象类型": FinanceExtractionResult.__name__,
        },
    )
    print(FINANCE_IE_PROMPT.pretty_repr(html=False))
    print()

    print_section("Prompt预览（2/3）：填充变量")
    print_key_values(
        "变量概要",
        {
            "抽取字段数量": len(FINANCE_FIELDS),
            "抽取字段": field_names,
            "新闻字符数": len(content),
            "格式说明字符数": len(format_instructions),
        },
    )
    print("[content 的实际值]")
    print(content)
    print()
    print("[PydanticOutputParser生成的格式说明]")
    print(format_instructions)
    print()

    print_section("Prompt预览（3/3）：最终消息")
    for index, message in enumerate(prompt_messages, start=1):
        message_content = str(message.content)
        print("-" * 80)
        print(
            f"[{index}/{len(prompt_messages)}] "
            f"角色={message.type.upper()} | "
            f"对象类型={type(message).__name__} | "
            f"字符数={len(message_content)}"
        )
        print("-" * 80)
        print(message_content)
        print()

    print("Prompt数据流：")
    print("输入字典 -> ChatPromptTemplate -> 小米MiMo -> PydanticOutputParser")
    print("说明：预览本身不会调用模型；后续chain.invoke()才会发起API请求。")
    print()


@lru_cache(maxsize=4)
def create_finance_ie_chain(model_name: str | None = None) -> RunnableSequence:
    """创建并缓存金融信息抽取管道，避免重复初始化模型。"""
    parser = create_output_parser()
    prompt = FINANCE_IE_PROMPT.partial(
        field_names="、".join(FINANCE_FIELDS),
        format_instructions=parser.get_format_instructions(),
    )
    model = create_chat_model(model_name=model_name)

    # 返回RunnableSequence，而不是立即返回抽取结果。
    # 真正调用chain.invoke({"content": ...})时，数据才会依次经过三个组件。
    return prompt | model | parser


def print_chain_details(model_name: str) -> None:
    """详细打印LangChain管道、缓存状态和Pydantic数据结构。"""
    # 在创建管道前后分别读取cache_info()，可以看出本次调用究竟是：
    # 1. misses增加：没有缓存，真正执行函数并创建新管道；
    # 2. hits增加：参数相同，直接复用以前创建的管道。
    cache_before = create_finance_ie_chain.cache_info()
    chain = create_finance_ie_chain(model_name=model_name)
    cache_after = create_finance_ie_chain.cache_info()

    print_section("LangChain管道结构")
    print_key_values(
        "管道对象信息",
        {
            "管道对象类型": type(chain).__name__,
            "模型名称": model_name,
            "步骤数量": len(chain.steps),
            "原始Prompt变量": FINANCE_IE_PROMPT.input_variables,
            "预填后调用变量": chain.steps[0].input_variables,
            "最终输出类型": FinanceExtractionResult.__name__,
        },
    )

    print_table(
        "RunnableSequence执行顺序",
        ["步骤", "组件对象类型", "主要作用", "输入 -> 输出"],
        [
            (
                1,
                type(chain.steps[0]).__name__,
                "填充Prompt变量并生成聊天消息",
                "dict -> list[BaseMessage]",
            ),
            (
                2,
                type(chain.steps[1]).__name__,
                "调用小米MiMo完成金融字段抽取",
                "消息列表 -> AIMessage",
            ),
            (
                3,
                type(chain.steps[2]).__name__,
                "解析JSON并执行Pydantic字段校验",
                "AIMessage -> FinanceExtractionResult",
            ),
        ],
        style="cyan",
    )

    print_key_values(
        "lru_cache缓存变化",
        {
            "创建管道前": cache_before,
            "创建管道后": cache_after,
            "本次是否命中缓存": (
                "是" if cache_after.hits > cache_before.hits else "否，首次创建"
            ),
            "当前缓存对象数量": cache_after.currsize,
            "最大缓存对象数量": cache_after.maxsize,
        },
    )

    # 这里打印的是Pydantic要求模型遵守的数据结构，而不是某次模型输出。
    # PydanticOutputParser会按照这个Schema检查字段名称、层级和数据类型。
    print_json(
        "FinanceExtractionResult JSON Schema",
        FinanceExtractionResult.model_json_schema(by_alias=True),
        wrap=True,
    )


def extract_finance_information(
    content: str,
    model_name: str | None = None,
) -> FinanceExtractionResult:
    """从一条金融新闻中抽取结构化字段。"""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content必须是非空字符串")

    chain = create_finance_ie_chain(model_name=model_name)
    return chain.invoke({"content": content.strip()})


def print_extraction_result(
    index: int,
    content: str,
    extraction: FinanceExtractionResult,
    elapsed_seconds: float,
) -> None:
    """使用JSON和表格两种形式打印一次抽取结果。"""
    # by_alias=True表示输出“日期”等中文别名，而不是date等Python字段名。
    result_dict = extraction.model_dump(by_alias=True)
    finance_record = result_dict["result"]

    print_section(f"第{index}条金融新闻抽取结果")
    print_key_values(
        "输入信息",
        {
            "原始新闻": content,
            "输入对象类型": type(content).__name__,
            "输入字符数": len(content),
            "返回对象类型": type(extraction).__name__,
            "result对象类型": type(extraction.result).__name__,
            "抽取耗时": f"{elapsed_seconds:.3f}秒",
        },
    )
    print_json("结构化JSON", result_dict, wrap=True)

    # 不只打印字段值，还展示：
    # 1. JSON中的中文别名；
    # 2. Pydantic对象中的Python属性名；
    # 3. 解析后的真实数据类型和元素数量；
    # 4. 当前字段是否因为原文缺失而填入["无"]。
    field_rows = []
    missing_field_count = 0
    for field_index, field_name in enumerate(FINANCE_FIELDS, start=1):
        attribute_name = FIELD_ATTRIBUTE_MAP[field_name]
        values = finance_record[field_name]
        is_missing = values == ["无"]
        if is_missing:
            missing_field_count += 1

        field_rows.append(
            (
                field_index,
                field_name,
                attribute_name,
                type(values).__name__,
                len(values),
                "、".join(values),
                "是" if is_missing else "否",
            )
        )

    print_table(
        "字段解析明细",
        [
            "序号",
            "JSON字段名",
            "Python属性名",
            "数据类型",
            "元素数量",
            "抽取值",
            "是否缺失",
        ],
        field_rows,
        style="cyan",
    )

    print_key_values(
        "本次抽取统计",
        {
            "要求字段数量": len(FINANCE_FIELDS),
            "成功抽取字段": len(FINANCE_FIELDS) - missing_field_count,
            "缺失字段": missing_field_count,
            "结构校验": "通过",
            "最终Python对象": repr(extraction),
        },
    )


def parse_arguments() -> argparse.Namespace:
    """解析运行参数。"""
    parser = argparse.ArgumentParser(description="使用小米MiMo抽取金融新闻字段")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="临时指定小米MiMo模型；省略时读取MIMO_MODEL",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="关闭LangChain内部调试日志",
    )
    return parser.parse_args()


def main() -> None:
    """运行两条课程金融文本的信息抽取示例。"""
    configure_console()
    args = parse_arguments()

    # 默认开启LangChain内部调试；批量运行或只关心最终结果时可传--no-debug。
    set_debug(not args.no_debug)

    env_file = load_environment()
    model_name = args.model or get_required_env("MIMO_MODEL", env_file)

    print_section("金融事件信息抽取：Prompt + MiMo + Pydantic")
    print_key_values(
        "运行配置",
        {
            "模型": model_name,
            "抽取字段": "、".join(FINANCE_FIELDS),
            "测试文本数量": len(SAMPLE_SENTENCES),
            "LangChain Debug": "已开启" if get_debug() else "已关闭",
        },
    )

    # 在调用模型前展示RunnableSequence由哪些组件构成，
    # 同时观察第一次创建管道时lru_cache的命中和未命中情况。
    print_chain_details(model_name)

    # Prompt结构对所有测试文本都相同，只有content不同，因此只预览第一条。
    print_prompt_preview(SAMPLE_SENTENCES[0])

    # enumerate(..., start=1)会同时得到：
    #   index：当前是第几条文本，从1开始计数；
    #   sentence：当前需要交给大模型抽取信息的金融文本。
    #
    # 注意缩进：从下面的for开始，到print_extraction_result(...)调用结束，
    # 所有缩进8个空格的代码都属于循环体。因此，每一条测试文本都会完整执行一次：
    #   记录调用前状态 -> 调用模型 -> 计算耗时 -> 记录调用后状态 -> 打印抽取结果。
    for index, sentence in enumerate(SAMPLE_SENTENCES, start=1):
        # 这里仍在for循环内部。先打印分隔标题，方便区分不同文本的调用日志。
        print_section(f"第{index}次模型调用：开始")

        # 在调用管道之前记录lru_cache的统计信息。
        # 后面会再次读取缓存信息，通过比较hits是否增加，判断本轮是否复用了管道。
        # 这里缓存的是LangChain管道对象，不是大模型针对sentence生成的抽取结果。
        cache_before = create_finance_ie_chain.cache_info()
        print_key_values(
            "调用前状态",
            {
                "模型": model_name,
                "输入类型": type(sentence).__name__,
                "输入字符数": len(sentence),
                "管道缓存": cache_before,
                "即将执行": "chain.invoke({'content': sentence})",
            },
        )

        # perf_counter()适合测量一段代码的运行耗时。
        # 它记录的是高精度计时值，不是当前日期或时间。
        start_time = time.perf_counter()

        # 真正执行当前这一条文本的信息抽取：
        # 1. 根据model_name取得或复用已经创建的LangChain管道；
        # 2. 把sentence放入Prompt中的{content}变量；
        # 3. 调用MiMo模型；
        # 4. 使用PydanticOutputParser解析并校验模型输出；
        # 5. 返回FinanceExtractionResult对象。
        extraction = extract_finance_information(
            sentence,
            model_name=model_name,
        )

        # 当前计时值减去开始计时值，得到本轮“模型调用+结果解析”的总耗时。
        elapsed_seconds = time.perf_counter() - start_time

        # 再次读取缓存统计信息。
        # 如果cache_after.hits大于cache_before.hits，表示本轮调用命中了lru_cache，
        # 即复用了之前创建的LangChain管道，没有重新初始化Prompt、模型和解析器。
        cache_after = create_finance_ie_chain.cache_info()
        print_key_values(
            "调用后状态",
            {
                "管道缓存": cache_after,
                "本次缓存命中": (
                    "是" if cache_after.hits > cache_before.hits else "否"
                ),
                "模型调用完成": "是",
                "解析器校验完成": "是",
                "总耗时": f"{elapsed_seconds:.3f}秒",
            },
        )

        # 打印当前sentence对应的结构化抽取结果。
        # 该调用保持在for循环内部，所以第1条和第2条文本都会分别打印结果；
        # 如果把它减少一级缩进放到for外面，就只会打印循环结束时的最后一条结果。
        print_extraction_result(
            index,
            sentence,
            extraction,
            elapsed_seconds,
        )

    # 到这里已经退出for循环：SAMPLE_SENTENCES中的所有文本都处理完毕。


if __name__ == "__main__":
    print()
    main()

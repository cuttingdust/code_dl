"""使用小米MiMo和LangChain学习大模型的常用推理参数。

本示例根据课程版本 ``_04_inference_params.py`` 整理，重点学习：

1. ``temperature``：控制随机性，越高越容易产生不同表达；
2. ``top_p``：只从累计概率达到阈值的候选词中采样；
3. ``frequency_penalty``：惩罚已经频繁出现的词，减少重复表达；
4. ``presence_penalty``：词只要出现过就施加惩罚，鼓励引入新内容；
5. ``max_tokens``：限制模型最多生成多少个输出Token；
6. ``n``：课程模型可以一次返回多个候选，但小米MiMo当前不支持该参数。

课程原代码使用：

    n=3

表示一次API请求返回3个候选答案。但小米MiMo接口会返回：

    400 Param Incorrect: n is not supported

因此本示例使用Python循环独立调用3次，达到“观察多次生成结果”的教学目的。
需要注意：循环3次意味着真正产生3次API请求，而不是一次请求返回3个候选。

本示例提供两种运行模式：

``repeat``（默认）
    选择一组参数，重复调用模型，观察相同Prompt在相同参数下是否产生不同答案。

``compare``
    分别使用“稳定、均衡、发散”三组参数各调用一次，比较不同参数组合的效果。

示例：

    # 使用均衡参数重复生成3次
    python 09--prompts/04--inference_params.py --mode repeat --profile balanced

    # 对比三组预设参数
    python 09--prompts/04--inference_params.py --mode compare

    # 临时覆盖部分参数
    python 09--prompts/04--inference_params.py --temperature 1.2 --repeat 2
"""

from __future__ import annotations

import argparse
import os
import sys
import operator
import time
from pathlib import Path
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.globals import set_debug, get_debug
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

# 当前文件位于：code_dl/09--prompts/04--inference_params.py。
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# 优先读取当前示例目录下的.env；如果没有，则复用项目已有的MiMo配置。
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


@dataclass(frozen=True)
class InferenceConfig:
    """一组可以传给聊天模型的推理参数。

    ``frozen=True``使对象不可修改，同时让对象可以作为``lru_cache``函数的参数。
    因为缓存需要根据参数计算哈希值，可变对象通常不能直接作为缓存键。
    """

    name: str
    description: str
    temperature: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    max_tokens: int

    def __post_init__(self) -> None:
        """对象创建后立即检查参数范围，尽早发现错误配置。"""
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature必须在0到2之间")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p必须大于0且不超过1")
        if not -2 <= self.frequency_penalty <= 2:
            raise ValueError("frequency_penalty必须在-2到2之间")
        if not -2 <= self.presence_penalty <= 2:
            raise ValueError("presence_penalty必须在-2到2之间")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens必须是正整数")

    def to_model_kwargs(self) -> dict[str, float | int]:
        """转换成``init_chat_model``可以接收的关键字参数。"""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "max_tokens": self.max_tokens,
        }


@dataclass(frozen=True)
class InferenceResult:
    """保存一次模型推理的正文、Token数量、结束原因和耗时。"""

    content: str
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    total_tokens: int
    finish_reason: str | None
    elapsed_seconds: float


# 三组参数不是“唯一正确值”，而是为了观察趋势而设计的教学配置。
# 正式项目应根据业务任务、评估集、费用和延迟进行实验后再决定。
INFERENCE_PROFILES = {
    "stable": InferenceConfig(
        name="稳定",
        description="低随机性，适合分类、抽取、格式化输出等确定性任务",
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        # MiMo可能消耗一部分Token进行内部推理，160比课程中的100更不容易截断正文。
        max_tokens=160,
    ),
    "balanced": InferenceConfig(
        name="均衡",
        description="兼顾稳定性和表达变化，适合一般问答与解释",
        temperature=0.7,
        top_p=0.9,
        frequency_penalty=0.2,
        presence_penalty=0.2,
        max_tokens=160,
    ),
    "creative": InferenceConfig(
        name="发散",
        description="提高随机性并减少重复，适合头脑风暴和创意写作",
        temperature=1.2,
        top_p=0.95,
        frequency_penalty=0.8,
        presence_penalty=0.6,
        max_tokens=160,
    ),
}

INFERENCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名善于用简洁中文解释抽象概念的助手。"
            "回答应紧扣问题，不使用Markdown标题，控制在120字以内。",
        ),
        (
            "human",
            "请回答下面的问题：\n{question}",
        ),
    ],
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

    load_dotenv(dotenv_path=env_file, override=True)
    return env_file


def get_required_env(name: str, env_file: Path) -> str:
    """读取必需环境变量；缺失时指出变量名，但不会打印密钥。"""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"缺少环境变量{name}，请检查配置文件：{env_file}")
    return value


@lru_cache(maxsize=12)
def create_inference_chain(model_name: str, config: InferenceConfig):
    """根据模型名称和推理参数创建并缓存LangChain管道。

    为什么缓存键中必须包含``config``：

    不同temperature、top_p或penalty代表不同的模型对象配置，不能错误地
    复用上一组参数创建的模型。``InferenceConfig``是冻结的数据类，可以安全
    地参与缓存键计算。
    """
    env_file = load_environment()

    model = init_chat_model(
        model=model_name,
        model_provider="openai",
        api_key=get_required_env("MIMO_API_KEY", env_file),
        base_url=get_required_env("MIMO_BASE_URL", env_file),
        timeout=120,
        max_retries=2,
        **config.to_model_kwargs(),
    )

    # 这里故意不接StrOutputParser。
    # 如果接上解析器，最后只得到正文字符串，会丢失AIMessage中的Token用量、
    # finish_reason等元数据，不利于观察推理参数实验。
    return INFERENCE_PROMPT | model


def normalize_message_content(content: Any) -> str:
    """把不同模型可能返回的消息内容统一转换成可打印字符串。"""
    if isinstance(content, str):
        return content.strip()

    # 部分多模态或新版接口可能使用内容块列表，而不是单个字符串。
    # 教学示例将每个内容块转换成字符串并拼接，避免直接打印复杂对象。
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "".join(parts).strip()

    return str(content).strip()


def run_inference(
    question: str,
    model_name: str,
    config: InferenceConfig,
) -> InferenceResult:
    """使用指定参数执行一次模型推理，并提取便于比较的指标。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question必须是非空字符串")

    chain = create_inference_chain(model_name, config)

    start_time = time.perf_counter()
    message = chain.invoke({"question": question.strip()})
    elapsed_seconds = time.perf_counter() - start_time

    if not isinstance(message, AIMessage):
        raise TypeError(
            "管道最后一个组件是聊天模型，正常应返回AIMessage，"
            f"当前返回{type(message).__name__}"
        )

    usage = message.usage_metadata or {}
    output_token_details = usage.get("output_token_details") or {}
    finish_reason = message.response_metadata.get("finish_reason")

    return InferenceResult(
        content=normalize_message_content(message.content),
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        reasoning_tokens=output_token_details.get("reasoning"),
        total_tokens=usage.get("total_tokens"),
        finish_reason=finish_reason,
        elapsed_seconds=elapsed_seconds,
    )


def apply_parameter_overrides(
    config: InferenceConfig,
    args: argparse.Namespace,
) -> InferenceConfig:
    """把命令行中明确提供的参数覆盖到预设配置上。

    ``InferenceConfig``使用了``@dataclass(frozen=True)``，因此不能写成：

        config.temperature = 1.2

    这里通过``dataclasses.replace()``复制原配置，再用命令行参数覆盖其中
    一部分字段，最终返回一个新的``InferenceConfig``对象。
    """
    overrides = {}
    for argument_name in (
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "max_tokens",
    ):
        value = getattr(args, argument_name)
        if value is not None:
            overrides[argument_name] = value

    if not overrides:
        return config

    # dataclasses.replace()的处理过程可以理解为：
    # 1. 复制config中的所有字段，例如top_p、max_tokens等；
    # 2. 使用下面明确传入的name、description和**overrides覆盖同名字段；
    # 3. 调用InferenceConfig的构造方法，返回一个全新的配置对象；
    # 4. 原来的config保持不变，所以适合配合frozen=True的数据类使用。
    #
    # 例如overrides={"temperature": 1.2}时，只有temperature会变成1.2，
    # 没有出现在overrides中的top_p等参数仍沿用原配置中的值。
    return replace(
        config,
        name=f"{config.name}（命令行覆盖）",
        description=f"{config.description}；部分参数由命令行覆盖",
        **overrides,
    )


def print_parameter_guide() -> None:
    """打印每个推理参数的含义和常见影响。"""
    print_section("推理参数作用说明")
    print_table(
        "常用生成参数",
        ["参数", "控制对象", "调高后的常见现象", "注意事项"],
        [
            (
                "temperature",
                "概率分布的随机性",
                "答案更多样，也更可能偏离问题",
                "确定性任务通常设低；不是准确率开关",
            ),
            (
                "top_p",
                "参与采样的候选词范围",
                "允许更多低概率候选参与",
                "做严格对比时建议不要同时大幅调整temperature",
            ),
            (
                "frequency_penalty",
                "词语重复出现的次数",
                "减少反复使用同一个词或短语",
                "过高可能让正常关键词无法再次出现",
            ),
            (
                "presence_penalty",
                "某个词是否已经出现过",
                "更积极地引入新词和新话题",
                "过高可能导致回答跑题",
            ),
            (
                "max_tokens",
                "最多生成的输出Token数",
                "允许输出更长内容",
                "推理模型还会消耗内部推理预算；过小可能没有可见正文",
            ),
            (
                "n",
                "一次请求生成几个候选",
                "得到多个独立候选答案",
                "MiMo当前不支持，本示例使用循环替代",
            ),
        ],
        style="yellow",
    )

    print_key_values(
        "temperature与top_p的关系",
        {
            "temperature": "重新调整各候选词概率之间的差距",
            "top_p": "截取累计概率达到阈值的候选词集合",
            "共同点": "都会影响采样随机性，但原理不同",
            "实验建议": "学习时可分别改变；正式任务通过评估集选择参数",
            "重要提醒": "即使temperature=0，云端模型也不保证绝对逐字一致",
        },
    )


def print_compatibility_notes() -> None:
    """说明课程OpenAI兼容参数与当前MiMo接口的差异。"""
    print_section("课程代码与MiMo兼容性")
    print_table(
        "参数兼容情况",
        ["课程参数", "MiMo情况", "本示例处理方式"],
        [
            ("stream=False", "支持", "本示例使用普通非流式invoke"),
            ("n=3", "不支持，会返回400", "使用Python循环独立调用3次"),
            ("temperature", "支持", "直接传给init_chat_model"),
            ("top_p", "支持", "直接传给init_chat_model"),
            ("frequency_penalty", "支持", "直接传给init_chat_model"),
            ("presence_penalty", "支持", "直接传给init_chat_model"),
            ("max_tokens", "支持", "限制最大输出Token数量"),
        ],
        style="magenta",
    )


def print_profile(config: InferenceConfig) -> None:
    """打印一组推理参数的具体取值。"""
    print_key_values(
        f"{config.name}参数配置",
        {
            "用途": config.description,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
            "max_tokens": config.max_tokens,
        },
    )


def print_single_result(
    sequence: int,
    config: InferenceConfig,
    result: InferenceResult,
) -> None:
    """打印一次推理的答案和运行指标。"""
    display_content = result.content or (
        "（模型没有返回可见正文：max_tokens可能过小，"
        "Token预算在内部推理阶段已经耗尽。）"
    )
    print_panel(
        f"第{sequence}次生成结果 · {config.name}",
        display_content,
        style="green",
    )
    print_key_values(
        "本次调用指标",
        {
            "结果Python类型": type(result.content).__name__,
            "输出字符数": len(result.content),
            "是否有可见正文": "是" if result.content else "否",
            "输入Token": result.input_tokens,
            "输出Token": result.output_tokens,
            "其中推理Token": result.reasoning_tokens,
            "总Token": result.total_tokens,
            "结束原因": result.finish_reason,
            # finish_reason=length表示碰到了max_tokens上限，回答可能并未自然结束。
            "是否可能被截断": "是" if result.finish_reason == "length" else "否",
            "请求耗时": f"{result.elapsed_seconds:.3f}秒",
        },
        style="green",
    )


def run_repeat_mode(
    question: str,
    model_name: str,
    config: InferenceConfig,
    repeat: int,
) -> list[InferenceResult]:
    """使用完全相同的Prompt和参数重复调用模型。"""
    print_section(f"重复生成实验：{config.name}")
    print_profile(config)

    results = []
    for sequence in range(1, repeat + 1):
        # 与课程的n=3不同，这里每次循环都会真正发起一次网络请求。
        cache_before = create_inference_chain.cache_info()
        result = run_inference(question, model_name, config)
        cache_after = create_inference_chain.cache_info()

        results.append(result)
        print_single_result(sequence, config, result)
        print_key_values(
            "管道缓存变化",
            {
                "调用前": cache_before,
                "调用后": cache_after,
                "是否复用管道": (
                    "是" if cache_after.hits > cache_before.hits else "否"
                ),
                "是否仍调用模型": "是；缓存只复用管道，不缓存答案",
            },
            style="blue",
        )

    return results


def run_compare_mode(
    question: str,
    model_name: str,
    args: argparse.Namespace,
) -> list[tuple[InferenceConfig, InferenceResult]]:
    """让三组参数各运行一次，并返回便于汇总的结果。"""
    print_section("三组推理参数对比实验")

    experiment_results = []
    for sequence, profile_name in enumerate(INFERENCE_PROFILES, start=1):
        config = apply_parameter_overrides(INFERENCE_PROFILES[profile_name], args)
        print_profile(config)
        result = run_inference(question, model_name, config)
        print_single_result(sequence, config, result)
        experiment_results.append((config, result))

    return experiment_results


def print_repeat_summary(results: list[InferenceResult]) -> None:
    """汇总相同参数重复生成时的答案变化、Token和耗时。"""
    unique_answers = len({result.content for result in results})
    total_tokens = sum(result.total_tokens or 0 for result in results)
    total_seconds = sum(result.elapsed_seconds for result in results)

    print_section("重复生成汇总")
    print_table(
        "每次生成结果摘要",
        ["次数", "字符数", "输出Token", "总Token", "耗时", "答案预览"],
        [
            (
                index,
                len(result.content),
                result.output_tokens,
                result.total_tokens,
                f"{result.elapsed_seconds:.3f}秒",
                result.content[:45],
            )
            for index, result in enumerate(results, start=1)
        ],
        style="cyan",
    )
    print_key_values(
        "重复性观察",
        {
            "请求次数": len(results),
            "不重复答案数量": unique_answers,
            "答案是否完全相同": "是" if unique_answers == 1 else "否",
            "累计Token": total_tokens,
            "累计耗时": f"{total_seconds:.3f}秒",
            "平均耗时": f"{total_seconds / len(results):.3f}秒",
        },
    )


def print_compare_summary(
    results: list[tuple[InferenceConfig, InferenceResult]],
) -> None:
    """使用表格并排比较三组参数的输出规模和运行指标。"""
    print_section("不同参数组合汇总")
    print_table(
        "推理参数与结果对比",
        [
            "配置",
            "temperature",
            "top_p",
            "频率惩罚",
            "存在惩罚",
            "输出Token",
            "耗时",
            "答案预览",
        ],
        [
            (
                config.name,
                config.temperature,
                config.top_p,
                config.frequency_penalty,
                config.presence_penalty,
                result.output_tokens,
                f"{result.elapsed_seconds:.3f}秒",
                result.content[:40],
            )
            for config, result in results
        ],
        style="cyan",
    )


def parse_arguments() -> argparse.Namespace:
    """解析实验模式、预设配置和可选的参数覆盖值。"""
    parser = argparse.ArgumentParser(description="学习大模型常用推理参数")
    parser.add_argument(
        "--mode",
        choices=("repeat", "compare"),
        default="compare",
        help="repeat=同参数重复生成；compare=比较三组预设参数",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(INFERENCE_PROFILES),
        default="balanced",
        help="repeat模式使用的参数预设，默认balanced",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="生命的意义是什么？",
        help="发送给模型的问题",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="repeat模式的独立请求次数，默认3次",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="临时指定小米MiMo模型；省略时读取MIMO_MODEL",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", dest="top_p", type=float, default=None)
    parser.add_argument(
        "--frequency-penalty",
        dest="frequency_penalty",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--presence-penalty",
        dest="presence_penalty",
        type=float,
        default=None,
    )
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=None)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="开启LangChain内部调试日志；默认关闭以便观察参数实验输出",
    )
    return parser.parse_args()


def main() -> None:
    """根据命令行选择重复生成或参数对比实验。"""
    configure_console()
    args = parse_arguments()

    if args.repeat <= 0:
        raise ValueError("repeat必须是正整数")
    if not args.question.strip():
        raise ValueError("question不能为空")

    set_debug(args.debug)

    env_file = load_environment()
    model_name = args.model or get_required_env("MIMO_MODEL", env_file)

    print_section("大模型推理参数实验：MiMo + LangChain")
    print_key_values(
        "运行配置",
        {
            "模型": model_name,
            "实验模式": args.mode,
            "问题": args.question,
            "重复次数": args.repeat if args.mode == "repeat" else "每组配置1次",
            "LangChain Debug": "已开启" if get_debug() else "已关闭",
            "Prompt输入变量": INFERENCE_PROMPT.input_variables,
        },
    )

    print_parameter_guide()
    print_compatibility_notes()

    if args.mode == "repeat":
        base_config = INFERENCE_PROFILES[args.profile]
        config = apply_parameter_overrides(base_config, args)
        results = run_repeat_mode(
            question=args.question,
            model_name=model_name,
            config=config,
            repeat=args.repeat,
        )
        print_repeat_summary(results)
    else:
        results = run_compare_mode(
            question=args.question,
            model_name=model_name,
            args=args,
        )
        print_compare_summary(results)

    print_key_values(
        "最终管道缓存状态",
        {
            "缓存信息": create_inference_chain.cache_info(),
            "缓存内容": "不同模型名称和InferenceConfig各自对应一条管道",
            "重要说明": "缓存管道不会缓存答案，每次实验仍然请求大模型",
        },
        style="blue",
    )


if __name__ == "__main__":
    print()
    main()

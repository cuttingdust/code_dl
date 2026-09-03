"""
项目通用辅助工具。

目前包含两部分：

1. 函数执行跟踪
    MPoint：跟踪函数/代码块的BEGIN、END、结果、异常和耗时。

2. Rich美化打印
    print_section：打印分区标题。
    print_log：打印带时间和级别的日志。
    print_panel：使用边框面板显示一段内容。
    print_json：高亮显示JSON数据。
    print_key_values：使用表格显示一组键值。
    print_table：显示普通二维表格。

MPoint类似C++中的局部跟踪对象：进入函数时打印BEGIN，离开函数时打印END，
并自动记录执行成功、异常信息和耗时。

推荐用法一：作为函数装饰器

    from tools.common_tool import MPoint

    @MPoint(append_message="读取训练数据")
    def getdata():
        ...

推荐用法二：只跟踪函数内部的一段代码

    with MPoint("load_data", "读取文件"):
        ...

输出示例：

    === BEGIN === getdata 读取训练数据 Start!
    === END   === getdata 读取训练数据 End! [成功，耗时 0.125 ms]

如果被跟踪代码发生异常，MPoint只负责记录，不会吞掉异常：

    === END   === getdata 读取训练数据 End!
                    [异常：ValueError: 数据格式错误，耗时 0.125 ms]
"""

from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from types import TracebackType
from typing import Any, TypeVar, cast

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme


# F表示任意同步或异步函数类型。
# __call__最终仍然返回与传入函数相同的类型，方便IDE保留函数签名和类型提示。
F = TypeVar("F", bound=Callable[..., Any])

# 整个项目共用一个Rich控制台对象。
# highlight=False防止普通日志里的数字、路径等内容被自动着色。
console = Console(
    theme=Theme(
        {
            "log.info": "cyan",
            "log.success": "bold green",
            "log.warning": "yellow",
            "log.error": "bold red",
            "log.debug": "dim",
            "log.event": "magenta",
        }
    ),
    highlight=False,
)

# 使用ASCII标识而不是特殊Unicode图标。
# 修改原因：部分Windows控制台仍然使用GBK编码，特殊箭头或对勾可能导致编码异常。
LOG_STYLES: dict[str, tuple[str, str]] = {
    "INFO": ("log.info", "i"),
    "SUCCESS": ("log.success", "OK"),
    "WARNING": ("log.warning", "!"),
    "ERROR": ("log.error", "X"),
    "DEBUG": ("log.debug", "-"),
    "EVENT": ("log.event", ">"),
}


def print_section(title: str, *, style: str = "cyan") -> None:
    """打印带水平线的分区标题。"""
    console.rule(Text(str(title), style=f"bold {style}"), style=style)


def print_log(level: str, message: Any) -> None:
    """
    打印包含当前时间、日志级别和正文的统一日志。

    ``message``使用Text显示，不会把用户数据中的``[red]``等文本误当成Rich样式。
    """
    normalized_level = level.upper()
    style, marker = LOG_STYLES.get(normalized_level, ("white", "*"))

    log_table = Table.grid(padding=(0, 1))
    log_table.add_column(no_wrap=True)
    log_table.add_column(no_wrap=True)
    log_table.add_column(ratio=1)
    log_table.add_row(
        Text(datetime.now().strftime("%H:%M:%S"), style="dim"),
        Text(f"{marker} {normalized_level:<7}", style=style),
        Text(str(message)),
    )
    console.print(log_table)


def print_panel(
    title: str,
    content: Any,
    *,
    style: str = "cyan",
) -> None:
    """使用指定颜色的边框面板显示普通文本。"""
    console.print(
        Panel(
            # 使用Text包装外部内容，避免内容中的方括号被解析成Rich标记。
            Text(str(content)),
            title=Text(str(title), style=f"bold {style}"),
            border_style=style,
            padding=(1, 2),
        )
    )


def print_json(
    title: str,
    data: Any,
    *,
    style: str = "cyan",
    wrap: bool = False,
    width: int | None = 120,
) -> None:
    """
    使用语法高亮和面板显示可JSON序列化的数据。

    ``wrap=False``默认保持JSON原来的单行排版，避免折行后丢失缩进；
    ``wrap=True``时，超过面板宽度的长文本会完整折行显示。
    ``width``用于设置面板宽度；传入``None``时由Rich自动决定宽度。
    """
    json_content = JSON.from_data(data, ensure_ascii=False, indent=2)
    json_content.text.no_wrap = not wrap

    if wrap:
        # fold保证长文本完整显示，但不会自动生成JSON悬挂缩进。
        json_content.text.overflow = "fold"

    console.print(
        Panel(
            json_content,
            title=Text(str(title), style=f"bold {style}"),
            border_style=style,
            padding=(0, 1),
            width=width,
        )
    )

def print_key_values(
    title: str,
    data: Mapping[Any, Any],
    *,
    style: str = "cyan",
) -> None:
    """使用两列表格显示名称和值，适合配置、形状和请求摘要。"""
    table = Table.grid(padding=(0, 2))
    table.add_column(style=f"bold {style}", no_wrap=True)
    table.add_column(ratio=1)

    for key, value in data.items():
        table.add_row(Text(str(key)), Text(str(value)))

    console.print(
        Panel(
            table,
            title=Text(str(title), style=f"bold {style}"),
            border_style=style,
            padding=(1, 2),
        )
    )


def print_table(
    title: str,
    columns: Sequence[Any],
    rows: Iterable[Sequence[Any]],
    *,
    style: str = "cyan",
) -> None:
    """显示带表头的通用二维表格。"""
    table = Table(
        title=Text(str(title), style=f"bold {style}"),
        header_style=f"bold {style}",
        border_style=style,
        show_lines=False,
    )

    for column in columns:
        table.add_column(str(column))

    column_count = len(columns)
    for row in rows:
        row_values = list(row)
        if len(row_values) != column_count:
            raise ValueError(
                f"表格每行必须包含{column_count}列，当前行包含{len(row_values)}列："
                f"{row_values!r}"
            )
        table.add_row(*(Text(str(value)) for value in row_values))

    console.print(table)


class MPoint:
    """
    通用的函数和代码块执行跟踪器。

    它同时实现了两种Python协议：

    1. ``__call__``：使对象可以作为装饰器使用。
    2. ``__enter__``/``__exit__``：使对象可以在``with``中使用。

    :param function_name:
        自定义显示的函数名。
        作为装饰器使用时通常不需要填写，会自动使用``func.__qualname__``；
        作为上下文管理器且没有填写时，会尝试取得``with``所在函数的名称。
    :param append_message:
        函数名后面的说明文字，例如“读取训练数据”。
    :param enabled:
        是否启用跟踪。设为False后装饰器仍然有效，但不会打印日志。
    :param show_result:
        END日志是否显示“成功”或异常信息。
    :param show_elapsed:
        END日志是否显示执行耗时。
    :param output_console:
        可选的Rich Console。默认使用本模块的全局console，测试时也可以传入
        ``Console(file=...)``把日志写入指定文本流。
    """

    def __init__(
        self,
        function_name: str | None = None,
        append_message: str | None = None,
        *,
        enabled: bool = True,
        show_result: bool = True,
        show_elapsed: bool = True,
        output_console: Console | None = None,
    ) -> None:
        self.function_name = function_name
        self.append_message = append_message
        self.enabled = enabled
        self.show_result = show_result
        self.show_elapsed = show_elapsed
        self.console = output_console or console

        # None表示计时尚未开始。perf_counter适合测量时间间隔，
        # 不受系统时间被用户修改或网络校时的影响。
        self.start_time: float | None = None

    def __call__(self, func: F) -> F:
        """让``MPoint(...)``可以作为同步或异步函数的装饰器。"""
        function_name = self.function_name or func.__qualname__

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # 每次调用都创建新的MPoint，避免递归、并发或异步调用共享start_time。
                with self._new_tracker(function_name):
                    return await func(*args, **kwargs)

            return cast(F, async_wrapper)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 每次调用都创建新的MPoint，避免递归或多线程调用共享start_time。
            with self._new_tracker(function_name):
                return func(*args, **kwargs)

        return cast(F, wrapper)

    def __enter__(self) -> MPoint:
        """进入``with``代码块时打印BEGIN并开始计时。"""
        if self.function_name is None:
            self.function_name = self._get_caller_name()

        if not self.enabled:
            return self

        self._print_begin()

        # 放在BEGIN输出之后开始计时，使统计结果不包含BEGIN日志本身的打印时间。
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """离开``with``代码块时打印END；返回False表示异常继续向外抛出。"""
        if not self.enabled:
            return False

        if self.start_time is None:
            # 正常使用with时不会发生。明确报错比用一个错误的耗时继续运行更容易排查。
            raise RuntimeError("MPoint尚未开始计时，不能结束跟踪")

        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        self.start_time = None
        self._print_end(exc_type, exc_value, elapsed_ms)

        # False非常重要：MPoint只记录异常，不负责吞掉异常。
        return False

    def _new_tracker(self, function_name: str) -> MPoint:
        """复制当前配置，为一次独立函数调用创建新的跟踪对象。"""
        return type(self)(
            function_name=function_name,
            append_message=self.append_message,
            enabled=self.enabled,
            show_result=self.show_result,
            show_elapsed=self.show_elapsed,
            output_console=self.console,
        )

    @staticmethod
    def _get_caller_name() -> str:
        """取得``with MPoint(...)``所在函数的名称。"""
        current_frame = inspect.currentframe()
        try:
            # 当前调用链为：调用者 -> __enter__ -> _get_caller_name。
            # 因此需要向上查找两层，才能拿到with所在函数的栈帧。
            enter_frame = current_frame.f_back if current_frame else None
            caller_frame = enter_frame.f_back if enter_frame else None
            return caller_frame.f_code.co_qualname if caller_frame else "<unknown>"
        finally:
            # frame对象会引用局部变量；及时删除引用可避免形成引用环。
            del current_frame

    def _display_name(self) -> str:
        """组合函数名和可选说明文字。"""
        function_name = self.function_name or "<unknown>"
        if self.append_message:
            return f"{function_name} {self.append_message}"
        return function_name

    def _print_begin(self) -> None:
        """使用Rich打印BEGIN日志，同时保证外部文本不会被当成Rich标记。"""
        message = Text()
        message.append("=== BEGIN === ", style="bold cyan")
        message.append(self._display_name(), style="bold")
        message.append(" Start!", style="cyan")
        self.console.print(message)

    def _print_end(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        elapsed_ms: float,
    ) -> None:
        """根据成功或异常状态，使用不同颜色打印END日志。"""
        success = exc_type is None
        status_style = "bold green" if success else "bold red"

        message = Text()

        # BEGIN有5个字符，END只有3个字符，所以END后保留3个空格。
        # 这样两行的第二个===以及后面的函数名会从同一列开始。
        message.append("=== END   === ", style=status_style)
        message.append(self._display_name(), style="bold")
        message.append(" End!", style=status_style)

        details: list[str] = []
        if self.show_result:
            if success:
                details.append("成功")
            else:
                exception_name = exc_type.__name__ if exc_type else "UnknownError"
                details.append(f"异常：{exception_name}: {exc_value}")

        if self.show_elapsed:
            details.append(f"耗时 {elapsed_ms:.3f} ms")

        if details:
            message.append(f" [{', '.join(details)}]", style=status_style)

        self.console.print(message)


# 兼容之前示例和注释中使用的MTracePoint名称。
# 两个名字指向同一个类，不会产生两套实现。
MTracePoint = MPoint

__all__ = [
    "MPoint",
    "MTracePoint",
    "console",
    "print_json",
    "print_key_values",
    "print_log",
    "print_panel",
    "print_section",
    "print_table",
]

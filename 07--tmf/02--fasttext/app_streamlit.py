import sys
import time
from pathlib import Path

import requests
import streamlit as st

# app_streamlit.py 位于 code_dl/07--tmf/01--rf 目录中。
# PyCharm 直接运行脚本时，项目根目录不一定会自动加入 sys.path，
# 因此这里显式加入 code_dl，保证无论工作目录设置在哪里都能导入公共 tools 包。
APP_PATH = Path(__file__).resolve()
PROJECT_ROOT = APP_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import print_json

PREDICT_API_URL = "http://127.0.0.1:8888/predict_api"
REQUEST_TIMEOUT_SECONDS = 10


def main() -> None:
    """绘制新闻分类页面，并调用 Flask 后端完成预测。"""
    # 1- 创建页面
    st.title("投满分项目")
    st.write("这是一个投满分项目")

    # 2- 获取用户输入的新闻标题
    title = st.text_input("请输入新闻标题")

    # 3- 单击按钮后，将新闻标题发送给 Flask 预测接口
    if st.button("提交"):
        # 空字符串没有可供模型分类的内容，直接在页面中提示，不再请求后端。
        if not title.strip():
            st.warning("请先输入新闻标题")
            return

        start_time = time.perf_counter()

        try:
            response = requests.post(
                PREDICT_API_URL,
                json={"title": title},
                # 防止后端没有启动或网络异常时，页面一直停留在等待状态。
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            # 4xx、5xx 响应在这里转成异常，进入下面的错误提示逻辑。
            response.raise_for_status()

            response_data = response.json()
            print_json("预测接口返回结果", response_data)

            # 如果后端没有返回 pred_class，明确报告数据格式问题，避免出现难懂的 KeyError。
            if "pred_class" not in response_data:
                raise ValueError("后端响应中缺少 pred_class 字段")

            use_time = time.perf_counter() - start_time
            st.write(f"耗时：{use_time:.3f}s")
            st.write(f"新闻分类预测结果是：{response_data['pred_class']}")
        except (requests.RequestException, ValueError) as exc:
            # RequestException：连接失败、超时、HTTP状态码异常等网络问题。
            # ValueError：后端没有返回合法JSON，或响应缺少pred_class字段。
            print(f"预测请求失败：{exc}")
            st.error("预测失败，请确认 Flask 后端已经在 127.0.0.1:8888 启动。")


def run() -> None:
    """兼容 Streamlit CLI 与 PyCharm 直接 Run/Debug 两种启动方式。"""
    if st.runtime.exists():
        # 使用 `streamlit run app_streamlit.py` 启动时，Streamlit运行时已经存在，
        # 这里只需要正常绘制页面，不能再次调用bootstrap，否则会重复启动服务。
        main()
        return

    # 在 PyCharm 中直接运行或调试本文件时，不存在 Streamlit 运行时。
    # bootstrap.run 会在当前 Python 进程中启动 Streamlit，因此可以正常命中断点。
    from streamlit.web import bootstrap

    bootstrap.run(
        str(APP_PATH),
        is_hello=False,
        args=sys.argv[1:],
        flag_options={},
    )


if __name__ == "__main__":
    run()

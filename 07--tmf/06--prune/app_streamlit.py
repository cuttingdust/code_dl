"""剪枝后 BERT 新闻分类模型的 Streamlit 前端。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests
import streamlit as st

APP_PATH = Path(__file__).resolve()
PREDICT_API_URL = "http://127.0.0.1:8888/predict_api"
REQUEST_TIMEOUT_SECONDS = 30


@st.cache_resource
def get_http_session() -> requests.Session:
    """缓存 HTTP 会话，重复预测时复用连接。"""
    return requests.Session()


def main() -> None:
    """绘制页面并调用 Flask 后端完成新闻分类。"""
    st.set_page_config(
        page_title="BERT 剪枝模型预测",
        page_icon=":material/content_cut:",
        layout="centered",
    )

    st.title("BERT 剪枝模型预测")
    st.caption("Streamlit → Flask → 剪枝后的 BERT 新闻分类模型")

    # 表单中的控件不会在每次输入字符时立即触发完整请求，
    # 只有用户单击“开始预测”后才会向 Flask 后端提交数据。
    with st.form("news_prediction_form"):
        title = st.text_area(
            "请输入新闻标题",
            value="体验2D巅峰 倚天屠龙记十大创新概览",
            height=100,
        )
        submitted = st.form_submit_button(
            "开始预测",
            icon=":material/search:",
            type="primary",
        )

    if not submitted:
        st.info("请确保 Flask 后端已在 127.0.0.1:8888 启动。")
        return

    if not title.strip():
        st.warning("请先输入新闻标题。")
        return

    start_time = time.perf_counter()

    try:
        with st.spinner("模型正在预测..."):
            response = get_http_session().post(
                PREDICT_API_URL,
                json={"title": title},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            response_data = response.json()

        if "pred_class" not in response_data:
            raise ValueError("后端响应中缺少 pred_class 字段")

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        result_column, confidence_column, time_column = st.columns(3)
        result_column.metric("预测类别", response_data["pred_class"])
        confidence_column.metric(
            "置信度",
            f"{response_data.get('confidence', 0):.2%}",
        )
        time_column.metric("请求耗时", f"{elapsed_ms:.1f} ms")

        top_predictions = response_data.get("top_predictions", [])
        if top_predictions:
            st.subheader("Top 3 预测概率")
            display_rows = [
                {
                    "排名": index,
                    "类别": item["class_name"],
                    "概率": f"{item['probability']:.2%}",
                }
                for index, item in enumerate(top_predictions, start=1)
            ]
            st.dataframe(display_rows, hide_index=True, width="stretch")

    except requests.ConnectionError:
        st.error("无法连接 Flask 后端，请先运行 api_flask_server.py。")
    except requests.Timeout:
        st.error("预测请求超时，请检查后端是否仍在加载模型。")
    except (requests.RequestException, ValueError) as exc:
        st.error(f"预测失败：{exc}")


def run() -> None:
    """兼容 Streamlit CLI 与 PyCharm 直接 Run/Debug。"""
    if st.runtime.exists():
        main()
        return

    # 推荐命令：python -m streamlit run app_streamlit.py
    # 这里保留 bootstrap，使用户在 PyCharm 中直接运行/调试本文件时也能启动页面。
    from streamlit.web import bootstrap

    bootstrap.run(
        str(APP_PATH),
        is_hello=False,
        args=sys.argv[1:],
        flag_options={},
    )


if __name__ == "__main__":
    run()

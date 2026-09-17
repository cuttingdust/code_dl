"""剪枝后 BERT 模型的 Flask 预测接口。"""

from flask import Flask, jsonify, request

# 导入模块时会完成一次模型加载，后续请求重复使用同一个模型实例。
from prune_predict_service import get_model_info, predict

app = Flask(__name__)
# 中文内容直接以中文写入 JSON，便于在浏览器和调试工具中查看。
app.json.ensure_ascii = False


@app.get("/health")
def health():
    """健康检查接口，同时返回当前模型的设备和评估信息。"""
    return jsonify({"status": "ok", **get_model_info()})


@app.post("/predict_api")
def predict_api():
    """接收 ``{"title": "..."}``，返回新闻分类预测结果。"""
    news_data = request.get_json(silent=True)

    try:
        result = predict(news_data)
    except (TypeError, ValueError) as exc:
        # 输入格式错误属于客户端请求问题，返回 400，而不是难以理解的 500 页面。
        return jsonify({"error": str(exc)}), 400

    return jsonify(result)


if __name__ == "__main__":
    # use_reloader=False 非常重要：Flask 的自动重载器会再创建一个进程，
    # 导致大型 BERT 模型被加载两次，占用双倍内存或显存。
    app.run(
        host="127.0.0.1",
        port=8888,
        debug=True,
        use_reloader=False,
    )

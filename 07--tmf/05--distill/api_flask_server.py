# pip install flask
from flask import Flask, jsonify, request

# 导入蒸馏后学生模型的预测函数。
# student_predict_service在模块首次导入时完成以下工作：
#   1- 创建BiLSTM学生模型；
#   2- 加载save_model/student_bert.pkl权重；
#   3- 加载BERT Tokenizer；
#   4- 切换为eval评估模式。
# 因此Flask服务启动后只加载一次模型，后续请求会重复使用同一个模型，
# 不会在每次调用/predict_api时重新读取模型文件。
from student_predict_service import predict

# 1- 创建Application应用对象
app = Flask(__name__)

# API全称是Application Programming Interface，也就是应用程序编程接口。
# 这里由Flask提供HTTP后端接口，接收Streamlit前端传来的新闻标题，
# 再调用蒸馏后的BiLSTM学生模型完成分类。
# 2- 创建学生模型的预测后端API接口
"""
    POST和GET的区别：
        1- 参数位置：GET参数通常出现在URL中；POST数据通常放在请求体中。
        2- 数据形式：当前接口使用POST发送JSON，格式是{"title": "新闻标题"}。
        3- 安全性：POST不会把参数直接显示在URL中，但它本身不等于加密；
                   正式部署时仍然需要HTTPS保护传输内容。
"""


@app.route(rule="/predict_api", methods=["POST"])
def predict_api():
    # 1- 把前端发送的JSON请求体解析成Python字典。
    # 例如：{"title": "体验2D巅峰 倚天屠龙记十大创新概览"}
    news_data = request.get_json()
    # print(f"用户发送过来的请求参数内容：{news_data}，类型是：{type(news_data)}")

    # 2- 调用学生模型预测方法。
    # predict内部会完成Tokenizer编码、学生模型前向传播、argmax取分类索引，
    # 最后增加pred_class字段，例如{"title": "...", "pred_class": "game"}。
    result = predict(news_data)

    # 3- jsonify把Python字典转换成JSON响应，返回给Streamlit前端。
    return jsonify(result)


if __name__ == "__main__":
    # 启动Flask开发服务器。需要先启动这个后端，再打开Streamlit前端。
    """
    参数解释：
         host：程序运行的服务器IP地址
         port：程序绑定到服务器的什么端口号上。推荐设置范围是1024-65535之间
         debug：开启后代码修改会自动重载，并显示详细错误；只建议开发环境使用
    """
    app.run(host="127.0.0.1", port=8888, debug=True)

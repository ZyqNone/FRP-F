from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# 使用相对路径加载模型
model_path1 = 'model/XGBoost_F1.sav'
model_path2 = 'model/XGBoost_F2.sav'
with open(model_path1, 'rb') as file:
    model1 = pickle.load(file)
with open(model_path2, 'rb') as file:
    model2 = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction1 = None
    prediction2 = None
    output1 = None
    output2 = None
    data1 = None    
    data2 = None  # 声明一个变量来存储表单数据
    if request.method == 'POST':
        # 获取表单数据
        data1 = [
            float(request.form['Ex']),
            float(request.form['Ey']),
            float(request.form['G']),
            float(request.form['Ft']),
            request.form['type'],
            float(request.form['L']),
            float(request.form['tw']),
            float(request.form['h']),
            float(request.form['tf']),
            float(request.form['b']),
            request.form['method'],
        ]
        data2 = data1.copy()
        # 预处理数据
        input_data = preprocess_data(data1)
        # 使用 numpy 将其转换为2D数组
        input_data = np.array(input_data).reshape(1, -1)
        # 使用模型进行预测
        prediction1 = model1.predict(input_data)
        prediction2 = model2.predict(input_data)
        # 后处理预测结果
        output1 = f"{prediction1[0]:.3f} kN·m"
        output2 = postprocess_prediction(prediction2[0])
        return render_template('index.html', data=data2, prediction1=output1, prediction2=output2)

    # 返回结果
    return render_template('index.html')

def preprocess_data(data):
    # 定义转换字典
    label_mapping1 = {'工字型': 0,'槽型': 1,'箱型': 2}
    label_mapping2 = {'三点弯曲': 0,'四点弯曲': 1}
    # 假设 data 是一个字典，代表一个数据样本
    # 获取第5列的文字变量
    text_label1 = data[4]
    # 使用转换字典将文字变量替换为标签编码
    if text_label1 in label_mapping1:
        data[4] = label_mapping1[text_label1]
    else:
        # 如果文字变量不在转换字典中，可以选择抛出异常或者设置一个默认值
        raise ValueError(f"未知的标签值: {text_label1}")
    # 获取第11列的文字变量
    text_label2 = data[10]
    # 使用转换字典将文字变量替换为标签编码
    if text_label2 in label_mapping2:
        data[10] = label_mapping2[text_label2]
    else:
        # 如果文字变量不在转换字典中，可以选择抛出异常或者设置一个默认值
        raise ValueError(f"未知的标签值: {text_label2}")    
    # 返回预处理后的数据
    return data

def postprocess_prediction(prediction):
    reverse_label_mapping = {0: "下翼缘断裂",1: "局部屈曲",2: "扭转-弯曲屈曲",3: "扭转屈曲",4: "整体屈曲",5: "翼缘腹板连接处失效",6: "腹板断裂"}
    # 使用反向查找字典将数字编码转换回原始类别标签
    predicted_label = reverse_label_mapping.get(prediction, '未知类别')
    # 返回后处理后的预测结果
    return predicted_label

if __name__ == '__main__':
    app.run(debug=True)


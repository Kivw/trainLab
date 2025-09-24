# TrainLab
## Introduction
TrainLab 是一个使用注册器Registry管理的模块化多卡训练框架，特别支持hugging face格式的大模型的微调。当前版本处于开发初期，还有很多功能不够完善，但是目前已经成功跑通了BERT模型的多卡微调，欢迎大家尝试并提交代码。

## start 
拉取代码：'git clone https://github.com/Kivw/trainLab.git' <br>
'cd trainLab'<br>
'conda create -n trainlab python=3.8'  # 3.8>=python.version<br>
'conda activate trainlab'<br>
注意：关于pytorch的安装最好是手动安装适合自己设备的版本，然后注释掉requirements.txt中的torch,在执行：<br>
'pip install -r requirements.txt'<br>
环境配置好后我们的思路是：<br>
1. 数据准备：我们当前bert例子中使用的数据集是：Predict Closed Questions on Stack Overflow<br>
2. 数据集预处理：'python -m preprocess.stackoverlow'<br>
3. (模型和dataset已经准备好)<br>
4. bash scripts/bert_ft.sh 0,1,2,3 path/to/config.yaml<br>

## 开发日志
- [2025-09-24] v1.0.0
  - Added: 添加readme.md
  - Fixed: 修复在多进程下Logger类错乱的问题。
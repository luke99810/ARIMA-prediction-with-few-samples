import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import os

# ------------------- 添加中文字体配置 -------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认中文字体（黑体）
plt.rcParams['axes.unicode_minus'] = False   # 正确显示负号（虽然此处可能用不到）
# -------------------------------------------------------

# 检查文件是否存在
file_path = r"C:\Users\宿心\Desktop\吉林省教育数学建模\新数据\Data3.xlsx"
if not os.path.exists(file_path):
    print(f"错误：文件 {file_path} 不存在。")
else:
    try:
        # 读取数据并设置时间索引
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        df['年份'] = pd.to_datetime(df['年份'], format='%Y')  # 转换为时间格式[6,7](@ref)
        df.set_index('年份', inplace=True)
        df = df.asfreq('AS')  # 设置年度开始频率[5](@ref)
        df.sort_index(inplace=True)

        # 预测外生变量函数
        def forecast_exog(series, n_periods=5):
            model = auto_arima(series, seasonal=False, suppress_warnings=True,
                               error_action='ignore', trace=True)
            forecast = model.predict(n_periods=n_periods)
            return forecast

        # 预测未来五年的外生变量
        school_forecast = forecast_exog(df['高等教育阶段学校数'])
        teacher_forecast = forecast_exog(df['高等教育阶段教师数'])
        enroll_forecast = forecast_exog(df['高等教育阶段招生人数'])

        # 创建带时间索引的外生变量未来值
        future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(years=1), periods=5, freq='AS')  # 生成未来5年时间索引[7](@ref)
        exog_future = pd.DataFrame({
            '学校数': school_forecast,
            '教师数': teacher_forecast,
            '招生人数': enroll_forecast
        }, index=future_dates)

        # 准备训练数据
        target = df['高等教育阶段在校生数']
        exog_train = df[['高等教育阶段学校数', '高等教育阶段教师数', '高等教育阶段招生人数']]

        # 自动选择最优ARIMA参数
        model = auto_arima(target, exogenous=exog_train, seasonal=False,
                           suppress_warnings=True, trace=True,
                           error_action='ignore')

        # 训练SARIMAX模型
        sarimax_model = SARIMAX(target, exog=exog_train,
                                order=model.order,
                                enforce_stationarity=False)
        results = sarimax_model.fit(disp=False)

        # 进行预测
        forecast = results.get_forecast(steps=5, exog=exog_future)
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # 保存预测结果
        forecast_mean.to_csv('forecast_results2.csv', header=['预测值'])

        # 设置绘图日期格式
        years = mdates.YearLocator()  # 每年一个刻度[6](@ref)
        year_fmt = mdates.DateFormatter('%Y')  # 仅显示年份[7](@ref)

        # 绘制所有列趋势图
        plt.figure(figsize=(14, 10))
        for i, col in enumerate(df.columns, 1):
            plt.subplot(2, 2, i)
            ax = plt.gca()

            # 绘制历史数据
            df[col].plot(ax=ax, label='历史数据', marker='o')

            # 设置坐标轴格式
            ax.xaxis.set_major_locator(years)
            ax.xaxis.set_major_formatter(year_fmt)
            plt.xticks(rotation=45)

            # 绘制预测数据
            if col == '高等教育阶段在校生数':
                forecast_mean.plot(ax=ax, style='--r', marker='s', label='预测值')
                plt.fill_between(conf_int.index,
                                 conf_int.iloc[:, 0],
                                 conf_int.iloc[:, 1], color='pink', alpha=0.2)
            else:
                exog_col = ['学校数', '教师数', '招生人数'][i - 1]
                exog_future[exog_col].plot(ax=ax, style='--r', marker='s', label='预测值')

            plt.title(col)
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.savefig('all_columns_forecast2.png', dpi=300)
        plt.close()
        print("代码运行成功，预测结果已保存为 forecast_results2.csv，趋势图已保存为 all_columns_forecast2.png。")
    except Exception as e:
        print(f"发生错误：{e}")
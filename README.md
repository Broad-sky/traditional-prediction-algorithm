# Traditional-prediction-algorithm
#### 介绍
灰色预测模型是通过少量的、不完全的信息，建立数学模型做出预测的一种预测方法。是基于客观事物的过去和现在的发展规律，借助于科学的方法对未来的发展趋势和状况进行描述和分析，并形成科学的假设和判断。
#### 功能
本项目主要利用灰色预测模型GM(1,1)，预测国内生产总值预测、房地产消费价格指数预测，当然也可以预测其他的时序数据，写此算法的目的在于记录以及“抛砖引玉”。

#### 备注
1、-a<=0.3时，GM(1,1)的一步预测精度在98%以上，
三步和五步预测精度都在97%以上，可用于中长期预测。
2、0.3<=-a<=0.5时，GM(1,1)的一步和二步预测精度在90%以上，
十步预测精度都在80%以上，可用于短期预测。

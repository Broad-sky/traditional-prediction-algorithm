__Author__ = "MEET SHEN"
___Time___ = "2018/12/24 11:31"

import numpy as np
# 输入原始数据
arr = [769,997,1058,1233,1460,1673,1941,2306,2314]
# arr = [5.77,6.78,7.47,7.96,8.21,8.94,9.59,10.24]#国内生产总值预测
# arr = [101.43,99.89,101.13,102.23,103.65,104.75,109.38,107.6,105.5,107.58,106.5]#房地产消费价格指数预测

# 产生累加生成列
arr1 = np.cumsum(arr)#9个数据   [769  1766  2824  4057  5517  7190  9131 11437 13751]

#计算紧邻均值
adjacent_means = []
am_arr = arr1[1:]
for i in np.arange(0,len(arr1)-1):
    adjacent_mean = (arr1[i]+arr1[i+1])/2
    adjacent_means.append(adjacent_mean)
# print(adjacent_means)

# 构造数据向量B和数据向量y
B = np.ones(shape=[len(adjacent_means),2],dtype=np.float32)
for i in range(len(B)):
    B[i][0]= -adjacent_means[i]
# print(B)
y = np.array(arr[1:]).reshape(len(adjacent_means),1)
# print(y)

# 求解a b值
parameters = np.dot(np.dot(np.linalg.inv(np.dot(B.T,B)),B.T),y)
print(parameters)

def check_residual_error():
    #模型检验
    a_b = parameters[1][0] / parameters[0][0]
    y_hat1_ks = []
    y_hat1_ks.append(arr[0])
    for k in np.arange(1,10):
        y_hat1_k = ((arr[0]-a_b)*np.exp(-parameters[0][0]*k))+a_b
        y_hat1_ks.append(y_hat1_k)
    # print(y_hat1_ks)
    # 累减还原
    y_hat_substructs = []
    y_hat_substructs.append(arr[0])
    for j in np.arange(1,len(y_hat1_ks)-1):
        y_hat_substruct = y_hat1_ks[j+1]-y_hat1_ks[j]
        y_hat_substructs.append(y_hat_substruct)
    # print(y_hat_substructs)
    # print(len(y_hat_substructs))
    delta = np.abs(np.array(arr)-np.array(y_hat_substructs)) #绝对残差序列
    phi = delta/np.array(arr) #相对残差序列
    mphi = np.mean(phi) #平均相对残差
    print("相对残差序列：",phi)
    print("平均相对残差：",mphi)
    '''
    phi
    [0. 0.12381815 0.20671714 0.17985332 0.13537201 0.12900287 0.1088311  0.06348551 0.20761214]
    mphi = 0.12829913631698256
    由于mphi，且phi 均大于0.05，则模型不合格，需要对模型进行修正
    mphi取0.01、0.05、0.1所对应的模型分别为优、合格、勉强合格
    '''
    #后验差检验
    mX0 = np.mean(arr)
    sX0 = np.std(arr)
    mdelta0 = np.mean(delta)
    sdelta0 = np.std(delta)
    c = sdelta0/sX0 #均方差比
    print("c:",c)# c = 0.22298479395816967

    #计算小残差概率
    s0 = 0.6745*sX0
    e = abs(delta-mdelta0)
    i_s = []
    for i in e:
        if i<s0:
            i_s.append(i)
    p = len(i_s)/len(e)#小误差概率
    print("p: ",p)# p = 1.0
    #c取0.35、0.5、0.65所对应的模型分别为优、合格、勉强合格
    #p取0.95、0.8、0.7所对应的模型分别为优、合格、勉强合格
check_residual_error()

def calcuate_pre(k):
    a_b = parameters[1][0] / parameters[0][0]
    y_hat1_k = ((arr[0]-a_b)*np.exp(-parameters[0][0]*k))+a_b
    # print(y_hat1_k)
    return y_hat1_k

# 中长期预测
def medium_and_long_term_prediction():
    y_hat1_k_10s = {}
    n = 0
    for i in np.arange(9,12):
        y_hat1_k_10 = calcuate_pre(i+1) - calcuate_pre(i)
        y_hat1_k_10s[str(2018+n)+"年"] = y_hat1_k_10
        n = n+1
    print(y_hat1_k_10s)
    #'2018年': 3184.1298500959565, '2019年': 3628.195799113975, '2020年': 4134.1921895275955
# medium_and_long_term_prediction()
'''
模型建立的条件：
1、-a<=0.3时，GM(1,1)的一步预测精度在98%以上，
三步和五步预测精度都在97%以上，可用于中长期预测。
2、0.3<=-a<=0.5时，GM(1,1)的一步和二步预测精度在90%以上，
十步预测精度都在80%以上，可用于短期预测。
'''

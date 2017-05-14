#coding:utf-8
import numpy as np
import time
import statsmodels.api as sm
from scipy import stats

'''
    linear regression with Stochastic Gradient Decent mothod
'''
def SGD(x, y, a, b):
# -------------------------------------------随机梯度下降算法----------------------------------------------------------

    # 两种终止条件
    loop_max = 10000   # 最大迭代次数(防止死循环)
    epsilon = 1e-6    

    alpha = 0.001       # 步长(注意取值过大会导致振荡,过小收敛速度变慢)
    diff = 0.           
    errorA = a
    errorB = b
    count = 0           # 循环次数
    finish = 0          # 终止标志
    m = len(x)          # 训练数据点数目

    while count < loop_max:
        #count += 1

        # 遍历训练数据集，不断更新权值
        for i in range(m):
            count += 1
            diff = a + b * x[i] - y[i]  # 训练集代入,计算误差值

            # 采用随机梯度下降算法,更新一次权值只使用一组训练数据
            a = a - alpha * diff
            b = b - alpha * diff * x[i]

            if ((a-errorA)*(a-errorA) + (b-errorB)*(b-errorB)) < epsilon:     
                # 终止条件：前后两次计算出的权向量的绝对误差充分小  
                finish = 1
                break
            else:
                errorA = a
                errorB = b
        if finish == 1:     # 跳出循环
            break

    #print 'loop count = %d' % count,  '\tweight:[%f, %f]' % (a, b)
    return a, b

def SGDWrong(x, target_data, a, b):
# -------------------------------------------随机梯度下降算法----------------------------------------------------------

    # 两种终止条件
    loop_max = 1000   # 最大迭代次数(防止死循环)
    epsilon = 1e-6    

    alpha = 0.001       # 步长(注意取值过大会导致振荡,过小收敛速度变慢)
    diff = 0.           
    errorA = a
    errorB = b
    count = 0           # 循环次数
    finish = 0          # 终止标志
    m = len(x) # 训练数据点数目

    while count < loop_max:
        count += 1

        # 遍历训练数据集，不断更新权值
        for i in range(m):  
            diff = a + b * x[i] - target_data[i]  # 训练集代入,计算误差值

            # 采用随机梯度下降算法,更新一次权值只使用一组训练数据
            a = a - alpha * diff
            b = b - alpha * diff * x[i]

            # ------------------------------终止条件判断-----------------------------------------
            # 若没终止，则继续读取样本进行处理，如果所有样本都读取完毕了,则循环重新从头开始读取样本进行处理。

        # ----------------------------------终止条件判断-----------------------------------------
        # 注意：有多种迭代终止条件，和判断语句的位置。终止判断可以放在权值向量更新一次后,也可以放在更新m次后。
        if ((a-errorA)*(a-errorA) + (b-errorB)*(b-errorB)) < epsilon:     
            # 终止条件：前后两次计算出的权向量的绝对误差充分小  
            finish = 1
            break
        else:
            errorA = a
            errorB = b
    print 'loop count = %d' % count,  '\tweight:[%f, %f]' % (a, b)
    return count, a, b

def SGD2(x, target_data, w):
# -------------------------------------------随机梯度下降算法----------------------------------------------------------

    # 两种终止条件
    loop_max = 1000   # 最大迭代次数(防止死循环)
    epsilon = 1e-3     

    alpha = 0.001       # 步长(注意取值过大会导致振荡,过小收敛速度变慢)
    diff = 0.           
    error = np.zeros(2)
    count = 0           # 循环次数
    finish = 0          # 终止标志
    m = len(x) # 训练数据点数目
    x0 = np.full(m, 1.0)                 
    input_data = np.vstack([x0, x]).T               # 将偏置b作为权向量的第一个分量

    while count < loop_max:
        count += 1

        # 遍历训练数据集，不断更新权值
        for i in range(m):  
            diff = np.dot(w, input_data[i]) - target_data[i]  # 训练集代入,计算误差值

            # 采用随机梯度下降算法,更新一次权值只使用一组训练数据
            w = w - alpha * diff * input_data[i]

            # ------------------------------终止条件判断-----------------------------------------
            # 若没终止，则继续读取样本进行处理，如果所有样本都读取完毕了,则循环重新从头开始读取样本进行处理。

        # ----------------------------------终止条件判断-----------------------------------------
        # 注意：有多种迭代终止条件，和判断语句的位置。终止判断可以放在权值向量更新一次后,也可以放在更新m次后。
        if np.linalg.norm(w - error) < epsilon:     # 终止条件：前后两次计算出的权向量的绝对误差充分小  
            finish = 1
            break
        else:
            error = w
    print 'loop count = %d' % count,  '\tw:[%f, %f]' % (w[0], w[1])
    return count, w

'''
def SGD(input_data, target_data):
# -----------------------------------------------梯度下降法-----------------------------------------------------------

    # 两种终止条件
    loop_max = 10000   # 最大迭代次数(防止死循环)
    epsilon = 1e-3     

    # 初始化权值
    np.random.seed(0)
    w = np.random.randn(2)
    #w = np.zeros(2)

    alpha = 0.001      # 步长(注意取值过大会导致振荡,过小收敛速度变慢)
    diff = 0.           
    error = np.zeros(2) 
    count = 0          # 循环次数
    finish = 0         # 终止标志
    time1 = time.time()

    while count < loop_max:
        count += 1

        # 标准梯度下降是在权值更新前对所有样例汇总误差，而随机梯度下降的权值是通过考查某个训练样例来更新的
        # 在标准梯度下降中，权值更新的每一步对多个样例求和，需要更多的计算
        sum_m = np.zeros(2)
        for i in range(m):
            dif = (np.dot(w, input_data[i]) - target_data[i]) * input_data[i]
            sum_m = sum_m + dif     # 当alpha取值过大时,sum_m会在迭代过程中会溢出

        w = w - alpha * sum_m       # 注意步长alpha的取值,过大会导致振荡
        #w = w - 0.005 * sum_m      # alpha取0.005时产生振荡,需要将alpha调小
        
        # 判断是否已收敛
        if np.linalg.norm(w - error) < epsilon:
            finish = 1
            break
        else:
            error = w
    print 'loop count = %d' % count,  '\tw:[%f, %f]' % (w[0], w[1])
'''

## Main functionality
def main():
    # 构造训练数据
    x = np.arange(0., 10., 0.2)
    m = len(x)                                      # 训练数据点数目
    print "m=%d" % m
    target_data = 2 * x + 5 + np.random.randn(m)
    N = 3000
    # 初始化权值
    #np.random.seed(0)
    #w = np.random.randn(2)
    #w = np.zeros(2)             # 7 times recurtion
    a = 3.273431                 # 104 times recurtion
    b = 2.948986
    #w = np.array([4.881609, 2.016084])    # 4 times recursion

    #SGD2(x, target_data, w)
    # loop from middle stat
    time1 = time.time()
    for i in range(0, N):
        SGD(x, target_data, a, b)
    time2 = time.time()
    print "optimised sgd loop time(s): ", time2-time1

    # full loop with sgd
    a2 = 0                      # 324 times recurtion
    b2 = 0
    time7 = time.time()
    for i in range(0, N):
        SGD(x, target_data, a2, b2)
    time8 = time.time()
    print "sgd loop time(s): ", time8-time7

    # check with scipy linear regression 
    time3 = time.time()
    for i in range(0, N):
        slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, target_data)
    time4 = time.time()
    print 'intercept = %s slope = %s' %(intercept, slope)
    print "linregress time(s): ", time4-time3

    time5 = time.time()
    X = sm.add_constant(x)
    for i in range(0, N):
        model = sm.OLS(endog=target_data, exog=X)
    result = model.fit()
    time6 = time.time()
    print "params", result.params
    print "OLS time(s): ", time6-time5

    '''
    plt.plot(x, target_data, 'k+')
    plt.plot(x, w[1] * x + w[0], 'r')
    plt.show()
    '''

if __name__ == "__main__":
    # Execute Main functionality
    main()
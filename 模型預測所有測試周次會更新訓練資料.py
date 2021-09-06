from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
from pygam import GAM
from sklearn.linear_model import LinearRegression
from pygam import LinearGAM


def data_matrix(data, data_range, window_size):
    expend_data=[[0,[],[],[],[]],[[],[],[],[],[]]]
    for i in range(0,len(data)):
        for j in range(0,len(data[i])):
            if(i==0 and j==0):
                continue
            for k in range(0,data_range-window_size+1):
                for l in range(0,len(data[i][j])):
                    expend_data[i][j].append(data[i][j].loc[l])
    df_data = [[0],[]]
    for i in range(0,len(expend_data)):
        for j in range(0,len(expend_data[i])):
            if(i==0 and j==0):
                continue
            df_data[i].append(pd.DataFrame(expend_data[i][j],columns = data[i][j].columns))
            df_data[i][j] = df_data[i][j].reset_index()
            df_data[i][j] = df_data[i][j].drop(columns=['index'])
    return df_data

def sliding_windows(df_fund, window_size, move_window, data_range, switch=0):

    df01 = df_fund.iloc[:,1]

    list1 =[]
    res = pd.DataFrame()
    i = 0
    j = 0
    x = window_size
    if(move_window == 1):
        y = data_range
    else:
        y = window_size
    z = y - x + 1 # y - x + 1=  z = 需要取樣次數
    p = df_fund.columns.size - y
    # df01 = df_fund.iloc[:,1]
    df02 = df_fund.iloc[:,p-switch:df_fund.columns.size - switch]
    # df03 = pd.concat([df01, df02],axis=1)
    # EX : 總共範圍取前10週 , 每次取 1~2 , 2~3 , 3~4 , 4~5 .....9~10 週 , 共取 z = 9 次
    #i = 基金編碼
    #j = 取樣 z 次
    for i in df01.index.unique():
        for j in np.arange(df02.columns.size):
            if j < z :
                list1.append([])
                list1[-1].append(df01[i])
                for k in range(0,x):
                    list1[-1].append(df02.iloc[i,j+k])
                #print(df02.iloc[i,j:j+x])
    res = res.append(list1)
    res = res.reset_index()
    res = res.drop(columns=['index'])
    new_columns = res.columns.values
    new_columns[0] = 'coid'
    res.columns = new_columns

    return res


data = [[],[]]

df_fund = pd.read_csv('allweek1.csv', encoding= "utf-8") #讀取"每週報酬率.csv"
df_s = pd.read_csv('result.csv', encoding= "utf-8") #讀取"每週報酬率.csv"
data[0].append(0)
data[0].append(pd.read_csv('market.csv' , encoding= "utf-8",index_col=0))
data[0].append(pd.read_csv('money.csv' , encoding= "utf-8",index_col=0))
data[0].append(pd.read_csv('target.csv' , encoding= "utf-8",index_col=0))
data[0].append(pd.read_csv('risk.csv' , encoding= "utf-8",index_col=0))
data[1].append(pd.read_csv('group.csv' , encoding= "utf-8",index_col=0))
data[1].append(pd.read_csv('group_market.csv' , encoding= "utf-8",index_col=0))
data[1].append(pd.read_csv('group_money.csv' , encoding= "utf-8",index_col=0))
data[1].append(pd.read_csv('group_target.csv' , encoding= "utf-8",index_col=0))
# data[1].append(pd.read_csv(r'GAM1205\group_risk.csv' , encoding= "utf-8",index_col=0))
data[1].append(pd.read_csv('group_target_market.csv' , encoding= "utf-8",index_col=0))


data_result = [[],[],[],[]]
data_future_result = [[],[]]
week_number = [4] #list # 12,16,20,24,28,32,36,40,44
# 8,12,16,20,24,28,32,36,40,44
for data_range in week_number:
    window_size = 4  #窗格大小
    data_range = data_range #取樣範圍週數

    if window_size>data_range:
        break
    else:

        move_window = 1 #是否移動窗格,1為是,0為否
        data_type = 4 #0=只計算分群(分群參數須為1),1=市場,2=幣別,3=標的別,4=市場*分群*標的別

        df_data = data_matrix(data, data_range, window_size)

        # switch=5
        # res = sliding_windows(df_fund, window_size, move_window, data_range, switch)
        # dummy = pd.concat([df_s.iloc[:,2],df_data[1][data_type]],axis = 1)
        # X_train = pd.merge(res.iloc[:,0:-1],dummy, on='coid')
        # X_train = X_train.drop(columns=['coid'])
        # Y_train = res.iloc[:,-1]

        def train_data_select(switch):
            res = sliding_windows(df_fund, window_size, move_window, data_range, switch)
            dummy = pd.concat([df_s.iloc[:,2],df_data[1][data_type]],axis = 1)
            X_train = pd.merge(res.iloc[:,0:-1],dummy, on='coid')
            X_train = X_train.drop(columns=['coid'])
            Y_train = res.iloc[:,-1]
            return X_train, Y_train

        X_train, Y_train = train_data_select(5)

        def test_data_select(switch):
            data_range = window_size
            df_data = data_matrix(data, data_range, window_size)
            res = sliding_windows(df_fund, window_size, move_window, data_range, switch)
            dummy=pd.concat([df_s.iloc[:,2],df_data[1][data_type]],axis = 1)
            X=pd.merge(res.iloc[:,0:-1],dummy, on='coid')
            X=X.drop(columns=['coid'])
            Y = res.iloc[:,-1]
            return X, Y
        X_test4, Y_test4 =  test_data_select(4)
        X_test3, Y_test3 =  test_data_select(3)
        X_test2, Y_test2 =  test_data_select(2)
        X_test1, Y_test1 =  test_data_select(1)
        X_test0, Y_test0 =  test_data_select(0)

        pd.concat([X_train,Y_train],axis=1).to_csv('for_excel.csv')

        caculate_MAE = lambda Y_train, result : round(sum(abs(result - Y_train.values))/(result.size),6)

        #################################################
        reg_linear = LinearRegression().fit(X_train, Y_train)  #  training linear model
        data_result[2].append(reg_linear.score(X_train, Y_train))  # caculate linear R square
        result = reg_linear.predict(X_train)
        data_result[0].append(caculate_MAE(Y_train, result)) # caculate linear MAE
        #################################################
        Linear_GAM = LinearGAM(n_splines=10).fit(X_train, Y_train)  #  training gam model
        data_result[3].append(Linear_GAM.statistics_['pseudo_r2']['explained_deviance']) #取出R平方
        result = Linear_GAM.predict(X_train)
        data_result[1].append(caculate_MAE(Y_train, result))
        #################################################
        #根據訓練好的模型，實際引入資料，接著預測未來資料，並算出MAE
        caculate_future_MAE = lambda model, X_test, Y_test : caculate_MAE(Y_test, model.predict(X_test))

        # def train_linear_model(X_test3, Y_test3):
        #     X_train, Y_train = train_data_select(4)
        #     reg_linear = LinearRegression().fit(X_train, Y_train)
        #     return caculate_future_MAE(reg_linear, X_test3, Y_test3)

        # 根據訓練好的模型，實際引入資料，接著預測未來資料，並算出MAE，存進 data_future_result
        data_future_result[0].append(caculate_future_MAE(reg_linear, X_test4, Y_test4))

        X_train, Y_train = train_data_select(4)
        reg_linear = LinearRegression().fit(X_train, Y_train)
        data_future_result[0].append(caculate_future_MAE(reg_linear, X_test3, Y_test3))

        X_train, Y_train = train_data_select(3)
        reg_linear = LinearRegression().fit(X_train, Y_train)
        data_future_result[0].append(caculate_future_MAE(reg_linear, X_test2, Y_test2))

        X_train, Y_train = train_data_select(2)
        reg_linear = LinearRegression().fit(X_train, Y_train)
        data_future_result[0].append(caculate_future_MAE(reg_linear, X_test1, Y_test1))

        X_train, Y_train = train_data_select(1)
        reg_linear = LinearRegression().fit(X_train, Y_train)
        data_future_result[0].append(caculate_future_MAE(reg_linear, X_test0, Y_test0))
        data_future_result[0].append(sum(data_future_result[0])/5)



        data_future_result[1].append(caculate_future_MAE(Linear_GAM, X_test4, Y_test4))

        X_train, Y_train = train_data_select(4)
        Linear_GAM = LinearGAM(n_splines=10).fit(X_train, Y_train)
        data_future_result[1].append(caculate_future_MAE(Linear_GAM, X_test3, Y_test3))

        X_train, Y_train = train_data_select(3)
        Linear_GAM = LinearGAM(n_splines=10).fit(X_train, Y_train)
        data_future_result[1].append(caculate_future_MAE(Linear_GAM, X_test2, Y_test2))

        X_train, Y_train = train_data_select(2)
        Linear_GAM = LinearGAM(n_splines=10).fit(X_train, Y_train)
        data_future_result[1].append(caculate_future_MAE(Linear_GAM, X_test1, Y_test1))

        X_train, Y_train = train_data_select(1)
        Linear_GAM = LinearGAM(n_splines=10).fit(X_train, Y_train)
        data_future_result[1].append(caculate_future_MAE(Linear_GAM, X_test0, Y_test0))
        data_future_result[1].append(sum(data_future_result[1])/5)
        print(data_range)
        data_future_result[0].append('linear error')
        data_future_result[1].append("GAM error")

# 下面4行code  是把實際預測未來資料算出來的MAE，存成CSV
df_future_result = {'linear MAE': data_future_result[0],
                    'GAM MAE ': data_future_result[1]}
df = pd.DataFrame(data=df_future_result)

# 第一欄是'linear '    第二欄是'GAM '
##如果有用移動窗格，這行會報錯  如果有用移動窗格，這行會報錯  如果有用移動窗格，這行會報錯  如果有用移動窗格，這行會報錯
df["index"] = ["test Error1","test Error2","test Error3","test Error4","test Error5","avg test error"," "] #如果有用移動窗格，這行會報錯
# 第一欄是'linear '    第二欄是'GAM '# 第一欄是'linear '    第二欄是'GAM '# 第一欄是'linear '    第二欄是'GAM '
df.to_csv('data_future_result.csv', mode='a', header=False, encoding= 'utf-8')
# 第一欄是'linear '    第二欄是'GAM '# 第一欄是'linear '    第二欄是'GAM '# 第一欄是'linear '    第二欄是'GAM '

pd.DataFrame(reg_linear.predict(X_train)).to_csv('data_virtual.csv', mode='a', header=False, encoding= 'utf-8')
df_result = {'linear MAE': data_result[0],
             'GAM MAE ': data_result[1],
             'linear R2': data_result[2],
             'GAM R2': data_result[3]}
df = pd.DataFrame(index = week_number, data=df_result)
df = df.T
df.to_csv('data_result.csv', mode='a', header=False, encoding= 'utf-8')

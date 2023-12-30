from GA_LSTM import GA
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
import matplotlib.pyplot as plt
import time

start_time=time.time()

def get_data():
    data = pd.read_csv(r'C:\Users\user\OneDrive\文件\Python Scripts\taiwan_stock_predict\臺灣加權指數歷史數據.csv')
    data=data["開市"]
    data = data.values
    data=data.reshape(-1, 1)
    sc = MinMaxScaler(feature_range=(0, 1)) # Feature Scaling
    training_set_scaled = sc.fit_transform(data)

    # 創建X_train和y_train
    X_train = []
    y_train = []
    for i in range(5, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-5:i, :])  # 使用五個column的資料
        y_train.append(training_set_scaled[i, :])      # 預測下一筆五個column的資料

    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshape 成3-dimension
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train

X_train, y_train = get_data()
fitness_plot = []
past_best_param = []
ga = GA(6, 0.8, 0.3, X_train, y_train, fitness_plot)

inital_param_list = []
inital_epoch_fitness = []
for i in range(10):  # 初始產生10組
    inital_param_list.append(ga.generate_gene())

temp_min_fitness = 1
for index, value in enumerate(inital_param_list):
    print(f'第{index}個')
    fitness, model = ga.model(value)
    min_fitness = min(ga.fitness_function(fitness))
    inital_epoch_fitness.append(min_fitness)
    if min_fitness < temp_min_fitness:
        temp_min_fitness = min_fitness
        model.save("ancestor_model.h5")
    else:
        print('我在釋放記憶體喔')
        K.clear_session()
temp_param, temp_fitness = ga.selection(inital_param_list, inital_epoch_fitness)
print(f'這次最佳六個fitness:{temp_fitness}')
past_best_param.append(temp_param[temp_fitness.index(min(temp_fitness))]) # 儲存尚未使用GA前的最佳參數

temp_min_fitness = 1
for iter in range(50):
    print(f'第{iter}代')
    cross_offspring = ga.crossover(temp_param)
    print(f'交配完:{cross_offspring}')
    mutation_offspring = ga.mutate(cross_offspring)
    print(f'突變完:{mutation_offspring}')
    child_fitness=[]

    for index, value in enumerate(mutation_offspring):
        print(f'第{index}個')
        fitness, model = ga.model(value)
        min_fitness = min(ga.fitness_function(fitness))
        child_fitness.append(min_fitness)
        if min_fitness < temp_min_fitness:
            temp_min_fitness = min_fitness
            model.save("child_model.h5")
            model.summary()
            print("我有蓋掉喔",min_fitness)


    elderly_child_param = temp_param + mutation_offspring
    elderly_child_fitness = temp_fitness + child_fitness
    temp_param, temp_fitness = ga.selection(elderly_child_param, elderly_child_fitness)
    print(f'這次最佳六個fitness:{temp_fitness}')
    past_best_param.append(temp_param[temp_fitness.index(min(temp_fitness))])

print(f'過去最佳參數:{past_best_param}')

end_time=time.time()
print("運算時間:",end_time-start_time)



plt.plot(fitness_plot)
plt.xticks(np.arange(0, len(fitness_plot), 1))
plt.show()




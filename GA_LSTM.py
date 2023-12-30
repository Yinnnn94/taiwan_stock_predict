import random as rand
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
import heapq

class GA:
    def __init__(self, best_num, cross_prob, mutation_prob, X_train, y_train, fitness_plot):
        self.best_num = best_num
        self.cross_prob = cross_prob
        self.mutation_prob = mutation_prob
        self.X_train = X_train
        self.y_train = y_train
        self.fitness_plot = fitness_plot # 一次迭代最好的fitness值，用來畫圖表的!!!

    def generate_gene(self):
        param = []
        for i in range(4):
            r = rand.randint(1, 100)
            param.append(r)
        param.append(round(rand.random(), 2))
        return param

    def fitness_function(self, history):
        return history.history['loss']
        
    def crossover(self, temp_gene_6):  # Uniform crossover，要直接放入6個基因
        parents_count = len(temp_gene_6)
        cross_offspring = []
        parents_seq = list(range(0, parents_count))
        sequence_list = rand.sample(parents_seq, parents_count)
        for i, j in zip(sequence_list[0::2], sequence_list[1::2]):
            print(f'Dad{temp_gene_6[i]}, Mom{temp_gene_6[j]}交配中...')
            offspring1 = []
            offspring2 = []
            for gene1, gene2 in zip(temp_gene_6[i], temp_gene_6[j]):
                if rand.random() < self.cross_prob:
                    offspring1.append(gene2)
                    offspring2.append(gene1)
                else:
                    offspring1.append(gene1)
                    offspring2.append(gene2)
            print(f'Child1{offspring1}, Child2{offspring2}')
            print('-' * 50)
            cross_offspring.append(offspring1)
            cross_offspring.append(offspring2)
        return cross_offspring

    def mutate(self, cross_offspring):  # 要直接放入6個基因
        mutation_offspring = []
        for gene in cross_offspring:
            offspring = []
            if rand.random() < self.mutation_prob:
                print('有mutation喔!')
                for i in range(len(gene)):
                    s = np.random.choice([-1, 1])
                    r = self.mutation_prob * np.random.uniform(10 ** (-6), 10 ** (-1))
                    a = 2 ** ((-np.random.rand()) * np.random.randint(4, 21))
                    units_ = rand.randint(-2, 2)
                    if i < 4:
                        offspring.append(int(gene[i] + units_))
                    else:
                        forget_rate = round(gene[i] + s * r * a, 2)
                        if forget_rate > 0.9:
                            forget_rate = 0.9
                        elif forget_rate < 0.1:
                            forget_rate = 0.1
                        offspring.append(forget_rate)
            else:
                print('pass')
                offspring = gene
            mutation_offspring.append(offspring)
        return mutation_offspring

    def model(self, param_list):
        regressor = Sequential()
        regressor.add(LSTM(units = param_list[0], return_sequences = True, input_shape = (self.X_train.shape[1], 1)))
        regressor.add(Dropout(param_list[-1]))
        regressor.add(LSTM(units = param_list[1], return_sequences = True))
        regressor.add(Dropout(param_list[-1]))
        regressor.add(LSTM(units = param_list[2], return_sequences = True))
        regressor.add(Dropout(param_list[-1]))
        regressor.add(LSTM(units = param_list[3]))
        regressor.add(Dropout(param_list[-1]))
        regressor.add(Dense(units = 1))  # 因為預測五個column的資料，所以輸出層units設為5
        # 編譯模型
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        # 進行訓練
        history = regressor.fit(self.X_train, self.y_train, epochs = 10, batch_size = 32, validation_split = 0.33)
        return history,regressor


    def selection(self, determine_param,  determine_fitness): # 直接放要判斷最佳的fitness值!!
        min_number_of_6 = heapq.nsmallest(6, determine_fitness)
        best_param_6 = []
        best_fitness_6 = []
        for t in min_number_of_6:
            best_param_6.append(determine_param[determine_fitness.index(t)])
            best_fitness_6.append(t)
        print(f'此次最佳的六個參數組合:{best_param_6}')
        self.fitness_plot.append(min(determine_fitness))

        return best_param_6, best_fitness_6
            


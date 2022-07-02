import numpy as np
import geatpy as ea
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_excel('data_generation.xlsx')
x=data['x']
y=data['y']
distance_matrix=np.zeros((len(x),len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        distance_matrix[i, j] = ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) ** 0.5

class MyProblem(ea.Problem):
    def __init__(self,distance_matrix):
        name='tsp_tests_with_geatpy'
        M=1
        maxormins=[1]
        self.city_num=len(distance_matrix)
        Dim=self.city_num
        varTypes=[1]*Dim
        lb=[0]*Dim
        ub=[Dim-1]*Dim
        lbin=[1]*Dim
        ubin=[1]*Dim
        self.distance_matrix=distance_matrix
        ea.Problem.__init__(self,name=name,M=M,maxormins=maxormins,Dim=Dim,varTypes=varTypes,lb=lb,ub=ub,lbin=lbin,ubin=ubin)

    def aimFunc(self,pop):
        x=pop.Phen.copy()
        distance_list=[]
        for i in range(pop.sizes):
            #print(pop.sizes)
            x_temp=x[i]
            #print(x_temp)
            sum_temp=0
            for j in range(self.city_num-1):
                start=int(x_temp[j])
                end=int(x_temp[j+1])
                sum_temp+=self.distance_matrix[start,end]
            sum_temp+=self.distance_matrix[x_temp[-1]][x_temp[0]]#最后加上从末尾到首端的距离
            distance_list.append(sum_temp)
        distance_list=np.array(distance_list).reshape(-1,1)
        #print(distance_list.shape)
        pop.ObjV=distance_list

problem=MyProblem(distance_matrix=distance_matrix)
algorithm=ea.soea_SEGA_templet(problem=problem,population=ea.Population(Encoding='P',NIND=40),MAXGEN=500,logTras=1)
res=ea.optimize(algorithm,verbose=False,drawing=1,outputMsg=True,drawLog=True)
print(res['ObjV'])

best_individual=res['Vars'][0]

plt.figure()
for i in range(len(best_individual)-1):
    idx1=best_individual[i]
    idx2=best_individual[i+1]
    plt.plot([x[idx1],x[idx2]],[y[idx1],y[idx2]])
    plt.scatter([x[idx1],x[idx2]],[y[idx1],y[idx2]])
plt.plot([x[best_individual[-1]],x[best_individual[0]]],[y[best_individual[-1]],y[best_individual[0]]])#将末端与首端连接
plt.scatter([x[best_individual[-1]],x[best_individual[0]]],[y[best_individual[-1]],y[best_individual[0]]])
plt.show()








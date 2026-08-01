import numpy as np
import logging
from attack_algorithm.attack_algorithm_base import AttackAlgorithmBase
from attack_problem.one_image_problem import SingleImageProblem
import random

class MLDE(AttackAlgorithmBase):

    def __init__(self, config):
        super().__init__(config)

        # 默认参数
        self.pop_size = config.get("pop_size", 20)
        self.F = config.get("F", 0.5)
        self.eps = config.get("eps", 0.05)
        # self.CR = config.get("CR", 0.9)

    # =====================================================
    #   evolve: MLDE 的 DE 主循环
    # =====================================================
    def evolve(self, problem: SingleImageProblem ):

        rnd = self.rnd
        return DE(self.pop_size, self.F,self.eps, problem, rnd)





def mating(pop,F, rnd):
    p2 = np.copy(pop)
    rnd.shuffle(p2)
    p3 = np.copy(p2)
    rnd.shuffle(p3)
    mutation = pop + F * (p2 - p3)
    return mutation

def select(pop,fitness,fit,off,off_fitness,off_fit):
   new_pop = pop.copy()
   new_fitness = fitness.copy()
   new_fit = fit.copy()
   i=np.argwhere(fitness>off_fitness)
   new_pop[i] = off[i].copy()
   new_fitness[i] = off_fitness[i].copy()
   new_fit[i] = off_fit[i].copy()
   return new_pop ,new_fitness ,new_fit

def complement (fit,pop, fitness,problem) :
    popnew = pop.copy()
    sort = np.argsort(fitness.reshape(-1))
    for q in range (len(pop)):
        i = sort[q]
        fit_item = fit.copy()
        c = np.argwhere(fit[i] == 0)
        fit_item[:, c] = 0
        fitness_tem = np.sum(fit_item, axis=1)
        j = np.argmin(fitness_tem)
        popnew[i] = pop[i] + pop[j]*0.5
    off_fitness_new, off_fit_new = problem.evaluate(popnew)
    if off_fit_new is None:
        return pop, fitness, fit, False
    pop1, fitness1, fit1 = select(pop, fitness, fit, popnew, off_fitness_new, off_fit_new)
    return pop1,fitness1, fit1, True


def DE(pop_size, F, eps, problem: SingleImageProblem , rnd: random):
    generation_save = np.zeros((10000,))

    # ========== 使用 problem 的 x_range 初始化种群 ==========
    dim = problem.get_dimension()
    x_range = problem.get_x_range()  # list of tuples [(min,max),...]

    pop = np.zeros((pop_size, dim))
    for i in range(dim):
        low, high = x_range[i]
        pop[:, i] = np.array([rnd.uniform(low, high)*eps for _ in range(pop_size)])

    norm_r = problem.l2_norm(pop)



    eval_count = 0
    fitness, fit = problem.evaluate(pop)
    eval_count += pop_size
    count = 0
    fitmin = np.min(fitness)
    generation_save[count] = fitmin
    
    if (len(np.where(fitness == 0)[0]) == 0):
        while (problem.evaluations < problem.max_evaluation):
            count += 1
            off = mating(pop,F,rnd)
            off_fitness , off_fit = problem.evaluate(off)
            if(off_fitness is None):
                break
            pop ,fitness ,fit = select (pop,fitness,fit,off,off_fitness,off_fit)
            pop, fitness, fit, TerminateFlag = complement(fit, pop, fitness, problem)
            if( not TerminateFlag):
                break   
            fitmin = np.min(fitness)
            pop_r= problem.l2_norm(pop) 
            r_min= np.min(pop_r)    
            print(f"Evaluation:{problem.evaluations}, Best fitness:{fitmin}, Best radius:{r_min}")
            generation_save[count] = fitmin
            if (len(np.where(fitness == 0)[0]) != 0):
                break
    if (len(np.where(fitness == 0)[0]) != 0):
        return pop[np.where(fitness == 0)[0][0]]
    else:
        return pop[0]
'''
Copyright (C) 2018 Xuan Liu and Haowei Zhao

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''





import numpy as np
from copy import deepcopy as dcp
from collections import deque
import math
# import csv
# import pickle

class Geometry:
    def Angles(self, s1, s2):
        l = s2[1] - s1[1]
        w = s2[0] - s1[0]
        return np.arctan2(l, w)

class BayesianLearnerHS:

    # TODO: the convergence of the computed belief is highlt related to the design of alpha(reward rescaling param), gamma(forgeting rate) and dropout(memory window)
    def __init__(self, goal_set, initial_state, alpha = .05, gamma = 0.85, dropout = 10):
        '''
        :param goal_set:
        :param initial_state:
        '''

        self.gSolver = Geometry()

        self.potential_goals = goal_set
        self.s_0 = initial_state

        self.angles = dict.fromkeys(goal_set.keys(), None)
        for g in self.angles:
            self.angles[g] = self.gSolver.Angles(self.s_0[0], self.potential_goals[g])

        self.dropout = dropout
        self.alpha = alpha
        self.gamma = gamma

        # averaged belief over all samples
        # update rule P(g|h) = alpha*P(g|h) + (1-alpha)*P(g|new) , h <- h , new

        self.init_belief = {}
        for g in self.potential_goals:
            self.init_belief[g] = 1.0/len(goal_set)

        # for decoy in self.Pi_g.keys():
        #     self.Pi_g[decoy] = self.get_Policy(self.decoys[decoy])

        # initialize P(|traj(t))
        # update rule P(g|traj(t+1)) propto P(g|traj(t))*P(s(t+1)|g, traj(t)), traj(t+1) <- traj(t).append(s(t+1))
        self.P_h_g = dcp(self.init_belief)
        self.traj_h = deque()
        self.traj_h.append(initial_state)

        self.traj_d = 0.001

    def P_s_g(self, traj, goal, g_theta):

        smoothing = 1/10**323

        g = np.array(goal)

        s0 = np.array(traj[0])
        st = np.array(traj[-1])

        d = np.linalg.norm(g - s0[0])
        d_ = np.linalg.norm(g - st[0])

        t = len(traj)

        try:
            T_ = t * int(d_ / self.traj_d)
        except OverflowError:
            # T_ = t * int(d_ / self.traj_d)
            print (self.traj_d, d, d_)

        T = t + T_


        C_p0_pt = 0
        C_pt_pT_opt = 0
        C_p0_pT_opt = 0

        for i in range(0, t):
            a = np.array(traj[i][0])
            theta = np.array(traj[i][1])

            C_p0_pt += self.gamma**(t-i)*(np.linalg.norm(g - a)) #+ 10*np.abs(g_theta - theta))

        C_p0_pt *= self.alpha

        # assume no theta different for optimal situation
        for i in range(0, T):
            C_p0_pT_opt += self.gamma**(T-i)*(d*(T-i)/T + 0.0)
        C_p0_pT_opt *= self.alpha

        for i in range(t+1, T):
            C_pt_pT_opt += self.gamma**(T-i)*(d_*(T-i)/T_ + 0.0)
        C_pt_pT_opt *= self.alpha

        result = (np.exp(-C_p0_pt)*np.exp(-C_pt_pT_opt))/(np.exp(-C_p0_pT_opt))

        # if math.isnan(result):
        #     print(C_p0_pt, C_p0_pT_opt, C_pt_pT_opt)

        return result

    def Update(self, s_t):
        # update state when new input is taken
        self.traj_d += np.linalg.norm(np.array(s_t[0]) - np.array(self.traj_h[-1][0]))
        if len(self.traj_h) >= self.dropout and np.linalg.norm(np.array(s_t[0]) - np.array(self.traj_h[-1][0])) > 0.0002:
            self.traj_d -= np.linalg.norm(np.array(self.traj_h[1][0]) - np.array(self.traj_h[0][0]))
            self.traj_h.popleft()

        self.traj_h.append(s_t)

        # update state angles
        for g in self.angles:
            self.angles[g] = self.gSolver.Angles(s_t[0], self.potential_goals[g])


    def Bayesian_inference(self, s_t):
        # update rule P(g|traj(t+1)) propto P(g|traj(t))*P(s(t+1)|g, traj(t)), traj(t+1) <- traj(t).append(s(t+1))

        # for t in range(0, len(traj)):
        # TODO: add orientation here!!!

        self.Update(s_t)
        # print(np.linalg.norm(np.array(self.traj_h[-1][0]) - np.array(self.traj_h[-2][0])) \
        #       + np.abs(self.traj_h[-1][1] - self.traj_h[-2][1]))

        if (np.linalg.norm(np.array(self.traj_h[-1][0]) - np.array(self.traj_h[-2][0])) \
                + np.abs(self.traj_h[-1][1] - self.traj_h[-2][1])) > 0.0002:
        # if 1:

            # print(np.linalg.norm(np.array(self.traj_h[-1][0]) - np.array(self.traj_h[-2][0])) \
            #       + np.abs(self.traj_h[-1][1] - self.traj_h[-2][1]))


            for goal in self.potential_goals:
                self.P_h_g[goal] = self.P_h_g[goal] * self.P_s_g(self.traj_h, self.potential_goals[goal], self.angles[goal])
                # if goal == 'g2':
                #     print(self.P_s_g(self.traj_h, self.potential_goals[goal], self.angles[goal]))

            # normalization
            temp_sum = sum(self.P_h_g.values())

            for goal in self.potential_goals:
                self.P_h_g[goal] /= temp_sum

                if self.P_h_g[goal] > 0.99:
                    self.P_h_g[goal] = 0.99
                    for goal_ in self.potential_goals:
                        if not goal_ == goal:
                            self.P_h_g[goal_] = (1-0.99)/(len(self.P_h_g) - 1)
                    break

            print (self.P_h_g)
        return dcp(self.P_h_g)

    # def get_belief(self):
    #     print(self.P_h_g)
    #

    # def traj_Learner(self, s):


if __name__ == '__main__':
    goal_set = {'g1':(-2, 2), 'g2':(0, 4), 'g3':(2, 2)}
    init_s = ((0, 0), 1.57)
    learner = BayesianLearnerHS(goal_set, init_s)
    traj = [((0, 0.11), 1.57),
            ((0, 0.11), 1.57),
            ((0, 0.11), 1.57),
            ((0, 0.11), 1.57),
            ((0, 0.11), 1.57),
            ((0, 0.11), 1.57),
            ((0, 0.11), 1.67),
            ((0, 0.11), 1.67),
            ((0, 0.11), 1.67),
            ((0, 0.11), 1.67),
            ((0, 0.27), 1.57),
            ((0, 0.32), 1.57),
            ((0, 0.45), 1.57),
            ((0, 0.59), 1.57),
            ((0, 0.59), 1.57),
            ((0, 0.59), 1.57),
            ((0, 1.59), 1.57),
            ((0, 1.59), 1.57),
            ((0, 1.59), 1.57),
            ((0, 1.62), 1.57),
            ((0, 1.78), 1.57),
            ((0.1, 1.78), 1.57),
            ((0.2, 1.78), 1.57),
            ((1.3, 1.78), 1.57),
            ((1.4, 1.78), 1.57),
            ((1.5, 1.78), 1.57),
            ((1.6, 1.78), 1.57),
            ((1.9, 1.78), 1.57),
            ((2.5, 1.78), 1.57)]

    print('state', init_s, 'belief:', learner.P_h_g)
    for s in traj:

        # TODO: deal with input (s, theta), they shouldn't be independent
        learner.Bayesian_inference(s)

        print('state', s, 'belief:', learner.P_h_g)

        # print(learner.traj_h, learner.traj_d)
        # print('angles:',learner.angles)
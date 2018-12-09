'''
Copyright (C) 2014 Terry Stewart and Travis DeWolf
Modified by Haowei Zhao and Xuan Liu

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
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Qt4Agg')


from matplotlib import pyplot as plt
from matplotlib import animation
import csv
import math
from BayesianLearnerHindSight import BayesianLearnerHS
from copy import deepcopy as dcp

class Runner:
    """
    A class for drawing the arm simulation.

    NOTE: If you're getting an error along the lines of 
    'xrange is not an iterator', make sure that you have 
    the most recent version of matplotlib, from their github.
    """
    def __init__(self, title='', dt=1e-3, control_steps=1, 
                       display_steps=1, t_target=1.0, 
                       control_type='', trajectory=None,
                       infinite_trail=False, mouse_control=False):
        self.dt = dt
        self.control_steps = control_steps
        self.display_steps = display_steps
        self.target_steps = int(t_target/float(dt*display_steps))
        self.trajectory = trajectory

        self.control_type = control_type 
        self.infinite_trail = infinite_trail
        self.mouse_control = mouse_control
        self.title = title
        self.threshold = 0.7
        self.alpha = 0.9
        self.takeoverflag = True

        self.sim_step = 0
        self.trail_index = 0
        self.trail_flag = True
        self.EE_trajectory = []

        # Set up start parameters and desired 3 targets

        self.start_x = 0
        self.start_y = 1.0
        self.target_desire_x_1 = -1.5
        self.target_desire_y_1 = 3.0
        self.target_desire_x_2 = 0
        self.target_desire_y_2 = 3.5
        self.target_desire_x_3 = 1.5
        self.target_desire_y_3 = 3.0
        self.start_q = -1.57
        self.goal_set = {'g1': (-1.5, 3.0), 'g2': (0, 3.5), 'g3': (1.5, 3.0)}
        self.goal_belief = {'g1': 1/3, 'g2': 1/3, 'g3': 1/3}
        self.init_s = ((self.start_x, self.start_y), self.start_q)

        self.bayes_solver = BayesianLearnerHS(self.goal_set, self.init_s)
        self.angles = dcp(self.bayes_solver.angles)

        self.tau = None


        
    def run(self, arm, control_shell, end_time=None):

        self.end_time = end_time

        self.arm = arm
        if arm.DOF == 1:
            box = [-1, 1, -.25, 1.5]
        elif arm.DOF == 2:
            box = [-.5, .5, -.25, .75]
        elif arm.DOF == 3:
            box = [-2, 2, -.5, 4]

        self.shell = control_shell
        
        fig = plt.figure(figsize=(8.1, 8.1), dpi=None)
        fig.suptitle(self.title);
        # set the padding of the subplot explicitly
        fig.subplotpars.left = .1; fig.subplotpars.right = .9
        fig.subplotpars.bottom = .1; fig.subplotpars.top = .9

        ax = fig.add_subplot(1, 1, 1, 
                             xlim=(box[0], box[1]), 
                             ylim=(box[2], box[3]))
        ax.xaxis.grid(); ax.yaxis.grid()
        # make it a square plot
        ax.set_aspect(1) 

        # set up plot elements
        self.trail, = ax.plot([], [], color='magenta', lw=3)
        self.arm_line, = ax.plot([], [], 'o-', mew=4, color='b', lw=5)
        #self.orien_line, = ax.plot([], [], 'o-', mew=0.1, color='r', lw=0.5)
        self.target_line, = ax.plot([], [], 'r-x', mew=4)
        self.target_desire_1, = ax.plot(self.target_desire_x_1,self.target_desire_y_1,'g-x',mew=4)
        self.target_desire_2, = ax.plot(self.target_desire_x_2,self.target_desire_y_2,'y-x',mew=4)
        self.target_desire_3, = ax.plot(self.target_desire_x_3,self.target_desire_y_3,'b-x',mew=4)
        self.info = ax.text(box[0]+abs(.1*box[0]), \
                            box[3]-abs(.1*box[3]), \
                            '', va='top')
        self.trail_data = np.ones((self.target_steps, 2), \
                                   dtype='float') * np.NAN
    
        if self.trajectory is not None:
            ax.plot(self.trajectory[:, 0], self.trajectory[:, 1], alpha=.3)

        # connect up mouse event if specified
        if self.mouse_control: 
            self.target = self.shell.controller.gen_target(arm)
            self.target = np.array([self.start_x, self.start_y])
            self.shell.controller.target = self.target
            # get pixel width of fig (-.2 for the padding)
            self.fig_width = (fig.get_figwidth() - .2 \
                                * fig.get_figwidth()) * fig.get_dpi()

            # Keyboard logic--translation, you can change the translation speed here
            def on_key_trans(event):
                if event.key is not None:
                    if event.key == "up":
                        self.start_y += 0.1
                    elif event.key == "down":
                        self.start_y -= 0.1
                    elif event.key == "left":
                        self.start_x -= 0.1
                    elif event.key == "right":
                        self.start_x += 0.1
                self.target = np.array([self.start_x, self.start_y])
                self.shell.controller.target = self.target
            # Keyboard logic--orientation, you can change the augluar speed here
            def on_key_orien(event):
                if event.key is not None:
                    if event.key == "=":
                        self.start_q += 0.1
                    if event.key == "-":
                        self.start_q -= 0.1

                    if abs(self.start_q) > 2*math.pi and self.start_q < 0:
                        self.start_q = -self.start_q % (2 * math.pi)
                    elif abs(self.start_q) > 2*math.pi and self.start_q > 0:
                        self.start_q = self.start_q % (2 * math.pi)
                    else:
                        pass

            # hook up function to keyboard event
            fig.canvas.mpl_connect('key_press_event', on_key_trans)
            fig.canvas.mpl_connect('key_press_event', on_key_orien)

        frames = 5000

        anim = animation.FuncAnimation(fig, self.anim_animate, 
                   init_func=self.anim_init, frames=frames, interval=50, blit=True)
        
        self.anim = anim


        # self.anim.save('result.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

        # print('task finished.')
        # print('save traj.')


    # TODO: add a parallel function to process the Bayesian Inference and Update the Belief

    def make_info_text(self):# Print probability of targets here
        text = []
        # text.append('goal belief: g1 = {g1:.2f}, g2 = {g2:.2f}, g3 = {g3:.2f}'.format(**self.goal_belief))
        # text.append('goal angle: g1 = {g1:.2f}, g2 = {g2:.2f}, g3 = {g3:.2f}'.format(**self.angles))
        # q_text = ' '.join('%4.2f,'%F for F in self.arm.q)
        # text.append('p2_yellow = ['+q_text+']')
        # u_text = ' '.join('%4.2f,'%F for F in self.shell.u)
        # text.append('p3_blue = ['+u_text+']')
        # theta_text = ' '.join('%1.2g'%self.start_q)
        # text.append('theta_in_r = '+theta_text)
        # text.append('t = ' % (self.sim_step * self.dt))
        return '\n'.join(text)

    def anim_init(self):
        self.info.set_text('')
        self.arm_line.set_data([], [])
        self.target_line.set_data([], [])
        self.trail.set_data([], [])
        #self.orien_line.set_data([],[])
        return self.arm_line, self.target_line, self.info, self.trail

    # simulator in real time
    def anim_animate(self, i):

        if self.end_time is not None:
            # check for run time
            if (self.sim_step * self.dt) > self.end_time:
                self.anim.stop()

                plt.close()

        if self.control_type == 'random':
            # update target after specified period of time passes
            if self.sim_step % (self.target_steps*self.display_steps) == 0:
                self.target = self.shell.controller.gen_target(self.arm)
        else:
            self.target = self.shell.controller.target
       
        # before drawing
        for j in range(self.display_steps):            
            # update control signal
            if (self.sim_step % self.control_steps) == 0 or self.tau is None:
                    self.tau = self.shell.control(self.arm)
            # apply control signal and simulate
            self.arm.apply_torque(u=self.tau, dt=self.dt)

            self.sim_step += 1

        self.goal_belief = self.bayes_solver.init_belief

        # update figure
        #self.arm_orien = np.array([[self.arm.position()[0,3],self.arm.position()[0,3]+0.5*np.cos(self.start_q)],
                                   #[self.arm.position()[1,3],self.arm.position()[1,3]+0.5*np.sin(self.start_q)]])
        #self.orien_line.set_data(self.arm_orien)
        self.arm_line.set_data(*self.arm.position())

        # update belief
        s_t = ((self.arm.position()[0, 3], self.arm.position()[1, 3]), self.start_q)
        # s_t = ((self.start_x, self.start_y), self.start_q)
        self.goal_belief = self.bayes_solver.Bayesian_inference(s_t)

        # take over process
        # if self.takeoverflag:
        #     if self.goal_belief['g1'] > self.threshold:
        #         self.target = self.alpha*np.array(self.goal_set['g1']) + (1-self.alpha)*self.target
        #     elif self.goal_belief['g2'] > self.threshold:
        #         self.target = self.alpha*np.array(self.goal_set['g2']) + (1-self.alpha)*self.target
        #     elif self.goal_belief['g3'] > self.threshold:
        #         self.target = self.alpha*np.array(self.goal_set['g3']) + (1-self.alpha)*self.target
        #     self.alpha = 1 - (1-self.alpha)*0.5
        # self.angles = dcp(self.bayes_solver.angles)

        # # Record end effector trajectory
        self.EE_trajectory.append([self.arm.position()[0, 3],
                                   self.arm.position()[1, 3],
                                   self.goal_belief['g1'],
                                   self.goal_belief['g2'],
                                   self.goal_belief['g3']])

        csvfile = 'trajectory_new.csv'
        with open(csvfile, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(self.EE_trajectory)

        # csvfile = 'trajectory_belief.csv'
        # with open(csvfile, "w") as output:
        #     writer = csv.writer(output, lineterminator='\n')
        #     writer.writerows(self.goal_belief['g1'])


        #self.info.set_text(self.make_info_text())
        self.trail.set_data(self.trail_data[:, 0], self.trail_data[:, 1])
        if self.target is not None:
            target = self.target
            self.target_line.set_data(target)

        # update hand trail
        if self.trail_flag:
            if self.infinite_trail:
                # if we're writing, keep all pen_down history
                self.trail_index += 1

                # if we've hit the end of the trail, double it and copy
                if self.trail_index >= self.trail_data.shape[0] - 1:
                    trail_data = np.zeros((self.trail_data.shape[0] * 2,
                                           self.trail_data.shape[1])) * np.nan
                    trail_data[:self.trail_index + 1] = self.trail_data
                    self.trail_data = trail_data

                self.trail_data[self.trail_index] = \
                    self.arm_line.get_xydata()[-1]
            else:
                # else just use a buffer window
                self.trail_data[:-1] = self.trail_data[1:]
                self.trail_data[-1] = self.arm_line.get_xydata()[-1]

        # if self.shell.pen_down:
        #     if self.infinite_trail:
        #         # if we're writing, keep all pen_down history
        #         self.trail_index += 1
        #
        #         # if we've hit the end of the trail, double it and copy
        #         if self.trail_index >= self.trail_data.shape[0]-1:
        #             trail_data = np.zeros((self.trail_data.shape[0]*2,
        #                                    self.trail_data.shape[1]))*np.nan
        #             trail_data[:self.trail_index+1] = self.trail_data
        #             self.trail_data = trail_data
        #
        #         self.trail_data[self.trail_index] = \
        #                                 self.arm_line.get_xydata()[-1]
        #     else:
        #         # else just use a buffer window
        #         self.trail_data[:-1] = self.trail_data[1:]
        #         self.trail_data[-1] = self.arm_line.get_xydata()[-1]
        # else:
        #     # if pen up add a break in the trail
        #     self.trail_data[self.trail_index] = [np.nan, np.nan]

        return self.target_line, self.info, self.trail, self.arm_line #,  self.orien_line

    def save(self):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        self.anim.save('im.mp4', writer=writer)

    def show(self):
        try:
            # plt.plot(self.)
            # self.save()
            plt.show()

            # self.save()
        except AttributeError:
            pass

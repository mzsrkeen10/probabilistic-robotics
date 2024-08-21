#!/usr/bin/env python
# coding: utf-8

# In[5]:


import copy
import random
import sys

from scipy.stats import multivariate_normal
import numpy as np
sys.path.append("../scripts/")
from robot import *


# In[2]:


class Particle:
    def __init__(self, init_pose, weight):
        self.pose = init_pose
        self.weight = weight

    def motion_update(self, nu, omega, time, noise_rate_pdf):
        ns = noise_rate_pdf.rvs()
        noised_nu = nu + ns[0] * math.sqrt(abs(nu)/time) + ns[1] * math.sqrt(abs(omega)/time)
        noised_omega = omega + ns[2] * math.sqrt(abs(nu)/time) + ns[3] * math.sqrt(abs(omega)/time)
        self.pose = IdealRobot.state_transition(noised_nu, noised_omega, time, self.pose)

    def observation_update(self, observation, envmap, distance_dev_rate, direction_dev):
        for d in observation:
            obs_pos = d[0]
            obs_id = d[1]

            pos_on_map = envmap.landmarks[obs_id].pos
            particle_suggest_pos = IdealCamera.observation_function(self.pose, pos_on_map)

            distance_dev = distance_dev_rate * particle_suggest_pos[0]
            cov = np.diag(np.array([distance_dev**2, direction_dev**2]))
            self.weight *= multivariate_normal(mean=particle_suggest_pos, cov=cov).pdf(obs_pos)


# In[13]:


class MCL:
    def __init__(
        self,
        envmap,
        init_pose,
        num,
        motion_noise_stds={"nn": 0.19, "no": 0.001, "on": 0.13, "oo": 0.2},
        distance_dev_rate=0.14,
        direction_dev=0.05,
    ):
        self.particles = [Particle(init_pose, 1.0/num) for _ in range(num)]
        self.map = envmap
        self.distance_dev_rate = distance_dev_rate
        self.direction_dev = direction_dev

        v = motion_noise_stds
        c = np.diag([v["nn"]**2, v["no"]**2, v["on"]**2, v["oo"]**2])
        self.motion_noise_rate_pdf = multivariate_normal(cov=c)

        self.ml = self.particles[0]
        self.pose = self.ml.pose

    def set_ml(self):
        i = np.argmax([p.weight for p in self.particles])
        self.ml = self.particles[i]
        self.pose = self.ml.pose

    def motion_update(self, nu, omega, time):
        for p in self.particles:
            p.motion_update(nu, omega, time, self.motion_noise_rate_pdf)

    def observation_update(self, observation):
        for p in self.particles:
            p.observation_update(observation, self.map, self.distance_dev_rate, self.direction_dev)
        self.set_ml()
        self.resampling()

    def resampling(self):
        # ws = [e.weight for e in self.particles]
        # if sum(ws) < 1e-100:
        #     ws = [e + 1e-100 for e in ws]
        # ps = random.choices(self.particles, weights=ws, k=len(self.particles))
        ws = np.cumsum([e.weight for e in self.particles])
        if ws[-1] < 1e-100:
            ws += 1e-100

        step = ws[-1] / len(self.particles)
        r = np.random.uniform(0.0, step)
        cur_pos = 0
        ps = []

        while len(ps) < len(self.particles):
            if r < ws[cur_pos]:
                ps.append(self.particles[cur_pos])
                r += step
            else:
                cur_pos += 1
        
        self.particles = [copy.deepcopy(e) for e in ps]
        for p in self.particles:
            p.weight = 1.0 / len(self.particles)

    def draw(self, ax, elems):
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]
        vxs = [math.cos(p.pose[2]) * p.weight * len(self.particles) for p in self.particles]
        vys = [math.sin(p.pose[2]) * p.weight * len(self.particles) for p in self.particles]
        elems.append(ax.quiver(xs, ys, vxs, vys, angles="xy", scale_units="xy", scale=1.5, color="blue", alpha=0.5))


# In[16]:


class EstimationAgent(Agent):
    def __init__(self, nu, omega, estimator, time_interval):
        super().__init__(nu, omega)
        self.estimator = estimator
        self.time_interval = time_interval

        self.prev_nu = 0.0
        self.prev_omega = 0.0

    def decision(self, observation=None):
        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        self.prev_nu, self.prev_omega = self.nu, self.omega
        self.estimator.observation_update(observation)
        return self.nu, self.omega

    def draw(self, ax, elems):
        self.estimator.draw(ax, elems)
        x, y, t = self.estimator.pose
        s = f"({x:.2f}, {y:.2f}, {int(t*180/math.pi)%360})"
        elems.append(ax.text(x, y+0.1, s, fontsize=8))


# In[5]:


# world = World(30, 0.1, debug=False)

# m = Map()
# for ln in [(-4, 2), (2, -3), (3, 3)]:
#     m.append_landmark(Landmark(*ln))
# world.append(m)

# initial_pose = np.array([2, 2, math.pi/6]).T
# estimator = MCL(initial_pose, 100)
# circling = EstimationAgent(0.2, 10.0/180*math.pi, estimator)
# r = Robot(initial_pose, sensor=Camera(m), agent=circling)
# world.append(r)

# world.draw()


# In[ ]:


# initial_pose = np.array([0, 0, 0]).T
# estimator = MCL(initial_pose, 100, motion_noise_stds={"nn": 0.01, "no": 0.02, "on": 0.03, "oo": 0.04})
# a = EstimationAgent(0.2, 10.0/180*math.pi, estimator, 0.1)
# estimator.motion_update(0.2, 10.0/180*math.pi, 0.1)
# for p in estimator.particles:
#     print(p.pose)


# In[9]:


def trial():
    time_interval = 0.1
    world = World(30, time_interval, debug=False)

    m = Map()
    for ln in [(-4, 2), (2, -3), (3, 3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)
    
    initial_pose = np.array([0, 0, 0]).T
    estimator = MCL(m, initial_pose, 100) #, motion_noise_stds)
    circling = EstimationAgent(0.2, 10.0/180*math.pi, estimator, time_interval)
    r = Robot(initial_pose, sensor=Camera(m), agent=circling, color="red")
    world.append(r)

    world.draw()


# In[19]:


# trial({"nn": 0.19, "no": 0.001, "on": 0.13, "oo": 0.2})
# trial()


# In[ ]:





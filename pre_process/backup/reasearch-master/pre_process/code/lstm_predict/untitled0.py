#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 18:56:40 2018

@author: vito
"""

from matplotlib import pyplot as plt
import numpy as np

num_episodes = 500

class EpisodeHistory:
    def __init__(self,
                 capacity,
                 title = "...",
                 ylabel = "...",
                 xlabel="...",
                 verbose = True,
                 plot_episode_count=num_episodes,
                 max_value_per_episode=1,
                 num_plot=1,
                 label=["0","1","2","3"]):

        self.datas = {}
        for i in range(num_plot):
            self.datas[i] = np.zeros(capacity, dtype=float)
        
        self.plot_episode_count = plot_episode_count
        self.max_value_per_episode = max_value_per_episode
        self.label = label
        
        self.num_plot = num_plot 
        self.point_plot = {}
        self.fig = None
        self.ax = None
        self.verbose = verbose
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        
        self.fig, self.ax = plt.subplots(figsize=(14, 7), facecolor='w', edgecolor='k')
        self.fig.canvas.set_window_title(self.title)
        
        self.ax.set_xlim(0, self.plot_episode_count + 5)
        self.ax.set_ylim(0, self.max_value_per_episode)
        self.ax.yaxis.grid(True)
        
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        
        color_set = ['b', 'g', 'r', 'c']
        for i in range(num_plot):
            self.point_plot[i] = plt.plot([], [], linewidth=2.0, c=color_set[i], label=self.label[i])

    def __getitem__(self, index):
        return self.datas[index]

    def __setitem__(self, index, value):
        self.datas[index] = value

    def update_plot(self, episode_index):
        plot_right_edge = episode_index
        plot_left_edge = max(0, plot_right_edge - self.plot_episode_count)

        # Update point plot.
        for i in range(self.num_plot):
            x = range(plot_left_edge, plot_right_edge)
            y = self.datas[i][plot_left_edge:plot_right_edge]
            self.point_plot[i][0].set_xdata(x)
            self.point_plot[i][0].set_ydata(y)
            self.ax.set_xlim(plot_left_edge, plot_left_edge + self.plot_episode_count)

        # Repaint the surface.
        plt.legend()
        plt.draw()
        plt.pause(0.0001)
        
a = EpisodeHistory(500,num_plot=4)
for i in range(4):
    a[i][:] = np.random.rand(num_episodes)
a.update_plot(500)




































#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 15:07:59 2016

@author: bernhardlangwallner
"""

        # Initialize Q-learning parameters
        self.Q_matrix = dict()
        self.epsilon_start = 0.98
        self.epsilon = self.epsilon_start
        self.eps_decay = 1 - 1e-2
        self.alpha = 0.2
        self.gamma_start = 0.3
        self.gamma = self.gamma_start
        self.gamma_decay = 1 - 7e-3
        # Initialize counters to track negative rewards and overall steps
        self.ctr_neg_rewards = 0
        self.ctr_steps = 0
        # Initialize overall sum of deadlines to track speed of arrival at destination
        self.sum_deadlines = 0
        # Counter to track success rate as learning occurs
        self.success_count = 0
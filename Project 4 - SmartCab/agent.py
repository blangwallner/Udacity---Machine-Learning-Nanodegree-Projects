import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # Initialize reduced set of actions for random step choice (exclude "None" to help explore)
        self.valid_real_actions = ['forward', 'left', 'right']

        # Initialize Q-learning parameters epsilon, alpha and gamma
        self.Q_matrix = dict()
        self.epsilon_start = 1
        self.eps_decay = .99
        self.epsilon = self.epsilon_start
        self.alpha_start = 0.2
        self.alpha_decay = 0.999
        self.alpha = self.alpha_start
        self.gamma = 0.1
        # Initialize counters to track negative rewards and overall steps
        self.ctr_neg_rewards = 0
        self.ctr_steps = 0
        # Initialize overall sum of deadlines to track speed of arrival at destination
        self.sum_deadlines = 0
        # Counter to track success rate as learning occurs
        self.success_count = 0
        # Track KPIs for every trial
        self.k_trial = -1 # running number of present trial
        success_log = np.zeros(100)
        penalties_log = np.zeros(100)
        efficiency_log = np.ones(100)
        steps_log = np.zeros(100)
        deadlines_log = np.zeros(100)
        self.data_log = pd.DataFrame({"Steps": steps_log, "Deadlines": deadlines_log, \
                                      "Penalties": penalties_log, "Success": success_log, "Efficiency": efficiency_log})
        self.data_log.index.name = "Trial"


    def reset(self, destination=None):
        self.planner.route_to(destination)
        deadline = self.env.get_deadline(self)
        self.sum_deadlines += deadline + 1

        # Prepare for a new trip; reset any variables here, if required
        
        # update state variable for new trial run
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        self.state = [inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint]

        # Determine first action to take by maximizing reduced Q-matrix Qsa
        Qsa = dict()
        for act in self.valid_real_actions:
            # If certain Q-matrix elements are not present, initialize to random value
            if not((tuple(self.state),act) in self.Q_matrix):
                # initialize random element with a value > 2 (optimistic initialization, as point out by reviewer)
                self.Q_matrix[(tuple(self.state),act)] = 2. + random.random() 
            Qsa[act] = self.Q_matrix[(tuple(self.state), act)]
        # Determine first action to take in this state based on Qsa
        next_action, _ = max(Qsa.iteritems(), key=lambda x:x[1])
        self.next_action = next_action
        self.k_trial += 1
        self.trial_step = 0 # counter for number of step in a trial
        self.data_log.ix[self.k_trial, "Deadlines"] = deadline + 1       
        

    def update(self, t):

        print "Current self.state: {}".format(self.state)        
        deadline = self.env.get_deadline(self)

        # First task: pick a random step
        rand_action = random.choice(self.env.valid_actions)
              
        # TODO: Select action according to your policy
        randomizer = random.random()
        if randomizer < self.epsilon:
            action = rand_action
        else:
            # next action has already been determined
            action = self.next_action
        
        # Update epsilon to reduce randomness as learning occurs
        self.epsilon = self.epsilon*self.eps_decay    

        # Save current state before updating (needed for update of Q matrix)
        self.prev_state = self.state

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.ctr_steps += 1
        self.data_log.ix[self.k_trial, "Steps"] += 1

        if reward < 0:
            self.ctr_neg_rewards += 1
            self.data_log.ix[self.k_trial, "Penalties"] += 1./self.data_log.ix[self.k_trial, "Deadlines"]
        
        # If agent is done, increase overall success count
        if self.env.done:
            self.success_count += 1
            self.data_log.ix[self.k_trial, "Success"] = 1
            self.data_log.ix[self.k_trial, "Efficiency"] = 1.*self.data_log.ix[self.k_trial, "Steps"]/self.data_log.ix[self.k_trial, "Deadlines"]
            
        # Update state variable after step taken
        inputs = self.env.sense(self)
        self.next_waypoint = self.planner.next_waypoint()
        self.state = [inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint]

        # Determine next action using a reduced, local Q-matrix Qsa
        Qsa = dict()
        for act in self.env.valid_actions:
            # if certain Q-matrix elements are not present, initialize with random value
            if not((tuple(self.state),act) in self.Q_matrix):
                self.Q_matrix[(tuple(self.state),act)] = 2. + random.random()
            Qsa[act] = self.Q_matrix[(tuple(self.state), act)]

        # Choose action with maximum value in local Q-matrix
        next_action, max_Qsa = max(Qsa.iteritems(), key=lambda x:x[1])
        self.next_action = next_action

        # Q-matrix element of state before action, that will be updated
        oldQsa = self.Q_matrix[(tuple(self.prev_state), action)]
        
        # Learn policy based on state, action, reward
        self.Q_matrix[(tuple(self.prev_state), action)] = (1-self.alpha)*oldQsa + self.alpha*(reward + self.gamma*max_Qsa)
        
        # Update the learning rate alpha
        self.alpha = self.alpha*self.alpha_decay

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=1e-5, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False
    n_tr = 100
    sim.run(n_trials=n_tr)  # run for a specified number of trials
    successes = sim.env.primary_agent.success_count
    print "\nSuccesses = {} / {} trials = {:.1%} success rate".format(successes, n_tr, float(successes)/n_tr)
    print "# Negative rewards in {} / {} total steps = {:.1%} negative reward rate".format(sim.env.primary_agent.ctr_neg_rewards, sim.env.primary_agent.ctr_steps, float(sim.env.primary_agent.ctr_neg_rewards)/sim.env.primary_agent.ctr_steps)
    print "# Steps = {}, total sum of deadlines = {}, share of time needed {:.1%}".format(sim.env.primary_agent.ctr_steps, sim.env.primary_agent.sum_deadlines, float(sim.env.primary_agent.ctr_steps)/sim.env.primary_agent.sum_deadlines)
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
   

    #fig, axes = plt.subplots(1, 3, figsize=(16,6))
    #axes[0].plot(sim.env.primary_agent.data_log.index, sim.env.primary_agent.data_log["Penalties"],'ro')
    #axes[0].set_title('Share of penalties in a trial')
    #axes[0].set_ylabel('Penalty rate')
    #axes[0].set_xlabel('Trials')
    #axes[0].set_ylim((-.1,1.1))
    #axes[1].plot(sim.env.primary_agent.data_log.index, sim.env.primary_agent.data_log["Success"],'bo')
    #axes[1].set_title('Successes')
    #axes[1].set_ylabel('Success yes/no')
    #axes[1].set_xlabel('Trials')
    #axes[1].set_ylim((-.1,1.1))
    #axes[2].plot(sim.env.primary_agent.data_log.index, sim.env.primary_agent.data_log["Efficiency"],'go')
    #axes[2].set_title('Efficiency (no. steps vs. deadline)')
    #axes[2].set_ylabel('# Steps needed vs. Deadline')
    #axes[2].set_xlabel('Trials')
    #axes[2].set_ylim((-.1,1.1))
    #plt.savefig('charts.pdf')
    #plt.show()
    
    penalties = pd.rolling_mean(sim.env.primary_agent.data_log["Penalties"],10)
    successes = pd.rolling_mean(sim.env.primary_agent.data_log["Success"],10)
    efficiency = pd.rolling_mean(sim.env.primary_agent.data_log["Efficiency"],10)
    

    fig, axes = plt.subplots(1, 3, figsize=(16,6))
    axes[0].plot(sim.env.primary_agent.data_log.index, penalties,'ro-')
    axes[0].set_title('Rolling mean of penalty share')
    axes[0].set_ylabel('Rolling mean of penalty share in a trial')
    axes[0].set_xlabel('Trials')
    axes[0].set_ylim((-.1,1.1))
    axes[1].plot(sim.env.primary_agent.data_log.index, successes,'bo-')
    axes[1].set_title('Rolling success rate')
    axes[1].set_ylabel('Rolling mean of success yes/no')
    axes[1].set_xlabel('Trials')
    axes[1].set_ylim((-.1,1.1))
    axes[2].plot(sim.env.primary_agent.data_log.index, efficiency,'go-')
    axes[2].set_title('Rolling mean of efficiency')
    axes[2].set_ylabel('Rolling mean of # Steps needed vs. Deadline')
    axes[2].set_xlabel('Trials')
    axes[2].set_ylim((-.1,1.1))
    plt.savefig('charts.pdf')
    plt.show()
    
if __name__ == '__main__':
    run()

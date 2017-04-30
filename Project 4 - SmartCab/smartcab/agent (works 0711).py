import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.Q_matrix = dict()
        # TODO: Initialize any additional variables here
        self.epsilon_start = 0.98
        self.epsilon = self.epsilon_start
        self.eps_decay = 1 - 4e-3
        self.alpha = 0.3
        self.gamma = 0.1
        self.success_count = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        # 
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        self.state = [inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint]
        Qsa = dict()
        for act in self.env.valid_actions:
            if not((tuple(self.state),act) in self.Q_matrix):
                self.Q_matrix[(tuple(self.state),act)] = random.random()
            Qsa[act] = self.Q_matrix[(tuple(self.state), act)]
        
        next_action, _ = max(Qsa.iteritems(), key=lambda x:x[1])
        
        self.next_action = next_action
        

    def update(self, t):
        # Gather inputs
        #self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        #inputs = self.env.sense(self)
        
        deadline = self.env.get_deadline(self)

        # First task: random step
        rand_action = random.choice(self.env.valid_actions)
        
        # TODO: Update state
        #self.state = [inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint]
        #self.state = [inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint]
        print "Current self.state: {}".format(self.state)
        
        # TODO: Select action according to your policy
        randomizer = random.random()
        if randomizer < self.epsilon:
            action = rand_action
        else:
            action = self.next_action
        
        self.epsilon = self.epsilon*self.eps_decay    

        self.prev_state = self.state

        # Execute action and get reward
        reward = self.env.act(self, action)
        
        # If agent is done, increase overall success count
        if self.env.done:
            self.success_count += 1
            
        # Update state
        inputs = self.env.sense(self)
        self.next_waypoint = self.planner.next_waypoint()
        self.state = [inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint]

        # Determine next action
        Qsa = dict()
        for act in self.env.valid_actions:
            if not((tuple(self.state),act) in self.Q_matrix):
                self.Q_matrix[(tuple(self.state),act)] = random.random()
            Qsa[act] = self.Q_matrix[(tuple(self.state), act)]
           
        next_action, max_Qsa = max(Qsa.iteritems(), key=lambda x:x[1])

        oldQsa = self.Q_matrix.get((tuple(self.prev_state), action),random.random())
        
        # Learn policy based on state, action, reward
        self.Q_matrix[(tuple(self.prev_state), action)] = (1-self.alpha)*oldQsa + self.alpha*(reward + self.gamma*max_Qsa)
        
        self.next_action = next_action
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False
    n_tr = 100
    sim.run(n_trials=n_tr)  # run for a specified number of trials
    print "Success rate = {} / {} trials = {}%".format(sim.env.primary_agent.success_count, n_tr, 100.*sim.env.primary_agent.success_count/float(n_tr))
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()

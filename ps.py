import numpy as np

class PSAgent(object):
    def __init__(self, num_actions, glow=0.1, damp=0., softmax=0.1):
        """
        Simple, 2-layered projective simulation (PS) agent.

        We initialize an h-matrix with a single row of `num_actions` entries
        corresponding to a dummy percept clip being connected to all possible
        actions with h-values of all 1.

        We initialize a g-matrix with a single row of `num_actions` entries with
        all 0s corresponding to the *glow* values of percept-action transitions.

        Args:
            num_actions (int): The number of available actions.
            glow (float, optional): The glow (or eta) parameter. Defaults to 0.1
            damp (float, optional): The damping (or gamma) parameter. 
                                    Defaults to 0.
            softmax (float, optional): The softmax (or beta) parameter. 
                                       Defaults to 0.1.                

        NOTE: This simple version misses some features such as clip deletion,
              emotion tags or generalization mechanisms.
        """
        self.num_actions = num_actions
        self.glow = glow
        self.damp = damp
        self.softmax = softmax
        #int: current number of percepts.
        self.num_percepts = 0
        #np.ndarray: h-matrix with current h-values. Defaults to all 1.
        self.hmatrix = np.ones([1,self.num_actions])
        #np.ndarray: g-matrix with current glow values. Defaults to all 0.
        self.gmatrix = np.zeros([1,self.num_actions])
        #dict: Dictionary of percepts as {"percept": index}
        self.percepts = {}
        
    def predict(self, observation):
        """
        Given an observation, returns an action.

        (1) Create a percept from an observation.
        (2) Add percept if it has not been encountered before.
        (3) Get action from h-values.
        (4) Update g-matrix.

        Args:
            percept (object): A percept in form of an object.
        
        Returns:
            action (int): The action to be performed.
        """
        # (1) create percept from observation
        percept = self._get_percept(observation)
        # (2) add percept to clip network if it has not been encountered before
        if percept not in self.percepts.keys():
            # add new percept
            self.percepts[percept] = self.num_percepts
            # increment number of percepts
            self.num_percepts += 1
            # add column to h-matrix
            self.hmatrix = np.append(self.hmatrix, 
                                     np.ones([1,self.num_actions]),
                                     axis=0)
            # add column to g-matrix
            self.gmatrix = np.append(self.gmatrix, 
                                     np.zeros([1,self.num_actions]),
                                     axis=0)
        
        # (3) get action from h-value
        # get index from dictionary entry
        percept_index = self.percepts[percept]
        # get h-values
        h_values = self.hmatrix[percept_index]
        # get probabilities from h-values through a softmax function
        prob = self._softmax(h_values)
        # get action
        action = np.random.choice(range(self.num_actions), p=prob)
        
        # (4) update g-matrix
        self.gmatrix[int(percept_index),int(action)] = 1.

        return action

    def train(self, reward):
        """
        Given a reward, updates h-matrix. 
        Updates g-matrix with glow.

        The h- and g-matrices are updated according to

        .. math::
            h^{(t+1)} = h^{(t)}-\gamma(h^{(t)}-1)+\lambda g^{(t)}\\
            g^{(t+1)} = (1-\eta)g^{(t)}

        Args:
            reward (float): The received reward.
        """
        # damping h-matrix
        self.hmatrix = self.hmatrix - self.damp*(self.hmatrix-1.)
        # update h-matrix
        self.hmatrix += reward*self.gmatrix
        # update g-matrix
        self.gmatrix = (1-self.glow)*self.gmatrix
    
    # ----------------- helper methods -----------------------------------------

    def _get_percept(self, observation):
        """
        Given an observation, returns a percept.
        This function is just to emphasize the difference between observations
        issued by the environment and percepts which describe the observations
        as perceived by the agent.

        Args:
            observation (object): The observation in some form of encoding.

        Returns:
            percept (str): The observation encoded as a percept.
        """
        percept = str(observation)
        return percept
    
    def _softmax(self, x):
        """
        Given an input, calculates the normalized exponential function.

        Args:
            x (np.ndarray): The input array.
        
        Returns:
            softmax_x (np.ndarray): The softmax distribution over the input.
        """
        # rescale exponential to avoid large numbers
        rescale = max(x)
        exp_x = np.exp(self.softmax*(x-rescale))
        # get normalization
        norm = sum(exp_x)
        # calculate normalized exponential
        softmax_x = exp_x/norm

        return softmax_x

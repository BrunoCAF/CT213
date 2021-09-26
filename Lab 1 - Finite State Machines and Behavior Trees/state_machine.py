import random
import math
from constants import *


class FiniteStateMachine(object):
    """
    A finite state machine.
    """
    def __init__(self, state):
        self.state = state

    def change_state(self, new_state):
        self.state = new_state

    def update(self, agent):
        self.state.check_transition(agent, self)
        self.state.execute(agent)


class State(object):
    """
    Abstract state class.
    """
    def __init__(self, state_name):
        """
        Creates a state.

        :param state_name: the name of the state.
        :type state_name: str
        """
        self.state_name = state_name

    def check_transition(self, agent, fsm):
        """
        Checks conditions and execute a state transition if needed.

        :param agent: the agent where this state is being executed on.
        :param fsm: finite state machine associated to this state.
        """
        raise NotImplementedError("This method is abstract and must be implemented in derived classes")

    def execute(self, agent):
        """
        Executes the state logic.

        :param agent: the agent where this state is being executed on.
        """
        raise NotImplementedError("This method is abstract and must be implemented in derived classes")


class MoveForwardState(State):
    def __init__(self):
        super().__init__("MoveForward")
        # Todo: add initialization code
        self.running_time = 0

    def check_transition(self, agent, state_machine):
        # Todo: add logic to check and execute state transition
        if agent.get_bumper_state():
            state_machine.change_state(GoBackState())
        if self.running_time * SAMPLE_TIME > MOVE_FORWARD_TIME:
            state_machine.change_state(MoveInSpiralState())
        pass

    def execute(self, agent):
        # Todo: add execution logic
        agent.set_velocity(FORWARD_SPEED, 0)
        self.running_time += 1
        pass


class MoveInSpiralState(State):
    def __init__(self):
        super().__init__("MoveInSpiral")
        # Todo: add initialization code
        self.running_time = 0
    
    def check_transition(self, agent, state_machine):
        # Todo: add logic to check and execute state transition
        if agent.get_bumper_state():
            state_machine.change_state(GoBackState())
        if self.running_time * SAMPLE_TIME > MOVE_IN_SPIRAL_TIME:
            state_machine.change_state(MoveForwardState())
        pass

    def execute(self, agent):
        # Todo: add execution logic
        b = SPIRAL_FACTOR
        r = INITIAL_RADIUS_SPIRAL + b*self.running_time*SAMPLE_TIME

        agent.set_velocity(FORWARD_SPEED, FORWARD_SPEED*(2*b*b + r*r)/(b*b + r*r)**(5/4))
        self.running_time += 1
        pass


class GoBackState(State):
    def __init__(self):
        super().__init__("GoBack")
        # Todo: add initialization code
        self.running_time = 0

    def check_transition(self, agent, state_machine):
        # Todo: add logic to check and execute state transition
        if self.running_time * SAMPLE_TIME > GO_BACK_TIME:
            state_machine.change_state(RotateState())
        pass

    def execute(self, agent):
        # Todo: add execution logic
        agent.set_velocity(BACKWARD_SPEED, 0)
        self.running_time += 1
        pass


class RotateState(State):
    def __init__(self):
        super().__init__("Rotate")
        # Todo: add initialization code
        self.running_time = 0
        self.theta = random.gauss(math.pi, 1)
        if self.theta < 0:
            self.theta += 2 * math.pi
        elif self.theta > 2 * math.pi:
            self.theta -= 2 * math.pi

    def check_transition(self, agent, state_machine):
        # Todo: add logic to check and execute state transition
        if self.running_time * SAMPLE_TIME > self.theta / ANGULAR_SPEED:
            state_machine.change_state(MoveForwardState())
        pass
    
    def execute(self, agent):
        # Todo: add execution logic
        agent.set_velocity(0, ANGULAR_SPEED)
        self.running_time += 1
        pass

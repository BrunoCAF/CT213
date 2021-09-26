import numpy as np
import random
from math import inf


def clamp(value_array, lower_bound, upper_bound):
    """
    Auxiliary method to keep the values from a given array between the given bounds.

    :param value_array: Array of values to be clamped.
    :type value_array: numpy array.
    :param lower_bound: Array of lower bounds.
    :type lower_bound: numpy array.
    :param upper_bound: Array of upper bounds.
    :type upper_bound: numpy array.
    :return: Array of clamped values.
    :rtype: numpy array.
    """
    for i in range(len(value_array)):
        value_array[i] = max(lower_bound[i], min(value_array[i], upper_bound[i]))

    return value_array


class Particle:
    """
    Represents a particle of the Particle Swarm Optimization algorithm.
    """
    def __init__(self, lower_bound, upper_bound):
        """
        Creates a particle of the Particle Swarm Optimization algorithm.

        :param lower_bound: lower bound of the particle position.
        :type lower_bound: numpy array.
        :param upper_bound: upper bound of the particle position.
        :type upper_bound: numpy array.
        """
        # Todo: implement
        self.x = np.zeros(np.size(lower_bound))
        self.v = np.zeros(np.size(lower_bound))

        for i in range(np.size(self.x)):
            self.x[i] = random.uniform(lower_bound[i], upper_bound[i])

        for i in range(np.size(self.v)):
            self.v[i] = random.uniform(-(upper_bound[i] - lower_bound[i]), upper_bound[i] - lower_bound[i])

        self.best_position, self.best_value = None, -inf


class ParticleSwarmOptimization:
    """
    Represents the Particle Swarm Optimization algorithm.
    Hyperparameters:
        num_particles: number of particles.
        inertia_weight: inertia weight.
        cognitive_parameter: cognitive parameter.
        social_parameter: social parameter.

    :param hyperparams: hyperparameters used by Particle Swarm Optimization.
    :type hyperparams: Params.
    :param lower_bound: lower bound of particle position.
    :type lower_bound: numpy array.
    :param upper_bound: upper bound of particle position.
    :type upper_bound: numpy array.
    """
    def __init__(self, hyperparams, lower_bound, upper_bound):
        # Todo: implement
        self.num_particles = hyperparams.num_particles
        self.w = hyperparams.inertia_weight
        self.phip = hyperparams.cognitive_parameter
        self.phig = hyperparams.social_parameter
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.particles = []
        for i in range(self.num_particles):
            p = Particle(self.lower_bound, self.upper_bound)
            self.particles.append(p)

        self.best_position, self.best_value = None, -inf

        self.particle_schedule = 0

    def get_best_position(self):
        """
        Obtains the best position so far found by the algorithm.

        :return: the best position.
        :rtype: numpy array.
        """
        # Todo: implement
        return self.best_position

    def get_best_value(self):
        """
        Obtains the value of the best position so far found by the algorithm.

        :return: value of the best position.
        :rtype: float.
        """
        # Todo: implement
        return self.best_value

    def get_position_to_evaluate(self):
        """
        Obtains a new position to evaluate.

        :return: position to evaluate.
        :rtype: numpy array.
        """
        # Todo: implement
        return self.particles[self.particle_schedule].x

    def advance_generation(self):
        """
        Advances the generation of particles.
        """
        # Todo: implement
        for particle in self.particles:
            rp, rg = random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)

            particle.v = self.w * particle.v + self.phip * rp * (particle.best_position - particle.x) + \
                self.phig * rg * (self.best_position - particle.x)
            vmin, vmax = -(self.upper_bound - self.lower_bound), (self.upper_bound - self.lower_bound)
            particle.v = clamp(particle.v, vmin, vmax)

            particle.x = clamp(particle.x + particle.v, self.lower_bound, self.upper_bound)

    def notify_evaluation(self, value):
        """
        Notifies the algorithm that a particle position evaluation was completed.

        :param value: quality of the particle position.
        :type value: float.
        """
        # Todo: implement
        p = self.particles[self.particle_schedule]

        if value > p.best_value:
            p.best_position, p.best_value = p.x, value

        if p.best_value > self.best_value:
            self.best_position, self.best_value = p.best_position, p.best_value

        self.particle_schedule += 1

        if self.particle_schedule == self.num_particles:
            self.advance_generation()
            self.particle_schedule = 0

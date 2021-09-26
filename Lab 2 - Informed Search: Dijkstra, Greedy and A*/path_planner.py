from grid import Node, NodeGrid
from math import inf
import heapq


class PathPlanner(object):
    """
    Represents a path planner, which may use Dijkstra, Greedy Search or A* to plan a path.
    """
    def __init__(self, cost_map):
        """
        Creates a new path planner for a given cost map.

        :param cost_map: cost used in this path planner.
        :type cost_map: CostMap.
        """
        self.cost_map = cost_map
        self.node_grid = NodeGrid(cost_map)

    @staticmethod
    def construct_path(goal_node):
        """
        Extracts the path after a planning was executed.

        :param goal_node: node of the grid where the goal was found.
        :type goal_node: Node.
        :return: the path as a sequence of (x, y) positions: [(x1,y1),(x2,y2),(x3,y3),...,(xn,yn)].
        :rtype: list of tuples.
        """
        node = goal_node
        # Since we are going from the goal node to the start node following the parents, we
        # are transversing the path in reverse
        reversed_path = []
        while node is not None:
            reversed_path.append(node.get_position())
            node = node.parent
        return reversed_path[::-1]  # This syntax creates the reverse list

    def dijkstra(self, start_position, goal_position):
        """
        Plans a path using the Dijkstra algorithm.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
        # Todo: implement the Dijkstra algorithm
        self.node_grid.reset()

        pq = []
        start = self.node_grid.get_node(start_position[0], start_position[1])
        start.g = 0
        heapq.heappush(pq, (start.g, start))

        while pq:
            node = heapq.heappop(pq)[1]
            if node.closed:
                continue
            else:
                node.closed = True
            (i, j) = node.get_position()
            for successor_position in self.node_grid.get_successors(i, j):
                successor = self.node_grid.get_node(successor_position[0], successor_position[1])
                if successor.g > node.g + self.cost_map.get_edge_cost((i, j), successor_position):
                    successor.g = node.g + self.cost_map.get_edge_cost((i, j), successor_position)
                    successor.parent = node
                    heapq.heappush(pq, (successor.g, successor))

        # The first return is the path as sequence of tuples (as returned by the method construct_path())
        # The second return is the cost of the path
        goal = self.node_grid.get_node(goal_position[0], goal_position[1])
        return self.construct_path(goal), goal.g  # Feel free to change this line of code

    def greedy(self, start_position, goal_position):
        """
        Plans a path using greedy search.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
        # Todo: implement the Greedy Search algorithm
        self.node_grid.reset()

        pq = []
        start = self.node_grid.get_node(start_position[0], start_position[1])
        start.g = 0
        heapq.heappush(pq, (start.distance_to(goal_position[0], goal_position[1]), start))

        while pq:
            node = heapq.heappop(pq)[1]
            if node.closed:
                continue
            else:
                node.closed = True

            (i, j) = node.get_position()
            for successor_position in self.node_grid.get_successors(i, j):
                successor = self.node_grid.get_node(successor_position[0], successor_position[1])
                if (not successor.closed) and successor.parent is None:
                    successor.parent = node
                    successor.g = node.g + self.cost_map.get_edge_cost(node.get_position(), successor_position)

                if successor_position == goal_position:
                    return self.construct_path(successor), successor.g

                heapq.heappush(pq, (successor.distance_to(goal_position[0], goal_position[1]), successor))

        # The first return is the path as sequence of tuples (as returned by the method construct_path())
        # The second return is the cost of the path

    def a_star(self, start_position, goal_position):
        """
        Plans a path using A*.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
        # Todo: implement the A* algorithm
        self.node_grid.reset()

        pq = []
        start = self.node_grid.get_node(start_position[0], start_position[1])
        start.g = 0
        start.f = start.distance_to(goal_position[0], goal_position[1])
        heapq.heappush(pq, (start.f, start))

        while pq:
            node = heapq.heappop(pq)[1]
            if node.closed:
                continue
            else:
                node.closed = True
            (i, j) = node.get_position()

            if (i, j) == goal_position:
                return self.construct_path(node), node.f
            
            for successor_position in self.node_grid.get_successors(i, j):
                successor = self.node_grid.get_node(successor_position[0], successor_position[1])
                h = successor.distance_to(goal_position[0], goal_position[1])
                if successor.f > node.g + self.cost_map.get_edge_cost((i, j), successor_position) + h:
                    successor.g = node.g + self.cost_map.get_edge_cost((i, j), successor_position)
                    successor.f = successor.g + h
                    successor.parent = node
                    heapq.heappush(pq, (successor.f, successor))

        # The first return is the path as sequence of tuples (as returned by the method construct_path())
        # The second return is the cost of the path

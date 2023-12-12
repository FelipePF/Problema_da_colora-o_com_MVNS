import random
from time import time

import numpy as np


class MVNS_GC:
    """Classe para implementar o Memorized VNS aplicado ao problema de
    coloração de grafos."""

    def __init__(self, path_to_instance, max_iterations=10, log=False):

        self.log = log
        self.max_iterations = max_iterations
        graph = self.read_graph(path_to_instance)
        self.k_max = 4  # quanidade de vizinhanças
        self.memory = []

        self.run(graph, max_iterations, self.k_max)

    def read_graph(self, path_to_instance):
        """Lê instancias no formato DIMACS.


        Args:
            path_to_instance (int): Caminho até a instância

        Returns:
            dict: Instancia do problema da coloração {vertice: [nós ligados]}
        """
        with open(path_to_instance) as instance:
            lines = instance.readlines()

            for line in lines:
                if line.startswith('c '):
                    continue

                if line.startswith('p '):
                    graph_len = line.split()[2]
                    graph = {str(v+1): [] for v in range(int(graph_len))}

                if line.startswith('e '):
                    source, destiny = line.removeprefix('e ').removesuffix('\n').split()  # noqa e501
                    if destiny not in graph[source]:
                        graph[source].append(destiny)
                    if source not in graph[destiny]:
                        graph[destiny].append(source)
            return graph

    def generate_initial_solution(self, graph):
        """Gera uma solução inicial aleatória para o problema de coloração.

        Args:
            graph (dict): Grafo da instância

        Returns:
            dict: Solução inicial do problema {Vértice: cor}
        """
        if self.log:
            print('\t[ MVNS ] Gerando solução inicial.')
        initial_solution = {}
        # vertices = list(graph.keys())
        # random.shuffle(vertices)

        # for vertice in vertices:
        #     cores_vizinhos = {initial_solution[v] for v in graph[vertice]
        #                       if v in initial_solution}
        #     cor = self.pick_color(cores_vizinhos)
        #     initial_solution[vertice] = cor

        initial_solution = {node: int(node) for node in graph}

        return initial_solution

    def is_valid(self, solution, graph):
        """Verifica se a solução é viável para o problema de coloração.

        Args:
            solution (dict): Solução do problema
            graph (dict): Grafo do problema

        Returns:
            bool: A solução é valida?
        """
        for node in graph:  # Para cada vertice no grafo
            for neighbor in graph[node]:  # Para cada  vizinho no grafo
                # Verifica se a cor do vértice é igual do vizinho
                if solution[node] == solution[neighbor]:
                    return False
        return True

    def pick_color(self, cores_vizinhos):
        """
        Escolhe uma cor que não está presente nos vértices vizinhos.
        """
        cores_disponiveis = set(range(len(cores_vizinhos) + 1))
        return min(cores_disponiveis - cores_vizinhos)

    def pertubation(self, graph, k, solution):
        new_sol = solution.copy()
        match k:
            case 1:  # Troca a cor de um vértice olhando sua vizinhança
                for _ in range(int(len(graph)*.05)):
                    while True:
                        node = random.choice(list(new_sol.keys()))
                        if node not in self.memory:
                            break

                    neighbors = graph[node]
                    neighbors_colors = {new_sol[v]
                                        for v in neighbors if v in new_sol}
                    new_color = self.pick_color(neighbors_colors)

                    new_sol[node] = new_color
                    self.memory.append(node)

                return new_sol
            case 2:  # Troca a cor de um vertice aleatorio
                for _ in range(int(len(graph)*.05)):

                    while True:
                        node = np.random.choice(len(new_sol.keys()), 1)[0]
                        if node not in self.memory:
                            break
                    # Troca a cor do vertice escolhido
                    new_sol[node] = str(random.randint(1, len(new_sol)))

                    self.memory.append(node)
                return new_sol

            case 3:
                for _ in range(int(len(graph)*.05)):
                    while True:
                        node = random.choice(list(new_sol.keys()))
                        if node not in self.memory:
                            break

                    neighbors = graph[node]
                    for neighbor in neighbors:
                        new_sol[neighbor] = random.choice(list(set(solution.values())))  # noqa e501

                return new_sol

            case 4:  # SWAP
                while True:
                    node1 = random.choice(list(new_sol.keys()))
                    if node1 not in self.memory:
                        break
                while True:
                    node2 = random.choice(list(new_sol.keys()))
                    # node 2 não está na memoria e é adjacente a node1
                    if (node2 not in self.memory and node2 in graph[node1]):
                        break

                new_sol[node1], new_sol[node2] = new_sol[node2], new_sol[node1]  # noqa e501

                self.memory.append(node1)
                self.memory.append(node2)
                return new_sol

    def local_search(self, k, solution, graph):
        """Implementa a busca local para o problema de coloração.

        Args:
            k (int): Númeor da vizinhaça que será usada
            solution (dict): Solução para a instancia analisada
            graph (dict): Grafo da instancia analisada

        Returns:
            dict: Nova solução para o problema
        """
        if self.log:
            print(f'Iniciando Busca Local na vizinhança {k}.')
        i = 0
        new_sol = solution.copy()
        while True:
            if i == 10:
                # Retorna a solução atual, caso fique 100 iterações sem melhora
                return new_sol

            match k:
                case 1:
                    for node in graph:
                        curr_color = new_sol[node]
                        neighbors = graph[node]
                        neighbors_colors = {new_sol[v]
                                            for v in neighbors if v in new_sol}
                        new_color = self.pick_color(neighbors_colors)

                        if new_color < curr_color:
                            new_sol[node] = new_color
                            if (self.is_valid(new_sol, graph) and
                                    self.get_cost(new_sol) < self.get_cost(solution)):  # noqa e501
                                return new_sol
                            else:
                                new_sol = solution.copy()

                case 2:  # Troca a cor de um vertice aleatorio
                    colors = set(solution.values())
                    node = random.choice(list(new_sol.keys()))
                    # Troca a cor do vertice escolhido
                    for color in colors:
                        if color != new_sol[node]:
                            new_sol[node] = color
                            if (self.is_valid(new_sol, graph) and
                                    self.get_cost(new_sol) < self.get_cost(solution)):  # noqa e501
                                return new_sol
                            else:
                                new_sol = solution.copy()
                case 3:  # Troca a cor de todos os vizinhos de um vertice
                    for node in graph.keys():
                        neighbors = graph[node]
                        for neighbor in neighbors:
                            new_sol[neighbor] = random.choice(list(set(solution.values())))  # noqa e501
                        if (self.is_valid(new_sol, graph) and
                                self.get_cost(new_sol) < self.get_cost(solution)):  # noqa e501
                            return new_sol
                        else:
                            new_sol = solution.copy()
                case 4:  # SWAP
                    for node1 in graph.keys():
                        for node2 in graph.keys():
                            # Garante que sejam adjacentes
                            if node2 not in graph[node1]:
                                continue
                            new_sol[node1], new_sol[node2] = new_sol[node2], new_sol[node1]  # noqa e501
                            if (self.is_valid(new_sol, graph) and
                                    self.get_cost(new_sol) < self.get_cost(solution)):  # noqa e501
                                return new_sol
                            else:
                                new_sol = solution.copy()
            i += 1

    def get_cost(self, solution):
        """Calcula o custo da solução (quantidade de cores diferentes).

        Args:
            solution (dict): Solução para a instancia analisada

        Returns:
            int: Custo da solução (Quantidade de cores diferentes)
        """
        return len(set(solution.values()))

    def vns(self, graph: dict, max_colors: int,
            max_iterations: int, k_max: int) -> dict:
        """Implementa a Variable Neighborhood Search para o problema
        de coloração.

        Args:
            graph (dict): Grafo da instancia analisada
            max_colors (int): Número maximo de cores
            max_iterations (int): Númeor máximo de iterações
            k_max (int): Quantidade de vizinhanças

        Returns:
            dict: Melhor solução encontrada
        """
        current_solution = self.generate_initial_solution(graph)

        print(f'Inital solution: {self.get_cost(current_solution)}')

        for iteration in range(max_iterations):
            k = 1
            while k <= k_max:
                perturbed_solution = self.pertubation(
                    graph, 1,
                    current_solution)
                new_solution = self.local_search(k, perturbed_solution, graph)

                if (self.get_cost(new_solution) < self.get_cost(current_solution)):  # noqa e501
                    self.memory = self.memory[-1:]
                    current_solution = new_solution.copy()
                    k = 1  # Reinicia o contador se uma melhoria foi encontrada

                else:
                    k += 1
            if iteration % 2 == 0:  # Limpa a memoria a cada 2 iterações
                self.memory = self.memory[-1:]
            print(f'Iteration {iteration + 1}/{max_iterations}', end=' - ')
            print(f'Melhor Solução: {self.get_cost(current_solution)}')

        return current_solution

    def run(self, graph, max_iterations, k_max):
        start = time()
        num_colors = len(graph)

        best_solution = self.vns(
            graph, num_colors, max_iterations, k_max)
        print(f'Melhor solução: {self.get_cost(best_solution)}')
        print(f'Tempo de processamento: {int(time() - start)}')


MVNS_GC('neighborhoods/DSJC500.5.col')

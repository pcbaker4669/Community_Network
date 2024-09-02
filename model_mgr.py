# color nodes by attribute
# https://stackoverflow.com/questions/28910766/python-networkx-set-node-color-automatically-based-on-number-of-attribute-opt

import random as rnd
import networkx as nx
import numpy as np
import agent as ag
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import rcParams

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg
)
import pydot
import graphviz as gv

# originally 123
rnd.seed(123)
# right now color is mapped to community, but we want color to be mapped
# to content, maybe shape to content is better?  like 0-1 square
com_color_list = [  # rgbv
    "violet", "blue", "purple", "red", "orange", "green",
    "black",
    "gray", "lightgray", "rosybrown", "lightcoral",
    "maroon", "goldenrod", "gold",
    "yellow", "greenyellow", "lightgreen", "turquoise"
                                           "teal", "cyan", "cadetblue", "dodgerblue",
    "darkblue", "slateblue", "mediumpurple", "blueviolet",
    "fuchsia", "deeppink", "hotpink", "pink"]

com_shape_list = ['o', 's', 'D', '^', '*', 'x', '>', 'p', 'h', '+']

temperatures = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]


class SNA_Model_Mgr:

    def __init__(self):
        self.start_node_num = 4
        self.tot_nodes = 100
        self.g = None
        self.sna_model = None
        self.selection_mult = 1  # how many attempts to find a good node times nodes
        self.network_change = .1
        self.color_map = []
        self.shape_map = []
        self.num_comm_groups = 0
        self.comm_avg_std = 0

        # book keeping values
        self.removals = 0
        self.old_tgt_skipped = 0
        self.edge_adds = 0
        self.new_nodes_skipped = 0
        self.tot_mods = 0

    def set_parameters(self, tot_nodes, network_change):
        self.tot_nodes = tot_nodes
        self.color_map = ["black"] * tot_nodes
        self.shape_map = ["o"] * tot_nodes
        self.network_change = network_change

    def setup(self):
        if self.sna_model:
            print("killing old graph")
        self.g = nx.DiGraph()
        self.sna_model = ag.SNA_Model()
        self.g.clear()
        n = []
        for i in range(0, self.start_node_num):
            starter_node = self.sna_model.create_rnd_node()
            coefficient = i / self.start_node_num + 1 / (2 + self.start_node_num)
            print("word metric coefficient =", coefficient)
            starter_node.set_word_metric(coefficient)
            n.append(starter_node)
            self.g.add_node(n[i].get_name())

        for i in range(1, self.start_node_num):
            self.g.add_edge(n[i - 1].get_name(), n[i].get_name())
        self.g.add_edge(n[self.start_node_num - 1].get_name(), n[0].get_name())

        # add some extra edges
        rnd.shuffle(n)
        for i in range(1, self.start_node_num):
            self.g.add_edge(n[i - 1].get_name(), n[i].get_name())

        rnd.shuffle(n)
        for i in range(1, self.start_node_num):
            self.g.add_edge(n[i - 1].get_name(), n[i].get_name())

        for i in range(0, self.start_node_num):
            degree_in = self.g.in_degree(n[i].get_name())
            n[i].set_in_degree(degree_in)
            self.sna_model.update_graph_totals()

    def go(self):
        nodes_to_add = self.tot_nodes - self.start_node_num
        for i in range(nodes_to_add):
            self.add_node_to_graph()
            if i % 100 == 0:
                print("node = ", i)
        self.do_community_detection()
        print("total nodes = ", len(self.g.nodes))

    def add_node_to_graph(self):
        new_node = self.sna_model.create_rnd_node()
        self.g.add_node(new_node.get_name())
        good_match = self.sna_model.get_good_match(new_node, self.selection_mult)
        if new_node == good_match:
            print("found its self in rec")
        self.g.add_edge(new_node.get_name(), good_match.get_name())
        parent_match = self.sna_model.get_good_match(new_node, self.selection_mult)
        if new_node == parent_match or parent_match == new_node:
            print("found its self for parent or parent is child")
        self.g.add_edge(parent_match.get_name(), new_node.get_name())

        good_match.set_in_degree(self.g.in_degree(good_match.get_name()))
        new_node.set_in_degree(self.g.in_degree(new_node.get_name()))
        parent_match.set_in_degree(self.g.in_degree(parent_match.get_name()))
        self.sna_model.update_graph_totals()

    def modify_graph(self, num_of_nodes):
        all_nodes = self.sna_model.get_nodes()
        keys = list(all_nodes.keys())
        node_mod_cnt = 0

        while node_mod_cnt < num_of_nodes:
            print("mod: {}, num: {}".format(node_mod_cnt, num_of_nodes))
            self.tot_mods += 1
            node_mod_cnt += 1
            k = rnd.choice(keys)
            source_node = all_nodes[k]
            # get worst match of source node
            edges = self.g.out_edges([source_node.get_name()])
            lst_to_check = []
            for e in edges:
                lst_to_check.append(all_nodes[e[1]])

            old_target_node = self.sna_model.get_least_sim_from_lst(source_node, lst_to_check)
            if old_target_node is None:
                print("target node removed skipped", self.old_tgt_skipped)
                self.old_tgt_skipped += 1
                continue

            old_tgt_in_degree = self.g.in_degree(old_target_node.get_name())

            # if old_tgt_in_degree is too low, don't get a new match
            # instead, get a new parent for the old target and remove edge
            new_match = None
            if old_tgt_in_degree > 1:
                new_match = self.sna_model.get_better_match(source_node, old_target_node, lst_to_check)
                if new_match is None:
                    print("new_nodes_skipped {}, total modified {}, cnt this run {}"
                          .format(self.new_nodes_skipped, self.tot_mods, node_mod_cnt))
                    self.new_nodes_skipped += 1
                    continue
                self.g.add_edge(source_node.get_name(), new_match.get_name())
                new_match_degree_in = self.g.in_degree(new_match.get_name())
                new_match.set_in_degree(new_match_degree_in)
            else:
                # need to find a better match for the old target, swap nodes,
                # old_target is the node to match, need. something to be source_node
                # no exclusion list
                parent_match = self.sna_model.get_better_match(old_target_node, source_node)
                if parent_match is None:
                    continue
                self.g.add_edge(parent_match.get_name(), old_target_node.get_name())
            self.g.remove_edge(source_node.get_name(), old_target_node.get_name())

            self.removals += 1
            self.edge_adds += 1

            old_tgt_degree_in = self.g.in_degree(old_target_node.get_name())
            old_target_node.set_in_degree(old_tgt_degree_in)

            self.sna_model.update_graph_totals()

            print("total modified {}, cnt this run {}"
                  .format(self.tot_mods, node_mod_cnt))

    def do_run_modifications(self):
        node_changes_per_update = self.tot_nodes * self.network_change
        self.modify_graph(node_changes_per_update)
        self.do_community_detection()

    def do_community_detection(self):
        community_lst = nx.community.louvain_communities(self.g, resolution=1, seed=1234)
        self.num_comm_groups = len(community_lst)
        self.sna_model.update_communities(community_lst)
        print("community list =", community_lst)

        cnt = 0
        for node in self.g:
            comm_num = 0
            for lst in community_lst:
                if node in lst:
                    #  print("node {} is in list {}".format(node, cnt))
                    self.color_map[cnt] = com_color_list[comm_num]
                comm_num += 1

            val = self.sna_model.get_nodes()[node].get_word_metric()
            print("val = ", val)

            cnt += 1

    def draw_network(self, run_num):
        plt.clf()
        std_list = self.sna_model.get_mean_std_dev_of_com_by_run()
        std = std_list[-1]
        max_in_deg = self.sna_model.get_max_in_degree()

        plt.title("Mod #: {}, # of Comm.: {}, Avg. Comm. Std: {:.3f}, Max iDeg: {}".
                  format((len(std_list)), self.num_comm_groups, std, max_in_deg))

        pos = nx.nx_pydot.pydot_layout(self.g)
        #pos = nx.spring_layout(self.g)
        cnt = 0
        low = 0
        high = 1
        norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
        d = {}
        for i in self.g.nodes:
            d[i] = self.sna_model.get_nodes()[i].get_word_metric()
        print("d = ", d)
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap='coolwarm')
        nx.draw_networkx(self.g, pos,
                         node_color=[mapper.to_rgba(i) for i in d.values()],
                         node_size=150, with_labels=True)


    def get_canvas(self, root):
        fig = plt.figure(frameon=True, figsize=(5, 5), dpi=100)
        canvas = FigureCanvasTkAgg(fig, root)
        return canvas

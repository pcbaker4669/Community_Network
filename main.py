import time
import tkinter as tk
from tkinter import *
import random as rnd
import model_mgr

# originally 123
rnd.seed(123)
start_node_num = 4
tot_nodes = 30
curr_num = 4

net_change = .1

root = Tk()
root.title("Community Network")
tot_nodes_var = tk.StringVar()

net_change_var = tk.StringVar()
tot_nodes_var.set(f"{tot_nodes}")

net_change_var.set(f"{net_change}")
m_mgr = model_mgr.SNA_Model_Mgr()

canvas = m_mgr.get_canvas(root)

canvas.draw()
canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)


def init():
    t_nodes = int(tot_nodes_var.get())
    n_change = float(net_change_var.get())
    m_mgr.set_parameters(t_nodes, n_change)
    m_mgr.setup()
    m_mgr.go()
    m_mgr.draw_network()
    canvas.draw()
    canvas.flush_events()


def step():
    m_mgr.do_run_modifications()
    m_mgr.draw_network()
    canvas.draw()
    canvas.flush_events()


def exit_sna():
    root.quit()


button_frame = Frame(root)
init_btn = tk.Button(button_frame, text="Initialize", command=lambda: init())
step_btn = tk.Button(button_frame, text="Step >>", command=lambda: step())
tot_nodes_lbl = tk.Label(button_frame, text="Total Nodes:")
tot_nodes_ent = tk.Entry(button_frame, textvariable=tot_nodes_var)

net_change_lbl = tk.Label(button_frame, text="Net Chg/Run:")
net_change_ent = tk.Entry(button_frame, textvariable=net_change_var)

init_btn.pack(side="left", padx=5)
step_btn.pack(side="left", padx=5)
tot_nodes_lbl.pack(side="left")
tot_nodes_ent.pack(side="left", padx=5)

net_change_lbl.pack(side="left")
net_change_ent.pack(side="left", padx=5)
button_frame.pack(side="bottom", pady=5)
root.protocol("WM_DELETE_WINDOW", exit_sna)
root.mainloop()

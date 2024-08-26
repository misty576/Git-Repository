import tkinter as tk 
import subprocess
import webbrowser
import os

def Monte_Carlo():
    subprocess.Popen(['python', 'MonteCarlo.py'])


def Time_Complexity():
    subprocess.Popen(['python', 'TIMECOMPLEXITYTEST2.py'])

def PC_Growth_Notebook():
    subprocess.Popen(['jupyter', 'notebook', 'PercentageDifferenceOfficial.ipynb'])

def Simple_Baseline_Refresh():
    subprocess.Popen(['python', 'SimpleBaselineRefresh.py'])


def Complex_Baseline_Refresh():
    subprocess.Popen(['python', 'ComplexBaselineRefresh.py'])


root = tk.Tk()
root.title('Program Launcher')
root.geometry('300x300')

btn_program_1 = tk.Button(root, text="Monte Carlo Simulation", command=Monte_Carlo)
btn_program_1.pack(pady=10)

btn_program_2 = tk.Button(root, text="Time Complexity", command=Time_Complexity)
btn_program_2.pack(pady=10)

btn_program_3 = tk.Button(root, text="Percentage Growth from Brute Force", command=PC_Growth_Notebook)
btn_program_3.pack(pady=10)

btn_program_4 = tk.Button(root, text="Simple Baseline Refresh", command=Simple_Baseline_Refresh)
btn_program_4.pack(pady=10)

btn_program_5 = tk.Button(root, text="Complex Baseline Refresh", command=Complex_Baseline_Refresh)
btn_program_5.pack(pady=10)

root.mainloop()
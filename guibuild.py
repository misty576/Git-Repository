import tkinter as tk 
import subprocess

def launch_program_1():
    subprocess.Popen(['python', 'MonteCarlo.py'])


def launch_program_2():
    subprocess.Popen(['python', 'TIMECOMPLEXITYTEST2.py'])

def launch_program_3():
    subprocess.Popen(['python', 'PercentageDifferenceOfficial.ipynb'])

def launch_program_4():
    subprocess.Popen(['python', 'SimpleBaselineRefresh.py'])


def launch_program_5():
    subprocess.Popen(['python', 'ComplexBaselineRefresh.py'])


root = tk.Tk()
root.title('Program Launcher')
root.geometry('300x300')

btn_program_1 = tk.Button(root, text="Monte Carlo Simulation", command=launch_program_1)
btn_program_1.pack(pady=10)

btn_program_2 = tk.Button(root, text="Time Complexity", command=launch_program_2)
btn_program_2.pack(pady=10)

btn_program_3 = tk.Button(root, text="Percentage Growth from Brute Force", command=launch_program_3)
btn_program_3.pack(pady=10)

btn_program_4 = tk.Button(root, text="Simple Baseline Refresh", command=launch_program_4)
btn_program_4.pack(pady=10)

btn_program_5 = tk.Button(root, text="Complex Baseline Refresh", command=launch_program_5)
btn_program_5.pack(pady=10)

root.mainloop()
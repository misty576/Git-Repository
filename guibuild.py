import tkinter as tk
import subprocess
import os

def Monte_Carlo():
    subprocess.Popen(['python', 'MonteCarlo.py'])

def Time_Complexity():
    subprocess.Popen(['python', 'TIMECOMPLEXITYTEST2.py'])

'''
def PC_Growth_Notebook():
    # Use raw string or double backslashes for Windows paths
    notebook_path = r'C:\Users\JKelly\Research\Git_Repository'
    
    # Check if Jupyter executable is in PATH; if not, use the full path to the executable
    jupyter_command = 'jupyter'
    
    # You might need to use the full path to the Jupyter executable
    # Example: jupyter_command = r'C:\path\to\jupyter.exe'
    
    subprocess.Popen([jupyter_command, 'notebook','pcgexported.py'],
                     cwd=os.path.dirname(notebook_path))

'''
    

def PC_Growth_Notebook():
    subprocess.Popen(['python', 'pcgexported.py'])

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

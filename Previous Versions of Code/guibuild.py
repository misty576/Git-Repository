import tkinter as tk
import subprocess
import webbrowser
import os

def Monte_Carlo():
    subprocess.Popen(['python', 'MonteCarlo.py'])

def Time_Complexity():
    subprocess.Popen(['python', 'TIMECOMPLEXITYTEST2.py'])

def PC_Growth_Notebook():
    # Path to the notebook
    notebook_path = r'C:/Users/JKelly/Research/Git_Repository/pcg.ipynb'
    
    # Start Jupyter Notebook server
    # subprocess.Popen([r'C:/path/to/jupyter.exe', 'notebook'], cwd=os.path.dirname(notebook_path))
    subprocess.Popen(['jupyter', 'notebook'], cwd=os.path.dirname(notebook_path))
    
    # URL to open in browser
    url = f'http://localhost:8888/notebooks/{os.path.basename(notebook_path)}'
    webbrowser.open(url)

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

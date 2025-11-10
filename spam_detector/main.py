# Placeholder for main.py
# main.py
from gui import SpamDetectorApp
import tkinter as tk

def main():
    root = tk.Tk()
    root.title("Spam Detector — Desktop")
    root.geometry("1100x700")
    app = SpamDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

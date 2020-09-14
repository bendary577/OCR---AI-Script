from tkinter import *
from PIL import Image
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import tkinter as tk

        
class GUI():
    def __init__(self):
        self.root = Tk()
        path_lb = Label(self.root, text='Enter The Path')
        path_lb.place(x=200, y=20)        
        self.entry_box = Entry(self.root)
        self.entry_box.place(x=100, y=70, width= 220)        
        browse_btn = Button(self.root, text = "Browse", command= self.browse_image)
        browse_btn.place(x=350, y=70)        
        self.next_btn = Button(self.root, text = "Next", command= self.page_2)
        self.next_btn.place(x=350, y=120)
        
        self.root.geometry("500x200")
        self.root.title("Team AI")
        self.root.configure(background = "#dda")    
        self.root.mainloop()

    def page_2(self):
        self.p2 = Tk()
        tit_lb = Label(self.p2, text="The Image is")
        tit_lb.place(x=200, y=20)
        
        next2_btn = Button(self.p2, text = "Next", command = self.page_3)
        next2_btn.place(x=450, y=450)
        self.p2.geometry("500x500")
        self.p2.title("Team AI")

    def page_3(self):
        self.p3 = Tk()
        tex = Label(self.p3, text='The Text Is :')
        tex.place(x=200, y=20)
        
        result = Label(self.p3, text='000000000000000000000000000000')
        result.place(x=200, y=60)
        
        self.p3.geometry("500x300")
        self.p3.title("Team AI")
        
    def browse_image(self):
        self.filename = askopenfilename()
        if self.filename.endswith(".jpg"):
            self.ok = 1
            print(self.filename)
            self.entry_box.delete(0, END)
            self.entry_box.insert(0, self.filename)
        
        
gui = GUI()


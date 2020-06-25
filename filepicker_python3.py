import tkinter
from tkinter import filedialog
import tkinter.messagebox
import sys
import pdb

def pickfile(path='.'): #add filetypes
    root = Tkinter.Tk()
    root.withdraw()
    f = filedialog.askopenfilename(parent=root,title='Choose a file')
    if f:
        root.destroy()
        del root
        return f
    else:
        print ("No file picked, exiting!")
        root.destroy()
        del root
        sys.exit()

def saveasfile(path='.', filetypes = [], defaultextension=''): #add filetypes
    root = tkinter.Tk()
    root.withdraw()
    f = filedialog.asksaveasfilename(parent=root,title='Choose a filepath to save as',filetypes = filetypes,defaultextension=defaultextension)
    if f:
        root.destroy()
        del root
        return f
    else:
        print ("No file picked, exiting!")
        root.destroy()
        del root
        sys.exit()


def pickfiles(path='.', filetypes = [], defaultextension=''):
    root = tkinter.Tk()
    root.withdraw()
    f = filedialog.askopenfilenames(parent=root,title='Choose a file',filetypes = filetypes)
    if f:
        f=root.tk.splitlist(f)
        root.destroy()
        del root
        return f
    else:
        print ("No file picked, exiting!")
        root.destroy()
        del root
        sys.exit()


def pickdir(path='.'):
    root = Tkinter.Tk()
    root.withdraw()
    dirname = filedialog.askdirectory(parent=root,initialdir=".",title='Please select a directory')

    root.destroy()
    if len(dirname ) > 0:
        return dirname
    else:
        print ("No directory picked, exiting!")
        sys.exit()

def askyesno(title = 'Display?',text = "Use interactive plotting?"):
    root = Tkinter.Tk()
    root.withdraw()
    tf = tkMessageBox.askyesno(title, text)
    root.destroy()
    return tf


if __name__=='__main__':

    fs = pickfiles(filetypes=[('AVI videos','*.avi'),('All files','*.*')])
    print (fs)
    print (type(fs))

from tkinter import *
from pandastable import Table, TableModel
import pandas as pd

from time import *

#df = pd.read_pickle('rnnInput.pd')

class TestApp(Frame):
    """Basic test frame for the table"""
    def __init__(self, dataFrame, parent=None, ):
        self.parent = parent
        Frame.__init__(self)
        self.main = self.master
        self.main.geometry('600x400+200+100')
        self.main.title('Table app')
        f = Frame(self.main)
        f.pack(fill=BOTH,expand=1)

        self.table = pt = Table(f, dataframe=dataFrame,
                                showtoolbar=True, showstatusbar=True)
        pt.show()
        return

def showDF(df, breakIt = False):
    app = TestApp(df)
    #launch the app
    app.mainloop()
    if breakIt:
        exit()

def main():
    import sys
    pickle = sys.argv[1]
    df = pd.read_pickle(pickle)
    if sys.argv[2] == "1":
        showDF(df)
    if sys.argv[2] == "0":
        print df

if __name__ == "__main__":
    main()

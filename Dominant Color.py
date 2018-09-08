from PIL import Image, ImageTk
from collections import namedtuple
import KMeans as km
import numpy as np
import Tkinter 


class ColorPallete(object):
    
    def __init__(self, filename):
        self.filename = None
        self.original = None
        self.thumbnail = None
        self.load_image(filename)
        
        self.points = self.get_points()
 
        self.show_frame = None
        self.btn_frame = None
        
 
        self.build()
        
        
    def load_image(self, filename):
        self.filename = filename
        self.original = Image.open(filename)
        self.thumbnail = self.original.copy()
        self.thumbnail.thumbnail((200, 200))
    
    def get_points(self):
        #FIXME all numpy
        points = []
        w, h = self.thumbnail.size
        for count, color in self.thumbnail.getcolors(w * h):
           for i in range(count):
               points.append(list(color))
        return np.array(points)


    def build(self):
        top = Tkinter.Tk()
        top.title("Dominant colors")
        top.geometry("600x415")
        top.configure(background="white")
        
        img_frame = Tkinter.Frame(top, width=600, height=350, bg="pink", colormap="new")
        
        photo = ImageTk.PhotoImage(self.original.resize((600, 350), Image.ANTIALIAS))
        
        label = Tkinter.Label(img_frame,width=600, height=350, image=photo)
        label.image = photo
        label.pack()
        
        img_frame.pack()
        
        self.show_frame = Tkinter.Frame(width=600, height=30, bg="gray")
        self.show_frame.pack_propagate(False) 
        self.show_frame.pack()
        
        self.btn_frame = Tkinter.Frame(top, width=600, height=30, bg="white")
        self.btn_frame.pack_propagate(False) 
        
        btn = Tkinter.Button(self.btn_frame, width=20, height=2, bg="gray" , text="Kluster")
        btn.pack(side=Tkinter.RIGHT)
        
        n = Tkinter.Text(self.btn_frame, width=20, bg="gray")
        n.insert(Tkinter.END ,"3")
        n.pack(side=Tkinter.RIGHT)
        
        self.btn_frame.pack()

        def type(event):
        
            aantal = int(float(n.get("1.0", "end-1c")))
            kmeans = km.KMeans(aantal_clusters=aantal, vector_lijst=self.points)
            def print_iteration(kmeans, iter):
                print iter,"\t",sum(kmeans.fout)
            kmeans.cluster(stap_callback=print_iteration)
            
            for widget in self.show_frame.winfo_children():
                widget.destroy()
            
            for centroid in kmeans.get_centroids():
                rgb =  tuple(centroid.astype(np.int32))
                color = "#%02x%02x%02x" % rgb
                print rgb,"\t", color

                col = Tkinter.Label(self.show_frame, width=1, height=2, bg=color)
                col.pack(side=Tkinter.LEFT, fill="both", expand=True)
            
        
        btn.bind("<Button-1>", type)
		
import urllib, cStringIO
URL = "https://images.unsplash.com/photo-1534832796130-37cfe118e925?ixlib=rb-0.3.5&s=9a2c452d95427cf5d73460ddf17e82b5&auto=format&fit=crop&w=1949&q=80"
file = cStringIO.StringIO(urllib.urlopen(URL).read())
ColorPallete(file)
Tkinter.mainloop()
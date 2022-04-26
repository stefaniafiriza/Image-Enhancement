import cv2
from skimage.filters import median, gaussian
from skimage.morphology import disk
from skimage.restoration import wiener, unsupervised_wiener
from tkinter import *
from tkinter import ttk
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog as fd
import numpy as np
from scipy.signal import convolve2d
from skimage.util import img_as_float
from matplotlib import pyplot as plt

root = tk.Tk()

class App:

    def __init__(self, root):
        
        frm = tk.Frame(root)
        frm.grid()
        self.label1 = tk.Label(root).grid(column=0, row=0)
        self.label2 = tk.Label(root).grid(column=5, row=0)
        self.label3 = tk.Label(root).grid(column=4, row=5)
        self.open_button = ttk.Button(root, text='Open a File', command=self.select_file).grid(column=4, row=2)

        #Menu drop-down
        options_drop_down = ['Median-Filtering', 'Gaussian-Filtering', 'Winer-Filtering', 'Histogram Equalization', 'Sharpening']
        clicked = StringVar(root)
        clicked.set('Choose a filter')
        self.drop_down = tk.OptionMenu(root , clicked , *options_drop_down, command=self.callback).grid(column=4, row=4)
        

    #select img to apply filter
    def select_file(self):
        global img
        global filename
        filetypes = (
            ('All files', '*.*'),
            ('image-png', '*.png'),
            ('image-jpeg', '*.jpeg')
        )

        filename = fd.askopenfilename(
            title='Open a file',
            initialdir='/Pictures',
            filetypes=filetypes)

        img = ImageTk.PhotoImage(Image.open(filename))
        self.image_label = tk.Label(root, image=img).grid(column=4, row=6)

        
    
    #select filter and apply to image
    def callback(self, selection):
        
        image_with_noice = cv2.imread(filename, 0)

        if selection == 'Median-Filtering':
            img_after_filtering = median(image_with_noice, disk(3), mode='constant', cval=0.0)
            cv2.imshow("After median filtering", img_after_filtering)
        elif selection == 'Gaussian-Filtering':
            img_after_filtering = gaussian(image_with_noice, sigma=1, mode='constant', cval=0.0)
            cv2.imshow("After gaussian filtering", img_after_filtering)
        elif selection == 'Winer-Filtering':
            img = img_as_float(image_with_noice)
            psf = np.ones((3, 3))
            img = convolve2d(img, psf, 'same')
            img += 0.1 * img.std() * np.random.standard_normal(img.shape)
            img_after_filtering = wiener(img, psf, 100)
            cv2.imshow("After winer filtering", img_after_filtering)
        elif selection == 'Histogram Equalization':
            hist,bins = np.histogram(image_with_noice.flatten(),256,[0,256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * float(hist.max()) / cdf.max()
            plt.plot(cdf_normalized, color = 'b')
            plt.hist(image_with_noice.flatten(),256,[0,256], color = 'r')
            plt.xlim([0,256])
            plt.legend(('cdf','histogram'), loc = 'upper left')
            plt.show()

            img_after_filtering = cv2.equalizeHist(image_with_noice)
            cv2.imshow("After Histogram Equalization", img_after_filtering)

            hist,bins = np.histogram(img_after_filtering.flatten(),256,[0,256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * float(hist.max()) / cdf.max()
            plt.plot(cdf_normalized, color = 'b')
            plt.hist(img_after_filtering.flatten(),256,[0,256], color = 'r')
            plt.xlim([0,256])
            plt.legend(('cdf','histogram'), loc = 'upper left')
            plt.show()
        elif selection == 'Sharpening':
            image_with_noice = cv2.imread(filename)
            kernel = np.array([[0, -1, 0], [-1, 5,-1],[0, -1, 0]])
            image_sharp = cv2.filter2D(src=image_with_noice, ddepth=-1, kernel=kernel)
            cv2.imshow('Image Sharpened', image_sharp)



a = App(root)

root.title('Image Enhacement')
root.mainloop()
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
window=tk.Tk()
window.title(" Mini Project")
window.resizable(0,0)

load1 = Image.open("res/hdr.png")
photo1 = ImageTk.PhotoImage(load1)

header = tk.Button(window, image=photo1)
header.place(x=5,y=0)

canvas1  = Canvas(window, width=500, height=300, bg='ivory')
canvas1.place(x=5, y=120)

l1=tk.Label(canvas1,text="Digit",font=('Algerian',20))
l1.place(x=5,y=0)

t1=tk.Entry(canvas1,width=20, border=5)
t1.place(x=150, y=5)

def screen_capture():
    import pyscreenshot as ImageGrab
    import time
    import os
    os.startfile("C:/ProgramData/Microsoft/Windows/Start Menu/Programs/Accessories/Paint")
    s1=t1.get()
    os.chdir("C:/Users/harshit/Desktop/Mini project/res/captured_images")
    os.mkdir(s1)
    os.chdir("C:/Users/harshit/Desktop/Mini project/res")

    images_folder="captured_images/"+s1+"/"
    time.sleep(15)
    for i in range(0,5):
        time.sleep(8)
        im=ImageGrab.grab(bbox=(60,170,400,550)) #x1,y1,x2,y2
        print("saved......",i)
        im.save(images_folder+str(i)+'.png')
        print("clear screen now and redraw now........")
    messagebox.showinfo("Result","Capturing screen is completed!!")
    
b1=tk.Button(canvas1,text="1. Open paint and capture the screen", font=('Algerian',15),bg="orange",fg="black",command=screen_capture)
b1.place(x=5, y=50)

def generate_dataset():
    import cv2
    import csv
    import glob

    header  =["label"]
    for i in range(0,784):
        header.append("pixel"+str(i))
    with open('res/dataset.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    for label in range(10):
        dirList = glob.glob("res/captured_images/"+str(label)+"/*.png")

        for img_path in dirList:
            im= cv2.imread(img_path)
            im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            im_gray = cv2.GaussianBlur(im_gray,(15,15), 0)
            roi= cv2.resize(im_gray,(28,28), interpolation=cv2.INTER_AREA)

            data=[]
            data.append(label)
            rows, cols = roi.shape

            ## Add pixel one by one into data array
            for i in range(rows):
                for j in range(cols):
                    k =roi[i,j]
                    if k>100:
                        k=1
                    else:
                        k=0
                    data.append(k)
            with open('dataset.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(data)
    messagebox.showinfo("Result","Generating dataset is completed!!")
    
b2=tk.Button(canvas1,text="2. Generate dataset", font=('Algerian',15),bg="pink",fg="blue",command=generate_dataset)
b2.place(x=5, y=100)

def train_save_accuracy():
    import pandas as pd
    from sklearn.utils import shuffle
    data  =pd.read_csv('res/dataset.csv')
    data=shuffle(data)
    X = data.drop(["label"],axis=1)
    Y= data["label"]
    from sklearn.model_selection import train_test_split
    train_x,test_x,train_y,test_y = train_test_split(X,Y, test_size = 0.2)
    import joblib
    from sklearn.svm import SVC
    classifier=SVC(kernel="linear", random_state=6)
    classifier.fit(train_x,train_y)
    joblib.dump(classifier, "res/model/digit_recognizer")
    from sklearn import metrics
    prediction=classifier.predict(test_x)
    acc=metrics.accuracy_score(prediction, test_y)
    messagebox.showinfo("Result",f"Your accuracy is {acc}")
    
b3=tk.Button(canvas1,text="3. Train the model and calculate accuracy", font=('Algerian',15),bg="green",fg="white",command=train_save_accuracy)
b3.place(x=5, y=150)

def prediction():
    import joblib
    import cv2
    import numpy as np #pip install numpy
    import time
    import pyscreenshot as ImageGrab
   
    model=joblib.load("res/model/digit_recognizer")
    
    img=ImageGrab.grab(bbox=(130,500,500,700))
    img.save("res/paint.png")
    
    im = cv2.imread("res/paint.png")
    load = Image.open("res/paint.png")
    load = load.resize((280,280))
    photo = ImageTk.PhotoImage(load)
    
    #Labels can be text or images
    img = Label(canvas3, image=photo, width=280, height=280)
    img.image=photo
    img.place(x=0,y=0)
    
    im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    im_gray  =cv2.GaussianBlur(im_gray, (15,15), 0)

    #Threshold the image
    ret, im_th = cv2.threshold(im_gray,100, 255, cv2.THRESH_BINARY)
    roi = cv2.resize(im_th, (28,28), interpolation  =cv2.INTER_AREA)

    rows,cols=roi.shape

    X = []

    ## Add pixel one by one into data array
    for i in range(rows):
        for j in range(cols):
            k = roi[i,j]
            if k>100:
                k=1
            else:
                k=0
            X.append(k)

    predictions  =model.predict([X])
    print("Prediction: ", predictions[0])
      
    a1 = tk.Label(canvas3, text="Prediction= ", font=("Algerian",20))
    a1.place(x=5, y=350)
    
    b1 = tk.Label(canvas3, text=predictions[0], font=("Algerian",20))
    b1.place(x=200, y=350)

   
    
    
        
b4=tk.Button(canvas1,text="4. prediction", font=('Algerian',15),bg="yellow",fg="red",command=prediction)
b4.place(x=5, y=200)
def liveprediction():
    from tensorflowTesting import testing
    import tensorflow as tf
    from keras.models import load_model
    import cv2
    import numpy as np
    import os

    from PIL import ImageTk, Image, ImageDraw
    import PIL
    import tkinter as ti

    classes=[0,1,2,3,4,5,6,7,8,9]
    width = 500
    height = 500
    center = height//2
    white = (255, 255, 255)
    green = (0,128,0)

    def paint(event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        cv.create_oval(x1, y1, x2, y2, fill="black",width=40)
        draw.line([x1, y1, x2, y2],fill="black",width=40)
    def model():
        filename = "image.png"
        image1.save(filename)
        pred=testing()
        print('argmax',np.argmax(pred[0]),'\n',
              pred[0][np.argmax(pred[0])],'\n',classes[np.argmax(pred[0])])
        txt.insert(ti.INSERT,"{}\nAccuracy: {}%".format(classes[np.argmax(pred[0])],round(pred[0][np.argmax(pred[0])]*100,3)))
        
    def clear():
        cv.delete('all')
        draw.rectangle((0, 0, 500, 500), fill=(255, 255, 255, 0))
        txt.delete('1.0', END)

    root =Toplevel(window)

    root.resizable(0,0)
    cv = Canvas(root, width=width, height=height, bg='white')
    cv.pack()

    image1 = PIL.Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(image1)

    txt=ti.Text(root,bd=3,exportselection=0,bg='WHITE',font='Helvetica',
                padx=10,pady=10,height=5,width=20)

    cv.pack(expand=YES, fill=BOTH)
    cv.bind("<B1-Motion>", paint)

    btnModel=Button(root,text="Predict",font=('Algerian',15),bg="black",fg="red",command=model)
    btnClear=Button(root,text="clear",font=('Algerian',15),bg="black",fg="pink",command=clear)
    btnModel.pack()
    btnClear.pack()
    txt.pack()
    root.title('live prediction')
    root.mainloop()

    
b5=tk.Button(canvas1,text="5. live prediction", font=('Algerian',15),bg="pink",fg="brown",command=liveprediction)
b5.place(x=5, y=250)

canvas2 = Canvas(window, width=500, height=425, bg='black')
canvas2.place(x=5, y=430)

def activate_paint(e):
    global lastx, lasty
    canvas2.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y
    
def paint(e):
    global lastx, lasty
    x,y = e.x, e.y
    canvas2.create_line((lastx,lasty,x,y), width=40, fill="white")
    lastx, lasty = x,y

canvas2.bind('<1>', activate_paint)    
   
def clear():
    canvas2.delete("all")
    
btn = tk.Button(canvas2, text="clear", fg="white", bg="green", command=clear)
btn.place(x=0,y=0)

canvas3 = Canvas(window, width=280, height=530, bg="green")
canvas3.place(x=515, y=120)

window.geometry("800x680")
window.mainloop()

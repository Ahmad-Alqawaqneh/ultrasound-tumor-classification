import tkinter as tk
from tkinter import filedialog,font
from PIL import Image, ImageTk
from skimage import io, transform
import pickle 
from tkinter import messagebox
import warnings
warnings.filterwarnings("ignore")

model=pickle.load(open('trained_model.pkl','rb'))

def load_preprocess_image(target_size, img_name):
    img1 = './data/img/' + img_name + '.png'
    img = io.imread(img1)
    img = transform.resize(img, target_size)
    return img

def predict():
    img_name = str(data_entry.get())  
    target_size = (100, 100)
    img = load_preprocess_image(target_size, img_name)
    prediction = model.predict([img.flatten()])[0]  
    value = f'Type of tumor : {prediction}' 
    print(value)
    messagebox.showinfo(title='Prediction', message=value)


user_info_frame = tk.Tk()
user_info_frame.title('Ultra Sound')

img = ImageTk.PhotoImage(Image.open('16.jpeg'))
my_label = tk.Label(image=img)
my_label.grid(column=0, row=0)

user_info_frame['background'] = 'light blue'
frame = tk.Frame(user_info_frame, bg="light blue")
frame.grid()

data_label = tk.LabelFrame(frame, text='Image number', bg="#d54161",fg="white")
data_label.grid(row=3, column=0)
bold_font = font.Font(data_label, data_label.cget("font"))
bold_font.configure(weight="bold")
data_label['font'] = bold_font
data_entry = tk.Entry(data_label)
data_entry.grid(row=4, column=0)
button = tk.Button(frame, text='Enter data', command=predict, bg='#063970', fg="white")
button.grid(row=6, column=0, sticky='news', padx=10, pady=10)

user_info_frame.mainloop()

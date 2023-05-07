#importing required modules
import os
import tkinter
import customtkinter
from PIL import ImageTk,Image
import cv2


customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("dark-blue")  # Themes: blue (default), dark-blue, green


app = customtkinter.CTk()  #creating cutstom tkinter window
app.geometry("12500x800")
app.maxsize(1250,800)
app.title('G-tech Login')
app.iconbitmap('icon.ico')

cap = cv2.VideoCapture(0)


def button_function():
    app.destroy()            # destroy current window and creating new one 
    w = customtkinter.CTk()  
    w.geometry("1280x720")
    update = False 
    w.maxsize(1250,725)
    w.minsize(1250,725)
    w.title('G-tech Home')
    w.iconbitmap('icon.ico')

    label = customtkinter.CTkLabel(master=w, text='')
    label.pack(pady=20, padx=20)
    # l1=customtkinter.CTkLabel(master=w, text="Home Page",font=('Century Gothic',60))
    # l1.place(relx=0.5, rely=0.5,  anchor=tkinter.CENTER)

    # camera.place_forget()


    frame=customtkinter.CTkFrame(master=w, width=900, height=690, corner_radius=15,fg_color='#1F1E3F')
    frame.place(relx=0.625, rely=0.5, anchor=tkinter.CENTER)

    logo = ImageTk.PhotoImage(Image.open("icon.png"))
    logo_label = customtkinter.CTkLabel(master=frame, image=logo, text="")
    logo_label.place(relx=0.009, rely=0.009, anchor=tkinter.NW)

    camera = tkinter.Label(master=frame, width=500, height=390, text='',borderwidth=2, relief="solid")
    nocamera = customtkinter.CTkFrame(master=frame, width=500, height=390, corner_radius=15,fg_color='#2E2B55', border_color='#F5D115', border_width=2)

    nocamera.place(relx=0.5, rely=0.4, anchor=tkinter.CENTER)

    nocameralabel = customtkinter.CTkLabel(master=nocamera, text="Launch Your Experience", font=('Century Gothic', 30, 'bold'))
    nocameralabel.place(relx=0.5, rely=0.5,  anchor=tkinter.CENTER)
    # camera = customtkinter.CTkFrame(master=frame, width=250, height=190, )
    
    # Center the text horizontally

    def show_frame():
        camera.place(relx=0.5, rely=0.4, anchor=tkinter.CENTER)
        nocamera.place_forget()
        update = True

    def close():
        camera.place_forget()
        nocamera.place(relx=0.5, rely=0.4,  anchor=tkinter.CENTER)
        # w.destroy()

    buttonconn = customtkinter.CTkButton(master=frame, text="Launch ", hover_color='#FF4500',font=('Helvetica', 16, 'bold'), border_color='#F5D115', border_width=2, command=show_frame )
    buttonconn.configure(width=300, height=50, corner_radius=6, fg_color='#29284E', text_color='#8465C7' , hover_color='#362067')
  # Prevent the label from shrinking
    buttonconn.place(relx=0.5, rely=0.8, anchor=tkinter.CENTER)

    buttondeconn = customtkinter.CTkButton(master=frame, text="Close ", hover_color='#FF4500',font=('Helvetica', 16, 'bold'), border_color='#F5D115', border_width=2, command=close )
    buttondeconn.configure(width=300, height=50, corner_radius=6, fg_color='#29284E', text_color='#8465C7' , hover_color='#362067')
    # Prevent the label from shrinking
    buttondeconn.place(relx=0.5, rely=0.9, anchor=tkinter.CENTER)


    sidebar = customtkinter.CTkFrame(master=w, width=200, height=670, fg_color='#1F1E3F',  corner_radius=15)
    sidebar.place(relx=0.01, rely=0.023, anchor=tkinter.NW)
    # sidebar.pack(side=tkinter.LEFT)

    # Add widgets to the sidebar
    label1 = customtkinter.CTkLabel(master=sidebar, text="AirWavecontrol", font=('Century Gothic', 15, 'bold'))
    label1.pack(pady=50, padx=45)

    button1 = customtkinter.CTkButton(master=sidebar, text="MS Power Point", hover_color='#FF4500',font=('Helvetica', 16, 'bold') )
    button1.configure(width=300, height=50, corner_radius=6, fg_color='#2E2B55', text_color='orange' , hover_color='#FF4500', )

    button1.pack(pady=10, padx=0)

    button2 = customtkinter.CTkButton(master=sidebar, text="Netflix", font=('Helvetica', 16, 'bold'), state=tkinter.DISABLED )
    button2.configure(width=300, height=50, corner_radius=6, fg_color='#2E2B55', text_color='red' , hover_color='#2E2B55' )
    button2.pack(pady=5, padx=0)

    button3 = customtkinter.CTkButton(master=sidebar, text="spotify", font=('Helvetica', 16, 'bold'), state=tkinter.DISABLED )
    button3.configure(width=300, height=50, corner_radius=6, fg_color='#2E2B55', text_color='red' , hover_color='#2E2B55' )
    button3.pack(pady=5, padx=0)

    button4 = customtkinter.CTkButton(master=sidebar, text="keynote", font=('Helvetica', 16, 'bold'), state=tkinter.DISABLED )
    button4.configure(width=300, height=50, corner_radius=6, fg_color='#2E2B55', text_color='red' , hover_color='#2E2B55' )
    button4.pack(pady=5, padx=0)

    button5 = customtkinter.CTkButton(master=sidebar, text="Zoom", font=('Helvetica', 16, 'bold'), state=tkinter.DISABLED )
    button5.configure(width=300, height=50, corner_radius=6, fg_color='#2E2B55', text_color='red' , hover_color='#2E2B55' )
    button5.pack(pady=5, padx=0)

    button6 = customtkinter.CTkButton(master=sidebar, text="Windows Os", font=('Helvetica', 16, 'bold'), state=tkinter.DISABLED )
    button6.configure(width=300, height=50, corner_radius=6, fg_color='#2E2B55', text_color='red' , hover_color='#2E2B55' )
    button6.pack(pady=5, padx=0)

    button7 = customtkinter.CTkButton(master=sidebar, text="Web Browser", font=('Helvetica', 16, 'bold'), state=tkinter.DISABLED )
    button7.configure(width=300, height=50, corner_radius=6, fg_color='#2E2B55', text_color='red' , hover_color='#2E2B55' )
    button7.pack(pady=5, padx=0)

    button8 = customtkinter.CTkButton(master=sidebar, text="Virtual env", font=('Helvetica', 16, 'bold'), state=tkinter.DISABLED )
    button8.configure(width=300, height=50, corner_radius=6, fg_color='#2E2B55', text_color='red' , hover_color='#2E2B55' )
    button8.pack(pady=5, padx=0)

    # Load the image files for the logos
    instagram_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(os.path.dirname(__file__), './ig.png')), size=(30, 30))
    twitter_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(os.path.dirname(__file__), './twit.png')), size=(30, 30))
    gmail_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(os.path.dirname(__file__), './fb.png')), size=(30, 30))

    # Create labels for each logo
    instagram_label = customtkinter.CTkLabel(master=sidebar, image=instagram_image, bg_color='transparent', text='')
    twitter_label = customtkinter.CTkLabel(master=sidebar, image=twitter_image, bg_color='transparent', text='')
    gmail_label = customtkinter.CTkLabel(master=sidebar, image=gmail_image, bg_color='transparent', text='')

    # Pack the labels horizontally
    instagram_label.pack(side=tkinter.LEFT, padx=20, pady=20)
    twitter_label.pack(side=tkinter.LEFT, padx=20,)
    gmail_label.pack(side=tkinter.LEFT, padx=(20,0))

    name = customtkinter.CTkLabel(master = sidebar,text="G-Tech",font=('Century Gothic',15, 'bold') )
    name.pack(pady=(15,10), padx=(1,5))
    # l1=customtkinter.CTkLabel(master=w, ))

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 390)


    def update_video():
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        camera.imgtk = imgtk
        camera.configure(image=imgtk)
        camera.after(1, update_video) # Schedule the next update

    update_video()
    w.mainloop()
    


img1=ImageTk.PhotoImage(Image.open("pattern.jpg"))
l1=customtkinter.CTkLabel(master=app,image=img1)
l1.pack()

#creating custom frame
frame=customtkinter.CTkFrame(master=l1, width=320, height=360, corner_radius=15)
frame.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

l2=customtkinter.CTkLabel(master=frame, text="Log into your Account",font=('Century Gothic',20))
l2.place(x=50, y=45)

entry1=customtkinter.CTkEntry(master=frame, width=220, placeholder_text='Username')
entry1.place(x=50, y=110)

entry2=customtkinter.CTkEntry(master=frame, width=220, placeholder_text='Password', show="*")
entry2.place(x=50, y=165)

l3=customtkinter.CTkLabel(master=frame, text="Forget password?",font=('Century Gothic',12))
l3.place(x=155,y=195)

#Create custom button
button1 = customtkinter.CTkButton(master=frame, width=220, text="Login", command=button_function, corner_radius=6)
button1.place(x=50, y=240)


img2=customtkinter.CTkImage(Image.open("Google__G__Logo.svg.webp").resize((20,20), Image.ANTIALIAS))
img3=customtkinter.CTkImage(Image.open("124010.png").resize((20,20), Image.ANTIALIAS))
button2= customtkinter.CTkButton(master=frame, image=img2, text="Google", width=100, height=20, compound="left", fg_color='white', text_color='black', hover_color='#AFAFAF')
button2.place(x=50, y=290)

button3= customtkinter.CTkButton(master=frame, image=img3, text="Facebook", width=100, height=20, compound="left", fg_color='white', text_color='black', hover_color='#AFAFAF')
button3.place(x=170, y=290)




# You can easily integrate authentication system 



app.mainloop()

import tkinter as tk
import os
import math
import numpy as np
import tkinter.font as tkFont
from tkinter import filedialog
from PIL import Image, ImageTk,ImageDraw
import threading

class bar():
    def __init__(self, Progess_len, Bar_Len=20, pad=6):
        self.set_Progess_len(Progess_len)
        self.Bar_Len = Bar_Len
        self.pad_container = ["♥", "♠", "♣", "♧", "*", "✿", "█"]
        self.pad = pad

    def set_Progess_len(self, Progress_len: int):
        self.Progress_len = Progress_len

    def prin(self, complet_len, verbose=False):
        now_len = complet_len if complet_len <= self.Progress_len else self.Progress_len
        rate = now_len / self.Progress_len
        pad_num = math.floor(rate * self.Bar_Len)
        tail = " complete: %.1f" % (rate * 100) + "%"
        bar = "[" + (self.pad_container[self.pad]) * pad_num + " " * (self.Bar_Len - pad_num) + "]" + tail
        if verbose:
            print("\r" + bar, end="")
        return bar


class root():
    def __init__(self):
        self.default_size()
        self.root_init()
        self.Menu_init()
        self.frame_init()
        self.Canvas_init()

    def default_size(self):
        self.size = {}
        self.size["root"] = "1200x950+600+300"
        self.size["root_frame"] = ["900", "500"]
        self.size["root_Canvas"] = ["880", "480"]

    def root_init(self):
        self.root = tk.Tk()
        self.root.title("paper image by 3312.anan and 3312.xiaolei")
        self.root.geometry(self.size["root"])
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.resizable(0, 0)
        try:
            self.root.iconphoto(False, tk.PhotoImage(file='./icon.png'))
        except Exception as e:
            print(e)

    #     self.root.bind('<Return>', self.enter_press)
    # def enter_press(self,event):
    #     pass
    def font(self, name="Fixdsys", size=10, weight_=0):
        weight_ = ["normal", "roman", "bold", "italic"][weight_]
        return tkFont.Font(family='Fixdsys', size=size, weight=weight_)

    def frame_init(self):
        self.root_frame = tk.Frame(self.root)
        self.root_frame["height"] = self.size["root_frame"][0]
        self.root_frame["width"] = self.size["root_frame"][1]
        self.root_frame.grid(row=0, column=0, rowspan=2, sticky="ns")
        self.vice_frame = tk.Frame(self.root, height=800, width=700)
        self.vice_frame.grid(row=0, column=1, sticky="ns")
        self.verbose_frame = tk.Frame(self.root, height=100, width=700)
        self.verbose_frame.grid(row=1, column=1, sticky="we")
        self.verbose_init()

    def verbose_init(self):
        self.verbose = {"d": None, "x": None, "y": None, "derection": None, "ok_b": None}
        self.x_start_tk = tk.StringVar()
        self.y_start_tk = tk.StringVar()
        self.d_tk = tk.StringVar()
        self.derection_tk = tk.StringVar()
        self.verbose_tk = tk.StringVar()
        x_l = tk.Label(self.verbose_frame, text="X:", font=self.font(size=13))
        self.verbose["x"] = tk.Entry(self.verbose_frame, width=5, font=self.font(size=13),
                                     textvariable=self.x_start_tk, )
        # validate="all",validatecommand=self.input_callback)
        y_l = tk.Label(self.verbose_frame, text="Y:", font=self.font(size=13))
        self.verbose["y"] = tk.Entry(self.verbose_frame, width=5, font=self.font(size=13),
                                     textvariable=self.y_start_tk, )
        # validate="all",validatecommand=self.input_callback)
        d_l = tk.Label(self.verbose_frame, text="d:", font=self.font(size=13))
        self.verbose["d"] = tk.Entry(self.verbose_frame, width=5, font=self.font(size=13), textvariable=self.d_tk, )
        # validate="all",validatecommand=self.input_callback)
        dr_l = tk.Label(self.verbose_frame, text="derection:", font=self.font(size=13))
        self.verbose["derection"] = tk.Entry(self.verbose_frame, width=5, font=self.font(size=13),
                                             textvariable=self.derection_tk, )
        # validate="all",validatecommand=self.input_callback)
        self.verbose["ok_b"] = tk.Button(self.verbose_frame, text="ok", command=self.input_callback,
                                         font=self.font(size=13))
        self.verbose_frame.columnconfigure(8, minsize=20)
        x_l.grid(row=0, column=0, sticky="w")
        self.verbose_frame.columnconfigure(2, minsize=20)
        self.verbose["x"].grid(row=0, column=1, sticky="w")
        y_l.grid(row=0, column=3, sticky="w")
        self.verbose["y"].grid(row=0, column=4, sticky="w")
        self.verbose_frame.columnconfigure(5, minsize=20)
        d_l.grid(row=0, column=6, sticky="w")
        self.verbose["d"].grid(row=0, column=7, sticky="w")
        self.verbose_frame.columnconfigure(8, minsize=20)
        dr_l.grid(row=0, column=9, sticky="w")
        self.verbose["derection"].grid(row=0, column=10, sticky="w")
        self.verbose_frame.columnconfigure(11, minsize=20)
        self.verbose["ok_b"].grid(row=0, column=12, sticky="w")
        tk.Label(self.verbose_frame,width=50,height=2, textvariable=self.verbose_tk, font=self.font(size=10)).grid(row=1, column=0,columnspan=12,
                                                                                                      sticky="w")
        for n in self.verbose:
            self.verbose[n]["state"] = "disabled"

    def activate_verbose(self):
        for n in self.verbose:
            self.verbose[n]["state"] = "normal"

    def input_callback(self):
        if self.d_tk.get() == "" or self.x_start_tk.get() == "" or self.y_start_tk.get() == "":
            return
        if self.derection_tk.get() not in ["ws", "es", "wn", "en"]:
            self.derection_tk.set("ws")
        self.x_start = int(self.x_start_tk.get())
        self.y_start = int(self.y_start_tk.get())
        d = int(self.d_tk.get())
        if self.derection_tk.get()[0] == "w":
            self.x_end = self.x_start - d
        else:
            self.x_end = self.x_start + d
        if self.derection_tk.get()[1] == "n":
            self.y_end = self.x_start - d
        else:
            self.y_end = self.y_start + d
        self.draw_rectanggle()
        self.vice_canvas_open_img()

    def v_frame_label_init(self):
        self.vframes = []
        self.vcanvas = {"master": [], "img_wdget": [], "psnr_wdget": [], "label_n": []}
        self.PSNR = {}
        self.SSIM = {}
        for i in range(len(self.label)):
            row = i // 3 + 1
            column = i % 3 + 1
            f = tk.Frame(self.vice_frame, height=280, width=250)
            self.vice_frame.rowconfigure(row, weight=1)
            self.vice_frame.columnconfigure(column, weight=1)
            f.grid(row=row, column=column, sticky="nesw")
            self.vframes.append(f)

        for label_name, frame in zip(self.label, self.vframes):
            l = tk.Label(frame, text=label_name, font=self.font("宋体"))

            def radio_click():
                if self.root_dir_tk.get() ==self.Root_Dir:
                    return
                self.Root_Dir=self.root_dir_tk.get()
                self.root_Canvas_open_img()
                self.vice_canvas_open_img()
                self.draw_rectanggle()
                psnr_thread = threading.Thread(target=self.show_psnr)
                psnr_thread.setDaemon(True)
                psnr_thread.start()


            R = tk.Radiobutton(frame, text="主图", variable=self.root_dir_tk, value=label_name, command=radio_click)
            self.PSNR[label_name] = 0
            self.SSIM[label_name] = 0
            # R.pack(side="top")
            # l.pack(side="top",fill="x")
            R.grid(row=0, column=0, sticky="n")
            l.grid(row=1, column=0, sticky="n")

            c = tk.Canvas(frame, height=220, width=220)
            c.grid(row=2, column=0, sticky="n")
            l = tk.Label(frame, text="PSNR:{:.2f} , SSIM:{:.2f}".format(self.PSNR[label_name], self.SSIM[label_name]),
                         font=self.font("宋体"))
            l.grid(row=3, column=0, sticky="n")
            self.vcanvas["master"].append(c)
            self.vcanvas["img_wdget"].append(None)
            self.vcanvas["label_n"].append(label_name)
            self.vcanvas["psnr_wdget"].append(l)

    def loop(self):
        self.x_start = 0
        self.y_start = 0
        self.root.mainloop()

    def Canvas_init(self):
        self.root_Canvas = tk.Canvas(self.root_frame, bg="green")
        self.root_Canvas["height"] = self.size["root_Canvas"][0]
        self.root_Canvas["width"] = self.size["root_Canvas"][1]
        Vbar = tk.Scrollbar(self.root_frame, orient="vertical")
        Vbar.config(command=self.root_Canvas.yview)
        Hbar = tk.Scrollbar(self.root_frame, orient="horizontal")
        Hbar.config(command=self.root_Canvas.xview)
        self.root_Canvas.config(yscrollcommand=Vbar.set, xscrollcommand=Hbar.set)
        # Vbar.pack(side="right", fill="y")
        # self.root_Canvas.pack(fill="both")
        # Hbar.pack(side="bottom", fill="x")
        self.root_Canvas.grid(row=0, column=0)
        Vbar.grid(row=0, column=1, sticky="wns", rowspan=2)
        Hbar.grid(row=1, column=0, sticky="new")

    def Get_dir_path(self):
        path = filedialog.askdirectory()
        return path

    def getfile(self, path, is_path=True, type="dir"):
        filelist = []
        filelist_ = os.listdir(path)
        if is_path:
            for i in range(len(filelist_)):
                filelist_[i] = os.path.join(path, filelist_[i])
        if type == "dir":
            for p in filelist_:
                if "." not in os.path.splitext(p)[1]:
                    filelist.append(p)
        elif type == "pic":
            for p in filelist_:
                if os.path.splitext(p)[1] in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ]:
                    filelist.append(p)

        return filelist

    def Get_file_path(self):
        """
        tkinter.filedialog.asksaveasfilename():选择以什么文件名保存，返回文件名
        tkinter.filedialog.asksaveasfile():选择以什么文件保存，创建文件并返回文件流对象
        tkinter.filedialog.askopenfilename():选择打开什么文件，返回文件名
        tkinter.filedialog.askopenfile():选择打开什么文件，返回IO流对象
        tkinter.filedialog.askdirectory():选择目录，返回目录名
        tkinter.filedialog.askopenfilenames():选择打开多个文件，以元组形式返回多个文件名
        tkinter.filedialog.askopenfiles():选择打开多个文件，以列表形式返回多个IO流对象
        """
        path = filedialog.askopenfilename()
        return path

    def canvas_show_img(self, widget, tk_image):
        return widget.create_image(0, 0, image=tk_image, anchor="nw")

    def Tk_image(self, img):
        return ImageTk.PhotoImage(image=img)

    def on_click(self, event):
        x = event.x
        y = event.y
        self.x_start = self.root_Canvas.canvasx(x)
        self.y_start = self.root_Canvas.canvasy(y)

    def click_motion(self, event):
        x = event.x
        y = event.y
        self.x_end = self.root_Canvas.canvasx(x)
        y_ = self.root_Canvas.canvasy(y)
        d = abs(self.x_end - self.x_start)
        self.y_end = self.y_start + d if y_ > self.y_start else self.y_start - d
        self.draw_rectanggle()

    def x_y_check(self):
        if self.canvas_rectanggle is None:
            return
        if self.y_end > self.imgh:
            self.y_end = self.imgh - 1
        if self.y_start > self.imgh:
            self.y_start = self.imgh - 1
        if self.y_end < 0:
            self.y_end = 1
        if self.y_start < 0:
            self.y_start = 1
        if self.x_end > self.imgw:
            self.x_end = self.imgw - 1
        if self.x_start > self.imgw:
            self.x_start = self.imgw - 1
        if self.x_end < 0:
            self.x_end = 1
        if self.x_start < 0:
            self.x_start = 1
        derection_tk = ["w", "n"]
        self.int_x_start = int(self.x_start)
        self.int_x_end = int(self.x_end)
        self.int_y_start = int(self.y_start)
        self.int_y_end = int(self.y_end)
        if self.x_start < self.x_end:
            derection_tk[0] = "e"
        else:
            derection_tk[0] = "w"
        if self.y_start < self.y_end:
            derection_tk[1] = "s"
        else:
            derection_tk[1] = "n"

        self.x_start_tk.set(str(self.int_x_start))
        self.y_start_tk.set(str(self.int_y_start))
        self.d_tk.set(str(abs(self.int_y_end - self.int_y_start)))
        self.derection_tk.set(derection_tk[0] + derection_tk[1])

    def draw_rectanggle(self):
        self.x_y_check()
        if self.x_start == 0:
            return
        self.canvas_clear(self.root_Canvas, self.canvas_rectanggle)
        self.canvas_rectanggle = self.root_Canvas.create_rectangle(self.x_start, self.y_start, self.x_end, self.y_end,
                                                                   outline="red",width=3)
        self.menu_dict["edit"].entryconfigure(0, state="normal")
        self.menu_dict["edit"].entryconfigure(1, state="normal")

    def canvas_clear(self, canvas_widget, img_widget):
        if img_widget is not None:
            canvas_widget.delete(img_widget)

    def next_pic(self):
        if self.root_Canvas_index >= (len(self.pic_list) - 1):
            return
        self.root_Canvas_index += 1
        self.vice_img_open()
        psnr_thread = threading.Thread(target=self.show_psnr)
        psnr_thread.setDaemon(True)
        psnr_thread.start()

        self.root_Canvas_open_img()
        if self.canvas_rectanggle is not None:
            self.vice_canvas_open_img()
            self.draw_rectanggle()

    def last_pic(self):
        if self.root_Canvas_index <= 0:
            return
        self.root_Canvas_index -= 1
        self.vice_img_open()

        psnr_thread = threading.Thread(target=self.show_psnr)
        psnr_thread.setDaemon(True)
        psnr_thread.start()

        self.root_Canvas_open_img()
        if self.canvas_rectanggle is not None:
            self.vice_canvas_open_img()
            self.draw_rectanggle()

    def Menu_init(self):
        self.Menu = tk.Menu(self.root)
        self.menu_dict = {}
        file = tk.Menu(self.Menu, tearoff=0)
        file.add_command(label="打开目录", command=self.file_sys_made, font=self.font())
        self.menu_dict["edit"] = tk.Menu(self.Menu, tearoff=0)
        self.menu_dict["edit"].add_command(label="清除", command=self.rectanggle_clear, font=self.font(), state="disable")
        self.menu_dict["edit"].add_command(label="保存", command=self.save, font=self.font(), state="disable")
        self.menu_dict["edit"].add_command(label="上一张", command=self.last_pic, font=self.font(), state="disable")
        self.menu_dict["edit"].add_command(label="下一张", command=self.next_pic, font=self.font(), state="disable")

        self.Menu.add_cascade(label="文件", menu=file, font=self.font())
        self.Menu.add_cascade(label="编辑", menu=self.menu_dict["edit"], font=self.font())
        self.root.config(menu=self.Menu)

    def rectanggle_clear(self):
        self.menu_dict["edit"].entryconfigure(1, state="disable")
        self.canvas_clear(self.root_Canvas, self.canvas_rectanggle)

    def file_sys_made(self):
        self.path = self.Get_dir_path()
        self.dir = self.getfile(self.path, False, type="dir")
        self.label = {}

        def sort_key(filename):
            file_name = filename[:-len(os.path.splitext(filename)[-1])]
            asc = 0
            for c in file_name[:7]:
                asc += ord(c)
            return asc

        for dir in self.dir:
            label_name = dir
            self.label[label_name] = self.getfile(os.path.join(self.path, dir), False, "pic")
            self.label[label_name].sort(key=sort_key)
        self.root_dir_tk = tk.StringVar()
        self.root_dir_tk.set(list(self.label)[0])
        self.Root_Dir=self.root_dir_tk.get()
        self.root_Canvas_index = 0
        self.root_canvas_img = None
        self.canvas_rectanggle = None
        self.v_frame_label_init()
        self.root_Canvas_open_img()

    def root_Canvas_open_img(self):
        self.activate_verbose()
        self.menu_dict["edit"].entryconfigure(2, state="normal")
        self.menu_dict["edit"].entryconfigure(3, state="normal")
        self.canvas_clear(self.root_Canvas, self.root_canvas_img)
        if self.root_canvas_img is None:
            self.vice_img_open()
            psnr_thread = threading.Thread(target=self.show_psnr)
            psnr_thread.setDaemon(True)
            psnr_thread.start()
        root_dir = self.Root_Dir
        self.pic_list = self.label[root_dir]  # 为了兼容不同文件夹下文件名可能不同
        path = os.path.join(self.path, root_dir, self.pic_list[self.root_Canvas_index])
        self.root_img = Image.open(path)
        self.imgw, self.imgh = self.root_img.size[0], self.root_img.size[1]
        if self.imgw < int(self.size["root_Canvas"][1]):
            self.root_Canvas["width"] = str(self.imgw)
        else:
            self.root_Canvas["width"] = self.size["root_Canvas"][1]
        if self.imgh < int(self.size["root_Canvas"][0]):
            self.root_Canvas["height"] = str(self.imgh)
        else:
            self.root_Canvas["height"] = self.size["root_Canvas"][0]
        self.root_tkimg = self.Tk_image(self.root_img)
        self.root_Canvas["scrollregion"] = (0, 0, self.imgw, self.imgh)
        self.root_canvas_img = self.canvas_show_img(self.root_Canvas, self.root_tkimg)
        self.root_Canvas.bind('<Button-1>', self.on_click)
        self.root_Canvas.bind('<B1-Motion>', self.click_motion)
        self.root_Canvas.bind('<ButtonRelease-1> ', self.vice_canvas_imgshow)

    def vice_canvas_imgshow(self, event):
        self.vice_canvas_open_img()

    def vice_canvas_open_img(self):
        self.x_y_check()
        self.vice_data = {"tkimg": [], "npimg": [], "label_n": []}
        for i, l in enumerate(self.label):
            img = self.vice_img["img"][i]
            npimg = np.array(img)
            try:
                npimg = npimg[self.int_y_start:self.int_y_end, self.int_x_start:self.int_x_end, :]
                img = Image.fromarray(npimg, mode="RGB")
            except:
                self.rectanggle_clear()
                return
            img = img.resize((220, 220), resample=Image.BICUBIC)
            tkimg = self.Tk_image(img)
            self.vice_data["tkimg"].append(tkimg)
            self.vice_data["npimg"].append(npimg)
            self.vice_data["label_n"].append(l)
        for i, vc in enumerate(self.vcanvas["master"]):
            self.canvas_clear(vc, self.vcanvas["img_wdget"][i])
            self.vcanvas["img_wdget"][i] = self.canvas_show_img(vc, self.vice_data["tkimg"][i])

    def vice_img_open(self):
        self.vice_img = {"img": [], "label_n": []}
        for l in self.label:
            pic_path = os.path.join(self.path, l, self.label[l][self.root_Canvas_index])
            img_ = Image.open(pic_path)
            self.vice_img["img"].append(img_)
            self.vice_img["label_n"].append(l)

    def show_psnr(self):
        # showPSNR

        root_Canvas_index = self.root_Canvas_index
        root_dir = self.Root_Dir
        root_label = self.Root_Dir
        root_npimg_index = self.vice_img["label_n"].index(root_label)
        root_npimg = np.array(self.vice_img["img"][root_npimg_index])
        ProgressBar = bar(len(self.vice_img["label_n"]), 10)
        for i, (img, label_name) in enumerate(zip(self.vice_img["img"], self.vice_img["label_n"])):
            self.verbose_tk.set(
                "PSNR cacing,root_dir is {}, pic_index is {},\nProgressBar：{}".format(root_label, root_Canvas_index,
                                                                                     ProgressBar.prin(i)))
            if root_Canvas_index != self.root_Canvas_index or root_dir != self.Root_Dir:
                self.verbose_tk.set("")
                return
            npimg = np.array(img)
            if label_name == root_label:
                self.PSNR[label_name] = 100
                self.SSIM[label_name] = 1
            else:
                if abs(npimg.shape[0] - root_npimg.shape[0]) > 3 or abs(npimg.shape[1] - root_npimg.shape[1]) > 3:
                    raise Exception(
                        "npimg.shape[0] {}-root_npimg.shape[0] {})>3 or abs(npimg.shape[1] {}-root_npimg.shape[1] {}".format(
                            npimg.shape[0], root_npimg.shape[0], npimg.shape[1], root_npimg.shape[1]))
                h = npimg.shape[0] if npimg.shape[0] < root_npimg.shape[0] else root_npimg.shape[0]
                w = npimg.shape[1] if npimg.shape[1] < root_npimg.shape[1] else root_npimg.shape[1]
                root_npimg = root_npimg[:h, :w, :]
                npimg = npimg[:h, :w, :]
                self.PSNR[label_name] = self.cac_psnr(root_npimg, npimg)
                self.SSIM[label_name] = self.cac_ssim(root_npimg, npimg)

            self.vcanvas["psnr_wdget"][i]["text"] = "PSNR:{:.2f} , SSIM:{:.2f}".format(self.PSNR[label_name],
                                                                                       self.SSIM[label_name])
            self.verbose_tk.set("")

    def save(self):
        ProgressBar=bar(len(self.vice_data["npimg"]))
        dirpath = "./resuls/" + self.pic_list[self.root_Canvas_index]
        log_path="./resuls/"+self.Root_Dir+".csv"

        def save_rootimg():#保存画布
            import copy
            root_img=copy.deepcopy(self.root_img)
            canvas=ImageDraw.Draw(root_img)
            canvas.rectangle((self.int_x_start,self.int_y_start,self.int_x_end,self.int_y_end),outline="red",width=5)
            root_img.save(os.path.join(dirpath, "root_img.png"))



        def write(path, *args):
            with open(path, "a") as log:
                for info in args:
                    log.writelines(info+",")
                log.writelines("\n")

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        head=["PSNR/SSIM"]
        info=[self.label[self.Root_Dir][self.root_Canvas_index]]
        for i,(l, npimg) in enumerate(zip(self.label, self.vice_data["npimg"])):
            self.verbose_tk.set("Saving, ProgessBar{}".format(ProgressBar.prin(i+1)))
            pic_name = self.label[l][self.root_Canvas_index]
            pic = Image.fromarray(npimg, mode="RGB")
            pic.save(os.path.join(dirpath, pic_name))
            psnr_ssim="{:.2f}/{:.2f}".format(self.PSNR[l],self.SSIM[l])
            head.append(l)
            info.append(psnr_ssim)
        if not os.path.exists(log_path):
            write(log_path,*head)
        write(log_path,*info)
        save_rootimg()
        self.verbose_tk.set("save complete")

    def cac_psnr(self, sr, hr, isY=True):
        import math
        import torch
        sr = torch.from_numpy(sr).permute(2, 0, 1).unsqueeze(dim=0).float()
        hr = torch.from_numpy(hr).permute(2, 0, 1).unsqueeze(dim=0).float()

        diff = (sr - hr) / 255
        # if isY:
        #     gray_coeffs = [65.738, 129.057, 25.064]
        #     convert = diff.new_tensor(gray_coeffs).view(1,3, 1, 1) / 256
        #     diff = diff.mul(convert).sum(dim=1)
        mse = diff.pow(2).mean()
        return -10 * math.log10(mse)

    def cac_ssim(self, img1, img2):

        import cv2
        import numpy as np
        def ssim(img1, img2):
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2
            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)
            kernel = cv2.getGaussianKernel(11, 1.5)
            window = np.outer(kernel, kernel.transpose())
            mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
            mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
            sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
            sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                    (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()

        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''
        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')

if __name__=="__main__":
    window = root()
    window.loop()

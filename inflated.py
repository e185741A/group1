'''画像の水増しを行うプログラム。
学習画像を垂直方向への反転、90度回転、270度回転、グレー化、ヒストグラムの変更を行うことで画像の水増しを実行する。
'''
import cv2, os, re,requests, time, bs4
from urllib.request import urlretrieve
from urllib import request as req
from urllib import error,parse
from PIL import Image
import numpy as np
import glob
import os.path

members = ["松本潤","二宮和也","相葉雅紀","大野智","櫻井翔"]
img_dir = "./arashi_image/"
cascade_file = "./haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_file)

def horizontal_flip(image, member):
    '''画像の水平方向への反転。
    Argments:
        image (PIL.JpegImagePlugin.JpegImageFile): arashi_imageファイルにある画像データ。
        member (str): メンバーの名前。
    Returns:
        なし。
    '''
    pil_img = image.transpose(Image.FLIP_LEFT_RIGHT)
    pil_img.save(img_dir +member + "/face/"+ "A"+ str(num)+".jpg")
    
def vertical_flip(image, member):
    '''画像の垂直方向への反転。
    Argments:
        image (PIL.JpegImagePlugin.JpegImageFile): arashi_imageファイルにある画像データ。
        member (str): メンバーの名前。
    Returns:
        なし。
    '''
    pil_img = image.transpose(Image.FLIP_TOP_BOTTOM)
    pil_img.save(img_dir +member + "/face/"+ "V"+ str(num)+".jpg")


def kaiten90(image, member):
    '''画像の90度反転。
    Argments:
        image (PIL.JpegImagePlugin.JpegImageFile): arashi_imageファイルにある画像データ。
        member (str): メンバーの名前。
    Returns:
        なし。
    '''
    img_rotate = image.rotate(90)
    img_rotate.save(img_dir +member + "/face/" + "B"+ str(num)+".jpg")

def kaiten270(image, member):
    '''画像の270度反転。
    Argments:
        image (PIL.JpegImagePlugin.JpegImageFile): arashi_imageファイルにある画像データ。
        member (str): メンバーの名前。
    Returns:
        なし。
    '''
    img_rotate = image.rotate(270)
    img_rotate.save(img_dir +member + "/face/" + "C"+ str(num)+".jpg")

def lighten(image, member):
    '''画像のヒストグラムの変更。
    Argments:
        image (numpy.ndarray): arashi_imageファイルにある画像データ。
        member (str): メンバーの名前。
    Returns:
        なし。
    '''
    image = image + 15
    pil_img = Image.fromarray(image)
    pil_img.save(img_dir +member + "/face/" + "E"+ str(num)+".jpg")
    
def gray(image, member):
    '''画像のグレー化。
    Argments:
        image (numpy.ndarray): arashi_imageファイルにある画像データ。
        member (str): メンバーの名前。
    Returns:
        なし。
    '''
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    pil_img = Image.fromarray(image)
    pil_img.save(img_dir +member + "/face/" + "D"+ str(num)+".jpg")
        
def bigger(image, member):
    '''画像の拡大。
    Argments:
        image (PIL.JpegImagePlugin.JpegImageFile): arashi_imageファイルにある画像データ。
        member (str): メンバーの名前。
    Returns:
        なし。
    '''
    height, width = image.shape[:2]
    size=(int(width*1.5) , int(height*1.5))
    bigimg = cv2.resize(image,size)
    pil_img = Image.fromarray(bigimg)
    pil_img.save(img_dir +member + "/face/" + "G"+ str(num)+".jpg")
 
def smaller(image, member):
    '''画像の縮小。
    Argments:
        image (PIL.JpegImagePlugin.JpegImageFile): arashi_imageファイルにある画像データ。
        member (str): メンバーの名前。
    Returns:
        なし。
    '''
    height, width = image.shape[:2]
    size=(int(width*0.5) , int(height*0.5))
    bigimg = cv2.resize(image,size)
    pil_img = Image.fromarray(bigimg)
    pil_img.save(img_dir +member + "/face/" + "G"+ str(num)+".jpg")
 
 
nums = np.arange(300)
for member in members:
    print(member+"の処理をしています")
    for num in nums:
        img_path = img_dir + member + "/face/" + str(num) + ".jpg"
        if os.path.exists(img_path):
            im = Image.open(img_path)
            im2 = np.array(Image.open(img_path))
            im3 = cv2.imread(img_path)
            vertical_flip(im, member)#画像の垂直方向への反転。
            kaiten90(im, member)#画像の90度反転。
            kaiten270(im, member)#画像の270度反転。
            gray(im3, member)#画像のグレー化。
            lighten(im2, member)#画像のヒストグラムの変更。

from matplotlib import pyplot as plt
import os, sys
from glob import glob
from matplotlib import pyplot

from PIL import Image
from PIL import EpsImagePlugin
import math

from PIL import ImageFilter
import numpy as np

from lxml import etree
import xml.etree.ElementTree as xml
import random
import cv2





def preSign():
    sign_data = glob('./Desktop/ETRI/DrawGen/data/standard/sign/*.png')
    # preprocessing (Sign)

    for s in sign_data:
        if 'DT_177' in s:
            img = cv2.imread(s)
            kernel = np.ones((3, 3), np.uint8)
            img = cv2.erode(img, kernel, iterations=1)

            cv2.imwrite(s, img)

        else:
            img = cv2.imread(s)
            kernel = np.ones((7, 7), np.uint8)
            img = cv2.erode(img, kernel, iterations=1)

            cv2.imwrite(s, img)


    size = 30, 30

    for s in sign_data:
        im = Image.open(s)
        im.thumbnail(size, Image.ANTIALIAS)
        im = im.convert("RGBA")
        datas = im.getdata()
        newData = []

        for item in datas:
            if item[0] > 200 and item[1] > 200 and item[2] > 200:
                newData.append((item[0], item[1], item[2], 0))
            else:
                newData.append(item)

            im.putdata(newData)
            im.save(s)



def ChangeColor(inputImg, r2, g2, b2):

    im = Image.open(inputImg)

    im = np.array(im)
    ret, im = cv2.threshold(im, 247, 255, cv2.THRESH_TOZERO)
    data = im

    imName = inputImg.split('sign')[1]
    imName = inputImg.split('\\')[1]
    imName = imName.split(".png")[0]

    r1, g1, b1 = 0, 0, 0  # Original value
    r2, g2, b2 = r2, g2, b2  # Value that we want to replace it with

    red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    data[:, :, :3][mask] = [r2, g2, b2]

    im = Image.fromarray(data)
    imDir = "./Desktop/ETRI/DrawGen/data/gen/sign/" + imName + '_r' + str(r2) + 'g' + str(g2) + 'b' + str(b2) + '.png'
    im.save(imDir)

    return imDir



def preDraw():
    draw_data = glob('./Desktop/ETRI/DrawGen/data/standard/draw/*.png')

    for i in draw_data:
        im = cv2.imread(i)
        ret, imGray = cv2.threshold(im,150,255, cv2.THRESH_BINARY_INV)

        gray = cv2.cvtColor(imGray, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(i, gray)


def genData(dr_fileName):
    # dr_fileName = 'C:/Users/USER/PycharmProjects/DrawGen/data/standard/draw/10f.png'

    l_img = cv2.imread(dr_fileName)
    h, w, d = l_img.shape

    sign_data = glob('./Desktop/ETRI/DrawGen/data/standard/sign/*.png')

    signList = []
    # set a color RGB
    r2, g2, b2 = 152,0,0

    # set an amoumnt of sign #
    amountList = []
    for a in range(10):
        amountList.append(random.randrange(0, 5))


    total = sum(amountList)

    new_OriFileName = 'GEN_' + str(int(random.random() * 100000))
    new_fileName = new_OriFileName + '.png'
    new_dirFileName = './Desktop/ETRI/DrawGen/data/gen' + new_fileName

    root = etree.Element("annotation")

    folder = etree.SubElement(root, "folder").text = 'dataset'
    filename = etree.SubElement(root, "filename").text = new_fileName
    path = etree.SubElement(root, "path").text = new_dirFileName

    source = etree.SubElement(root, "source")
    database = etree.SubElement(source, "database").text = 'Unknown'

    size = etree.SubElement(root, "size")
    width = etree.SubElement(size, "width").text = str(w)
    height = etree.SubElement(size, "height").text = str(h)
    depth = etree.SubElement(size, "depth").text = str(d)

    segmented = etree.SubElement(root, "segmented").text = '0'

    print(total) 
    print(new_dirFileName)  

    xmin = random.sample(range(50, w), total)
    ymin = random.sample(range(50, h), total)

    location = list(zip(xmin, ymin))

    for i, j in zip(sign_data, amountList):

        signDir = i
        clssName = signDir.split('\\sign')[1]
        clssName = clssName .split(clssName[0])[1]
        clssName = clssName.split('.png')[0]

        signList.append(clssName)

        imDir = ChangeColor(signDir, r2, g2, b2)

        s_img = cv2.imread(imDir, -1)
        alpha_s = s_img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for k in range(0, j):

            x1, y1 = location.pop()
            x2 = x1 + s_img.shape[1]
            y2 = y1 + s_img.shape[0]

            print(x1, x2, y1, y2)


            for c in range(0, 3):
                l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] + alpha_l * l_img[y1:y2, x1:x2, c])

            obj = etree.SubElement(root, "object")
            name = etree.SubElement(obj, "name").text = clssName
            poset = etree.SubElement(obj, "pose").text = 'Unspecified'
            truncated = etree.SubElement(obj, "truncated").text = '0'
            difficult = etree.SubElement(obj, "difficult").text = '0'

            bndbox = etree.SubElement(obj, "bndbox")
            xmin = etree.SubElement(bndbox, "xmin").text = str(x1)
            ymin = etree.SubElement(bndbox, "ymin").text = str(y1)
            xmax = etree.SubElement(bndbox, "xmax").text = str(x2)
            ymax = etree.SubElement(bndbox, "ymax").text = str(y2)


    cv2.imwrite(new_dirFileName, l_img)
    tree = etree.ElementTree(root)

    with open('./Desktop/ETRI/DrawGen/data/standard/draw/' + new_OriFileName + '.xml', "bw") as fh:
        tree.write(fh)


preSign()
#preDraw()



drawData = glob('./Desktop/ETRI/DrawGen/data/standard/draw/*')


for index, val in enumerate(drawData):
    print(index)
    try:
        genData(val)
    except:
        continue




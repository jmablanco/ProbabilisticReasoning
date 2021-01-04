import math
import numpy as np
from PIL import Image, ImageTk

def length(x):
    return math.sqrt(x[0]**2 + x[1]**2)

def angle_bw(x, y):
    normx = np.linalg.norm(x)
    normy = np.linalg.norm(y)
    if normx <= 1e-8 or normy <= 1e-8:
        return 0
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    # using cross-product formula
    return -math.degrees(math.asin((x[0] * y[1] - x[1] * y[0])/(length(x)*length(y))))
    # the dot-product formula, left here just for comparison (does not return angles in the desired range)
    # return math.degrees(math.acos((self.a * other.a + self.b * other.b)/(self.length()*other.length())))

def add_noise(x, std):
    return x + np.random.normal(0, std)

def load_image(path, scale):
    try:
        img = Image.open(path)
        new_width = int(img.width * float(scale))
        new_height = int(img.height * float(scale))
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        return img, ImageTk.PhotoImage(img)
    except IOError as e:
        print(e)
        sys.exit(1)

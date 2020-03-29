#download sudo aptitude install python-imaging

import PIL
from PIL import Image

basewidth = 1920
img = Image.open('/home/rukesh/Desktop/frontend/images/aa.png')
wpercent = (basewidth / float(img.size[0]))
hsize = int((float(img.size[1]) * float(wpercent)))
img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
img.save('/home/rukesh/Desktop/frontend/images/wo.png')


baseheight = 1080
img = Image.open('/home/rukesh/Desktop/frontend/images/wo.png')
hpercent = (baseheight / float(img.size[1]))
wsize = int((float(img.size[0]) * float(hpercent)))
img = img.resize((wsize, baseheight), PIL.Image.ANTIALIAS)
img.save('/home/rukesh/Desktop/frontend/images/wo.png')

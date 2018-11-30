from PIL import Image
from pytesseract import pytesseract

img = Image.open('/Users/brian/Storage/6th.png')
data = pytesseract.image_to_string(img, lang='chi_sim')
print(data)

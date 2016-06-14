import cv2
import zbar
from PIL import Image

def scanImage(image):
    scanner = zbar.ImageScanner()
    scanner.parse_config('enable')
    bwImg = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(bwImg)
    #pil_im.show()
    #pic = Image.open("/Users/susan/Desktop/qrcode.fox.png")
    pic2 = pil_im.convert("L")
    wid, hgt = pic2.size
    raw = pic2.tobytes()

    img = zbar.Image(wid, hgt, 'Y800', raw)
    result = scanner.scan(img)
    if result == 0:     
        print "Scan failed"
    else:
        for symbol in img:
            pass
    
        del(img)
        data = symbol.data.decode(u'utf-8')
        print "Data found:", data


cam = cv2.VideoCapture(0)

while True:
    res, frame = cam.read()
    cv2.imshow("Video", frame)
    
    x = cv2.waitKey(50)
    ch = chr(x & 0xFF)
    if ch == 'q':
        break
    elif ch == 'd':
        scanImage(frame)

cv2.destroyAllWindows()


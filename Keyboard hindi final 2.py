# %%
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import colorsys
import math 
import time
import pandas as pd



# %%
Hindi_Dict = {
    "LHTap":{
        "IndexTop":"\u0D05",
        "IndexMiddle":"\u0D07",
        "IndexBottom":"\u0D09",
        "IndexNail":"x",
        "MiddleTop":"\u0D0E",
        "MiddleMiddle":"\u0D12",
        "MiddleBottom":"\u0D14",
        "MiddleNail":"\u0D59",
        "RingTop":"\u0D15",
        "RingMiddle":"\u0D17",
        "RingBottom":"\u0D19",
        "RingNail":"\u0D5A",
        "LittleTop":"\u0D1A",
        "LittleMiddle":"\u0D1C",
        "LittleBottom":"\u0D1E",
        "LittleNail":"\u0D5B",
        "Hold":"X"
        },
    "RHTap":{
        "IndexTop":"\u0D2F",
        "IndexMiddle":"\u0D32",
        "IndexBottom":"\u0D39",
        "IndexNail":"\u0D3C",
        "MiddleTop":"\u0D2A",
        "MiddleMiddle":"\u0D2C",
        "MiddleBottom":"\u0D2E",
        "MiddleNail":"\u0D3A",
        "RingTop":"\u0D24",
        "RingMiddle":"\u0D26",
        "RingBottom":"\u0D28",
        "RingNail":"\u0D3B",
        "LittleTop":"\u0D1F",
        "LittleMiddle":"\u0D21",
        "LittleBottom":"\u0D23",
        "LittleNail":"\u0D5C",
        "Hold":"X"
        },
      "LHLongPress":{
        "IndexTop":"LL1",
        "IndexMiddle":"LL2",
        "IndexBottom":"LL3",
        "IndexNail":"LLq",
        "MiddleTop":"LL4",
        "MiddleMiddle":"LL5",
        "MiddleBottom":"LL6",
        "MiddleNail":"LLw",
        "RingTop":"LL7",
        "RingMiddle":"LL8",
        "RingBottom":"LL9",
        "RingNail":"LLe",
        "LittleTop":"LL*",
        "LittleMiddle":"LL0",
        "LittleBottom":"LL#",
        "LittleNail":"LLt",
        "Hold":"X"
        },
    "RHLongPress":{
        "IndexTop":"RL1",
        "IndexMiddle":"RL2",
        "IndexBottom":"RL3",
        "IndexNail":"RLq",
        "MiddleTop":"RL4",
        "MiddleMiddle":"RL5",
        "MiddleBottom":"RL6",
        "MiddleNail":"RLw",
        "RingTop":"RL7",
        "RingMiddle":"RL8",
        "RingBottom":"RL9",
        "RingNail":"RLe",
        "LittleTop":"RL*",
        "LittleMiddle":"RL0",
        "LittleBottom":"RL#",
        "LittleNail":"RLt",
        "Hold":"X"
        },
    "RHFold":{
        "IndexTop":"\u0D06",
        "IndexMiddle":"\u0D08",
        "IndexBottom":"\u0D0A",
        "IndexNail":"\u0D02",
        "MiddleTop":"\u0D0F",
        "MiddleMiddle":"\u0D13",
        "MiddleBottom":"\u0D10",
        "MiddleNail":"\u0D03",
        "RingTop":"\u0D16",
        "RingMiddle":"\u0D18",
        "RingBottom":"X",
        "RingNail":"X",
        "LittleTop":"\u0D1B",
        "LittleMiddle":"\u0D1D",
        "LittleBottom":"\u0D4D",
        "LittleNail":"X",
        "Hold":"X"
        },
    "LHFold":{
        "IndexTop":"\u0D30",
        "IndexMiddle":"\u0D35",
        "IndexBottom":"X",
        "IndexNail":"\u0D31",
        "MiddleTop":"\u0D2B",
        "MiddleMiddle":"\u0D2D",
        "MiddleBottom":"\u0D38",
        "MiddleNail":"\u0D34",
        "RingTop":"\u0D25",
        "RingMiddle":"\u0D27",
        "RingBottom":"\u0D37",
        "RingNail":"\u0D33",
        "LittleTop":"\u0D20",
        "LittleMiddle":"\u0D22",
        "LittleBottom":"\u0D36",
        "LittleNail":"X",
        "Hold":"X"
        },
    "LHLongPressFold":{
        "IndexTop":"\u0D30",
        "IndexMiddle":"\u0D35",
        "IndexBottom":"X",
        "IndexNail":"\u0D31",
        "MiddleTop":"\u0D2B",
        "MiddleMiddle":"\u0D2D",
        "MiddleBottom":"\u0D38",
        "MiddleNail":"\u0D34",
        "RingTop":"\u0D25",
        "RingMiddle":"\u0D27",
        "RingBottom":"\u0D37",
        "RingNail":"\u0D33",
        "LittleTop":"\u0D20",
        "LittleMiddle":"\u0D22",
        "LittleBottom":"\u0D36",
        "LittleNail":"X",
        "Hold":"X"
    },
    "RHLongPressFold":{
        "IndexTop":"\u0D30",
        "IndexMiddle":"\u0D35",
        "IndexBottom":"X",
        "IndexNail":"\u0D31",
        "MiddleTop":"\u0D2B",
        "MiddleMiddle":"\u0D2D",
        "MiddleBottom":"\u0D38",
        "MiddleNail":"\u0D34",
        "RingTop":"\u0D25",
        "RingMiddle":"\u0D27",
        "RingBottom":"\u0D37",
        "RingNail":"\u0D33",
        "LittleTop":"\u0D20",
        "LittleMiddle":"\u0D22",
        "LittleBottom":"\u0D36",
        "LittleNail":"X",
        "Hold":"X"
    }
}

# %%
Keys_Dict = {
    "LHTap":{
        "IndexTop":"LT1",
        "IndexMiddle":"LT2",
        "IndexBottom":"LT3",
        "IndexNail":"LTq",
        "MiddleTop":"LT4",
        "MiddleMiddle":"LT5",
        "MiddleBottom":"LT6",
        "MiddleNail":"LTw",
        "RingTop":"LT7",
        "RingMiddle":"LT8",
        "RingBottom":"LT9",
        "RingNail":"LTe",
        "LittleTop":"LT*",
        "LittleMiddle":"LT0",
        "LittleBottom":"LT#",
        "LittleNail":"LTt",
        "Hold":"Lh"
        },
    "RHTap":{
        "IndexTop":"RT1",
        "IndexMiddle":"RT2",
        "IndexBottom":"RT3",
        "IndexNail":"RTq",
        "MiddleTop":"RT4",
        "MiddleMiddle":"RT5",
        "MiddleBottom":"RT6",
        "MiddleNail":"RTw",
        "RingTop":"RT7",
        "RingMiddle":"RT8",
        "RingBottom":"RT9",
        "RingNail":"RTe",
        "LittleTop":"RT*",
        "LittleMiddle":"RT0",
        "LittleBottom":"RT#",
        "LittleNail":"RTt",
        "Hold":" "
        },
    "LHLongPress":{
        "IndexTop":"LL1",
        "IndexMiddle":"LL2",
        "IndexBottom":"LL3",
        "IndexNail":"LLq",
        "MiddleTop":"LL4",
        "MiddleMiddle":"LL5",
        "MiddleBottom":"LL6",
        "MiddleNail":"LLw",
        "RingTop":"LL7",
        "RingMiddle":"LL8",
        "RingBottom":"LL9",
        "RingNail":"LLe",
        "LittleTop":"LL*",
        "LittleMiddle":"LL0",
        "LittleBottom":"LL#",
        "LittleNail":"LLt",
        "Hold":"Lh"
        },
    "RHLongPress":{
        "IndexTop":"RL1",
        "IndexMiddle":"RL2",
        "IndexBottom":"RL3",
        "IndexNail":"RLq",
        "MiddleTop":"RL4",
        "MiddleMiddle":"RL5",
        "MiddleBottom":"RL6",
        "MiddleNail":"RLw",
        "RingTop":"RL7",
        "RingMiddle":"RL8",
        "RingBottom":"RL9",
        "RingNail":"RLe",
        "LittleTop":"RL*",
        "LittleMiddle":"RL0",
        "LittleBottom":"RL#",
        "LittleNail":"RLt",
        "Hold":" "
        },
    "LHFold":{
        "IndexTop":"LF1",
        "IndexMiddle":"LF2",
        "IndexBottom":"LF3",
        "IndexNail":"LFq",
        "MiddleTop":"LF4",
        "MiddleMiddle":"LF5",
        "MiddleBottom":"LF6",
        "MiddleNail":"LFw",
        "RingTop":"LF7",
        "RingMiddle":"LF8",
        "RingBottom":"LF9",
        "RingNail":"LFe",
        "LittleTop":"LF*",
        "LittleMiddle":"LF0",
        "LittleBottom":"LF#",
        "LittleNail":"LFt",
        "Hold":"Lh"
        },
    "RHFold":{
        "IndexTop":"RF1",
        "IndexMiddle":"RF2",
        "IndexBottom":"RF3",
        "IndexNail":"RFq",
        "MiddleTop":"RF4",
        "MiddleMiddle":"RF5",
        "MiddleBottom":"RF6",
        "MiddleNail":"RFw",
        "RingTop":"RF7",
        "RingMiddle":"RF8",
        "RingBottom":"RF9",
        "RingNail":"RFe",
        "LittleTop":"RF*",
        "LittleMiddle":"RF0",
        "LittleBottom":"RF#",
        "LittleNail":"RFt",
        "Hold":" "
        },
    "LHLongPressFold":{
        "IndexTop":"LLF1",
        "IndexMiddle":"LLF2",
        "IndexBottom":"LLF3",
        "IndexNail":"LLFq",
        "MiddleTop":"LLF4",
        "MiddleMiddle":"LLF5",
        "MiddleBottom":"LLF6",
        "MiddleNail":"LLFw",
        "RingTop":"LLF7",
        "RingMiddle":"LLF8",
        "RingBottom":"LLF9",
        "RingNail":"LLFe",
        "LittleTop":"LLF*",
        "LittleMiddle":"LLF0",
        "LittleBottom":"LLF#",
        "LittleNail":"LLFt",
        "Hold":"Lh"
        },
    "RHLongPressFold":{
        "IndexTop":"RLF1",
        "IndexMiddle":"RLF2",
        "IndexBottom":"RLF3",
        "IndexNail":"RLFq",
        "MiddleTop":"RLF4",
        "MiddleMiddle":"RLF5",
        "MiddleBottom":"RLF6",
        "MiddleNail":"RLFw",
        "RingTop":"RLF7",
        "RingMiddle":"RLF8",
        "RingBottom":"RLF9",
        "RingNail":"RLFe",
        "LittleTop":"RLF*",
        "LittleMiddle":"RLF0",
        "LittleBottom":"RLF#",
        "LittleNail":"RLFt",
        "Hold":" "
    }
}

# %%
Keys = pd.DataFrame(Keys_Dict,
             index = ["IndexTop", "IndexMiddle", "IndexBottom", "IndexNail", 
                      "MiddleTop", "MiddleMiddle", "MiddleBottom", "MiddleNail", 
                      "RingTop", "RingMiddle", "RingBottom", "RingNail", 
                      "LittleTop", "LittleMiddle", "LittleBottom", "LittleNail","Hold"])
Keys

# %%
Hindi = pd.DataFrame(Hindi_Dict,
             index = ["IndexTop", "IndexMiddle", "IndexBottom", "IndexNail", 
                      "MiddleTop", "MiddleMiddle", "MiddleBottom", "MiddleNail", 
                      "RingTop", "RingMiddle", "RingBottom", "RingNail", 
                      "LittleTop", "LittleMiddle", "LittleBottom", "LittleNail","Hold"])
Hindi

# %%
PredtoKey = ["1", "2", "3", "q", "4", "5", "6", "w", "7", "8", "9", "e", "*", "0", "#", "t","h","f","-1"]

# %%
def KeyPrediction(cord):
    try:
        f_arr = fold(cord)
        if 1 in f_arr:
            if f_arr == [1,1,1,1]:
                return -2,4
            else:
                for i,num in enumerate(f_arr):
                    if num==1:
                        return (4*i+3),4
        dist = np.ndarray((4,4),dtype=float)
        min = [75,-4]
        key = [0,0,1,2,4,4,6,6,8,8,10,10,12,12,14,14]
        lst = [8,7,6,5,12,11,10,9,16,15,14,13,20,19,18,17]
        for i,id in enumerate(lst):
            dist[i%4,int(i/4)] = math.sqrt((cord[id][0]-cord[4][0])**2+(cord[id][1]-cord[4][1])**2+(cord[id][2]-cord[4][2])**2/3)
            if min[0] > dist[i%4,int(i/4)]:
                min[0] = dist[i%4,int(i/4)]
                min[1] = i
        if dist[0,0]<50 and dist[0,1]<50 and dist[0,2]<50 and dist[0,3]<50:
            return -3,4
        if min[1]==-4:
            return -1,4
        return key[min[1]],lst[min[1]]
    except IndexError:
        return -1,4


# %%
prevkey = [""]
Consonant_dict = [
    "\u0D15","\u0D16", "\u0D17", "\u0D18", "\u0D19",  "\u0D1A", "\u0D1B","\u0D1C", "\u0D1D", "\u0D1E","\u0D1F", "\u0D20", "\u0D21" ,"\u0D22", "\u0D23", "\u0D24",  
    "\u0D25", "\u0D26","\u0D27","\u0D28", "\u0D2A", "\u0D2B", "\u0D2C","\u0D2D", "\u0D2E","\u0D2F", "\u0D30", "\u0D31", "\u0D32", "\u0D33", "\u0D34","\u0D35", "\u0D36", "\u0D37", "\u0D38", "\u0D39" ]
Vowel_dict = [
    "\u0D06","\u0D07", "\u0D08", "\u0D09","\u0D0A","\u0D0B","\u0D0E", "\u0D0F", "\u0D10","\u0D12","\u0D13", "\u0D14"]
VowelSign_dict = ["\u0D3E","\u0D3F", "\u0D40", "\u0D41", "\u0D42", "\u0D43","\u0D45", "\u0D46", "\u0D47","\u0D48", "\u0D4A", "\u0D4B", "\u0D4C"]
def kpress(key):
    global prevkey
    if prevkey[-1] in Consonant_dict and key in Vowel_dict:
        print(VowelSign_dict[Vowel_dict.index(key)])
    else:
        if key == 'X':
            print(Key.backspace)
            prevkey.pop()
        else:
            print(key)
        
    prevkey.append(key)

def UpdateKeypad(keypad,key):
    if key == "Lh":
        if keypad[-1] == " ":
            return keypad[:-1]
        else:
            while keypad[-1]!="R" and keypad[-1]!="L":
                keypad = keypad[:-1]
            return keypad[:-1]
    return keypad+key 



# %%
rest_state = [["-1","-1"],["f","-1"],["-1","f"],["f","f"]]
def Keypress(ptime , ctime , pkey , ckey , keypad, keypad_hindi):
    timediff = ctime-ptime
    if pkey in rest_state and ckey in rest_state:
        return ckey,0,keypad,keypad_hindi 
    elif pkey in rest_state and ckey not in rest_state:
        return ckey,ctime,keypad,keypad_hindi 
    elif pkey==ckey and timediff<1:
        return pkey,ptime,keypad,keypad_hindi 
    elif pkey==ckey:
        if ckey[1]=="-1":
            keypad = UpdateKeypad(keypad,Keys["LHLongPress"][PredtoKey.index(ckey[0])])
            keypad_hindi = UpdateKeypad(keypad_hindi,Hindi["LHLongPress"][PredtoKey.index(ckey[0])])
            try:
                kpress(Hindi["LHLongPress"][PredtoKey.index(ckey[0])])
            except ValueError:
                pass
            return ["r","-1"],0,keypad,keypad_hindi 
        elif ckey[0]=="-1":
            keypad = UpdateKeypad(keypad,Keys["RHLongPress"][PredtoKey.index(ckey[1])])
            keypad_hindi = UpdateKeypad(keypad_hindi,Hindi["RHLongPress"][PredtoKey.index(ckey[1])])
            try:
                kpress(Hindi["RHLongPress"][PredtoKey.index(ckey[1])])
            except ValueError:
                pass
            return ["-1","r"],0,keypad,keypad_hindi 
        elif ckey[1]=="f":
            keypad = UpdateKeypad(keypad,Keys["LHLongPressFold"][PredtoKey.index(ckey[0])])
            keypad_hindi = UpdateKeypad(keypad_hindi,Hindi["LHLongPressFold"][PredtoKey.index(ckey[0])])
            try:
                kpress(Hindi["LHLongPressFold"][PredtoKey.index(ckey[0])])
            except ValueError:
                pass
            return ["-1","r"],0,keypad,keypad_hindi
        elif ckey[0]=="f":
            keypad = UpdateKeypad(keypad,Keys["RHLongPressFold"][PredtoKey.index(ckey[1])])
            keypad_hindi = UpdateKeypad(keypad_hindi,Hindi["RHLongPressFold"][PredtoKey.index(ckey[1])])
            try:
                kpress(Hindi["RHLongPressFold"][PredtoKey.index(ckey[1])])
            except ValueError:
                pass
            return ["-1","r"],0,keypad,keypad_hindi
        return ["-1","-1"],0,keypad,keypad_hindi
    elif pkey!=ckey and ("r" not in pkey) and pkey!=["-1","-1"] and timediff>0.3:
        if pkey[1]=="-1":
            keypad = UpdateKeypad(keypad,Keys["LHTap"][PredtoKey.index(pkey[0])])
            keypad_hindi = UpdateKeypad(keypad_hindi,Hindi["LHTap"][PredtoKey.index(pkey[0])])
            try:
                kpress(Hindi["LHTap"][PredtoKey.index(pkey[0])])
            except ValueError:
                pass
            return ["r","-1"],0,keypad,keypad_hindi 
        elif pkey[1]=="f":
            keypad = UpdateKeypad(keypad,Keys["LHFold"][PredtoKey.index(pkey[0])])
            keypad_hindi = UpdateKeypad(keypad_hindi,Hindi["LHFold"][PredtoKey.index(pkey[0])])
            try:
                kpress(Hindi["LHFold"][PredtoKey.index(pkey[0])])
            except ValueError:
                pass
            return ["r","-1"],0,keypad,keypad_hindi 
        elif pkey[0]=="-1":
            keypad = UpdateKeypad(keypad,Keys["RHTap"][PredtoKey.index(pkey[1])])
            keypad_hindi = UpdateKeypad(keypad_hindi,Hindi["RHTap"][PredtoKey.index(pkey[1])])
            try:
                kpress(Hindi["RHTap"][PredtoKey.index(pkey[1])])
            except ValueError:
                pass
            return ["-1","r"],0,keypad,keypad_hindi 
        elif pkey[0]=="f":
            keypad = UpdateKeypad(keypad,Keys["RHFold"][PredtoKey.index(pkey[1])])
            keypad_hindi = UpdateKeypad(keypad_hindi,Hindi["RHFold"][PredtoKey.index(pkey[1])])
            try:
                kpress(Hindi["RHFold"][PredtoKey.index(pkey[1])])
            except ValueError:
                pass
            return ["-1","r"],0,keypad,keypad_hindi 
        return ["-1","-1"],0,keypad,keypad_hindi 
    elif "r" in pkey and ckey in rest_state:
        return ckey,0,keypad,keypad_hindi 
    elif pkey!=ckey and "r" not in pkey:
        return ["-1","-1"],0,keypad,keypad_hindi 
    else:
        return pkey,0,keypad,keypad_hindi 

# %%
def fold(cord):
    X = np.append(np.array(cord[5::4])[:,0].reshape(4,1), np.ones((4, 1)),axis = 1)
    Y = np.array(cord[5::4])[:,1]
    [m,c] = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    f = [0,0,0,0]
    for i,lno in enumerate([8,12,16,20]):
        if cord[lno][1]-m*cord[lno][0]-c>0:
            f[i] = 1
    return f

# %%
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      min_detection_confidence=0.8)
ptime = 0
pkey = ["r","r"]
keypad = ""
dist = 0
handkey = {'Left':1 , 'Right':0}
keypad = ""
nkeypad = ""
keypad_hindi = ""

while cv2.waitKey(1)!=27:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    h, w, c = img.shape
    tempi = [np.zeros((500,500,3),dtype=np.uint8),np.zeros((500,500,3),dtype=np.uint8)]
    ctime = time.time()
    Pred = ["-1","-1"]
    
    if results.multi_hand_landmarks:
        for no,hand in enumerate(results.multi_hand_landmarks):
            hand.landmark[4].x = hand.landmark[3].x + (hand.landmark[4].x - hand.landmark[3].x)*1.1
            cord = []
            y_max = 0
            x_max = 0
            x_min = w
            y_min = h
            for i in hand.landmark:
                x, y = int(i.x * w), int(i.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            hno = handkey[results.multi_handedness[no].classification[0].label]
            for id,lm in enumerate(hand.landmark):
                cord.append([int((250-(x_max-x_min)/2)+lm.x *w-x_min),int((250-(y_max-y_min)/2)+lm.y*h-y_min),lm.z*w])
                cv2.circle(tempi[hno], (cord[id][0],cord[id][1]), 3, tuple(255*i for i in colorsys.hsv_to_rgb(abs(lm.z*2),0.8,0.8)), cv2.FILLED) 
            KeyPred,pt = KeyPrediction(cord)
            Pred[hno] = PredtoKey[KeyPred]
        ckey = Pred
        pkey , ptime, keypad, keypad_hindi = Keypress(ptime , ctime , pkey , ckey , keypad , keypad_hindi)
    final = np.concatenate(tempi,axis=1)
    cv2.putText(final,str(Pred[0]) + str(Pred[1]), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3) 
    cv2.putText(final, str(keypad[-20:]), (10,110), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    cv2.imshow("Image", final)
cv2.destroyAllWindows()

# %%




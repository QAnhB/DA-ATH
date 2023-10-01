from ultralytics import YOLO
import pandas as pd
import cv2
import numpy as np

class ObjectDetection:
    def __init__(self):
        self.model = self.load_model()
    
    def load_model(self):
        model = YOLO('D:\\Downloads\\Clone\\new\\model\\shecodes-da-ath\\detection\\best.pt')
        return model    
    
    def describe(self, class_list,df):
        class_list = np.unique(class_list)

        key_dict = {}
        for name,cate,describe in zip(df['Name'], df['Category'],df['Description']):
            key_dict[name] = [cate,describe]
        
        for top in class_list:
            if top == 'glass':
                print(*key_dict['brown glass'])
            elif top in key_dict.keys() and top != 'glass':
                print(*key_dict[top])
            else:
                print('cannot detect')

    def __call__(self):
        vid = cv2.VideoCapture(0)

        df = pd.read_csv('D:\\Downloads\\Clone\\new\\model\\shecodes-da-ath\\Class_description.csv')
        class_names = df['Name']
        list_class = []

        while True:
            ret,frame = vid.read()
            results = self.model(frame)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1,y1,x2,y2 = box.xyxy[0]
                    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

                    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255),3)
                    cls = int(box.cls[0])
                    list_class.append(class_names[cls])
                    cv2.putText(frame, class_names[cls], [x1,y1], cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

            cv2.imshow('Detect',frame)

            if cv2.waitKey(1) == ord('q'):
                self.describe(list_class,df)
                break
        
        vid.release()
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    model = ObjectDetection()
    model()
            
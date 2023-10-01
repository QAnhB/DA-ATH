from typing import Any
from ultralytics import YOLO
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

class ImageClassification:
    def __init__(self,path):
        self.img = path
        self.model = self.load_model()
        self.top1 = self.results()
    
    def load_model(self): 
        model = YOLO('D:\\Downloads\\Clone\\new\\model\\shecodes-da-ath\\best.pt')  # load a custom model
        return model
    
    def results(self):
        result_list = []
        result = self.model.predict(self.img)
        result_list.append(result)
        
        top = []
        for r in result:
            top1 = r.probs.top1
            classes = r.names
            top1_class  = classes[top1]
            top.append(top1_class)
        return top
    
    def describe(self):
        df = pd.read_csv('D:\\Downloads\\Clone\\new\\model\\shecodes-da-ath\\Class_description.csv')

        key_dict = {}
        for name,cate,describe in zip(df['Name'], df['Category'],df['Description']):
            key_dict[name] = [cate,describe]
        
        for top in self.top1:
            if top == 'glass':
                class_name = key_dict['brown glass'][0]
                descript = key_dict['brown glass'][1]
                text = f'{class_name} \n {descript}'
            elif top in key_dict.keys() and top != 'glass':
                class_name = key_dict[top][0]
                descript = key_dict[top][1]
                text = f'{class_name} \n {descript}'
            else:
                text = 'Cannot detect'
        
        return text
    
    # def __call__(self, *args: Any, **kwds: Any) -> Any:
    #     self.describe()
    
if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    file_path = askopenfilename()

    model = ImageClassification(file_path)
    text = model.describe
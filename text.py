from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

class TextExtract:
    def __init__(self, sentence):
        self.sentence = sentence
        self.keywords = self.extracted_keyword()
    
    def extracted_keyword(self):
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(self.sentence)
        keywords = [word.lower() for word in tokens if word.lower() not in stop_words]
        return keywords

    def keywords_detect(self):
        df = pd.read_csv('D:\\Downloads\\Clone\\new\\model\\DA-ATH\\Class_description.csv')

        key_dict = {}
        for name,cate,describe in zip(df['Name'], df['Category'],df['Description']):
            key_dict[name] = [cate,describe]
        
        for key in self.keywords:
            if key == 'glass':
                class_name = key_dict['brown glass'][0]
                descript = key_dict['brown glass'][1]
                text = f'{class_name} \n {descript}'
            elif key not in key_dict.keys():
                continue
            elif key in key_dict.keys():
                class_name = key_dict[key][0]
                descript = key_dict[key][1]
                text = f'{class_name} \n {descript}'
        
        return text

if __name__ == "__main__":
    sentence = input('Hi! What do you need?\n')
    model = TextExtract(sentence)
    text = model.keywords_detect()
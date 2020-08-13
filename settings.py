import os

data_path="..." #path to data
label_var="labels"
path_var="path"
label_sep="|"
result_dir="..." # path to result dir. where results from various models are saved
prediction_dir="..."
manuscript_dir= '...'


class Config:
    def __init__(self,image_path,train_csv,validation_csv,labels,multilabel=False,class_weights=None):
        self.image_path=image_path
        self.train_csv=os.path.join(image_path,train_csv)
        self.validation_csv=os.path.join(image_path,validation_csv)
        self.labels=labels
        self.multilabel=multilabel
        self.class_weights=None if class_weights is None else os.path.join(image_path,class_weights)


perch_config = Config(
    image_path='...', #path to perch images
    train_csv='classification_train.csv',
    validation_csv='classification_val.csv',
    labels={
        0:'Consolidation',
        1:'Other Infiltrate',
        2:'Consolidation and Other Infiltrate',
        3:'Normal',
        4:'Uninterpretable'},
    multilabel=False,
    class_weights='class_weights.txt'
)



chestray_config = Config(
    image_path='...', #path to chestray images

    train_csv='classification_train.csv',
    validation_csv='classification_val.csv',
    labels={
        0: 'Fibrosis',
        1: 'Mass',
        2: 'No Finding',
        3: 'Edema',
        4: 'Atelectasis',
        5: 'Emphysema',
        6: 'Cardiomegaly',
        7: 'Pleural_Thickening',
        8: 'Nodule',
        9: 'Infiltration',
        10: 'Pneumothorax',
        11: 'Hernia',
        12: 'Effusion',
        13: 'Consolidation',
        14: 'Pneumonia'
    },
    multilabel=True,
    class_weights='class_weights.txt'
)




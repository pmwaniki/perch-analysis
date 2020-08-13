import numpy as np
import tempfile

class Checkpoint():
    def __init__(self,model,patience=5,minimize=True,restore_best=True):
        self.queue=[]
        self.patience=patience
        self.path=tempfile.mktemp()
        self.model=model
        self.minimize=minimize
        self.resore_best=restore_best

    def save(self):
        self.model.save_weights(self.path)

    def restore(self):
        self.model.load_weights(self.path)
        print("<<Weight restored from checkpoint>>")

    def better(self,value):
        if self.minimize:
            return value< np.min(self.queue)
        else:
            return value>np.max(self.queue)

    def compare(self,value):
        if len(self.queue)<1:
            self.queue.append(value)
            self.save()
            return True
        elif len(self.queue)<self.patience:
            if self.better(value):
                self.save()
                self.queue=[]
            self.queue.append(value)
            return True
        else:
            if self.better(value):
                self.save()
                self.queue=[value]
                return True
            else:
                print("Metric did not improve after %d epochs" % self.patience)
                if self.resore_best: self.restore()
                return False







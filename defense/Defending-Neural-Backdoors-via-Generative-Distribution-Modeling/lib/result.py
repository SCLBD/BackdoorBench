class Result:
    def __init__(self, correct, total):
        self.correct = correct
        self.total = total
    
    def report_accuracy(self):
        print("Accuracy: %.2f%%" % (100. * self.correct / self.total))
    
    def accuracy(self):
        acc = 100. * self.correct / self.total
        return float('%.2f' % acc)
    
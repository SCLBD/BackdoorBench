class TriggerPool:
    def __init__(self):
        self.triggers = [] # Trigger list
        self.results = [] # Result list
        
    def add(self, trigger):
        """add one trigger to the pool"""
        self.triggers.append(trigger)
    
    def test(self, model, data):
        """test untested triggers"""
        untested_triggers = range(len(self.results), len(self.triggers))
        for i in untested_triggers:
            self.results.append(model.test(data, 0.1, self.triggers[i]))
            
    def expand(self, num=1):
        """add new triggers based on self-expansion rules"""
        scores = [result.accuracy() for result in self.results] # TODO: add density punishment
        best_trigger = self.triggers[scores.index(max(scores))]
        def spawn(trigger):
            return trigger.duplicate().add_noise(type_="Gaussian",args={"std":0.1})
        for i in range(num):
            self.add(spawn(best_trigger))
            
    def success_triggers(self, threshold=90):
        """threshold 0~100. return a list of triggers"""
        return [self.triggers[i] 
                for i in range(len(self.results))
                if self.results[i].accuracy() >= threshold]
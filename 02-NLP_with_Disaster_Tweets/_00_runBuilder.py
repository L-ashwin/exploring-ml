#inspiration:https://www.youtube.com/watch?v=NSKghk0pcco
class RunBuilder():
    @staticmethod
    def get_runs(params):
        from itertools import product
        keys = params.keys()   # parameter name/ID
        vals = params.values() # list of values for parameter
        
        runs = [] 
        # keys for output dict are parameter name/ID (same as input)
        # values for each output dict is a different combination of values
        # from the list of values for parameter
        for vals_out in product(*vals):
            runs.append({key:val for key, val in zip(keys, vals_out)})
        
        return runs
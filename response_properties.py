class QueryResults:
    def __init__(self):
        
        self.indice = None
        self.distance = None
        self.response = None

    def update_fields(self, new_index, new_distance, new_response):
        self.indice = new_index
        self.distance = new_distance
        self.response = new_response

class Keyphrase:
    def __init__(self):
        self.key_phrase = None
        self.ID = None
        self.start_index = None
        self.end_index = None
        self.label = None
        self.text_path = None
        self.sentence = None

    def __str__(self):
        return_string = ""
        if self.ID is not None:
            return_string += self.ID + " "
        if self.label is not None:
            return_string += self.label + " "
        if self.start_index is not None:
            return_string += str(self.start_index) + " "
        if self.end_index is not None:
            return_string += str(self.end_index) + " "
        if self.key_phrase is not None:
            return_string += self.key_phrase
        return return_string

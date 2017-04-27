class Relation:
    def __init__(self):
        self.Arg1 = None
        self.Arg2 = None
        self.ID = None  #helps find classification/relationship
        self.label = None
        self.text_path = None
        self.sentence1 = None
        self.sentence2 = None
        self.key_phrase1 = None
        self.key_phrase2 = None

    #print statement
    def __str__(self):
        return_string = ""
        if self.ID is not None:
            return_string += str(self.ID) + " "
        if self.label is not None:
            return_string += str(self.label) + " "
        if self.Arg1 is not None:
            return_string += str(self.Arg1) + " "
        if self.Arg2 is not None:
            return_string += str(self.Arg2) + " "
        if self.sentence1 is not None:
            return_string += str(self.sentence1) + " "
        if self.sentence2 is not None:
            return_string += str(self.sentence2) + " "
        if self.key_phrase1 is not None:
            return_string += self.key_phrase1
        if self.key_phrase2 is not None:
            return_string += self.key_phrase2
        return return_string

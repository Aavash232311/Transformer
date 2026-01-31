
class Tokenizer:

    def __init__(self, chunk):
        self.all_chars = list(dict.fromkeys(chunk)) 

        self.characters = {} # {"H": 0}
        self.index = {} # {0: "H"} for encoding and decoding

        for i in range(0, len(self.all_chars)):
            self.characters[i] = self.all_chars[i]
            self.index[self.all_chars[i]] = i

    def encode(self, string):
        encoded = []
        input_list = list(string)
        for i in range(0, len(input_list)):
            current_input = input_list[i]
            encoded.append(
                self.index[current_input]
            )
        return encoded
    
    def decoder(self, string):
        decoded = []
        for i in range(0, len(string)):
            current_pos = string[i]
            decoded.append(
                self.characters[current_pos]
            )
        return "".join(decoded)
    
    def unique_characters(self):
        return self.all_chars

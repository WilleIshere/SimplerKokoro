class SimplerKokoro:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}! Welcome to Simpler Kokoro."

    def farewell(self):
        return f"Goodbye, {self.name}! See you next time."
    
    def get_name(self):
        return self.name
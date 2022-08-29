import interpreter as lang

class Exception:
    def __init__(self, start, end, name, description) -> None:
        self.end = end
        self.start = start
        self.name = name
        self.description = description
    def __repr__(self) -> str:
        result = f"""{self.name}: {self.description}
|-> Code file {self.start.name}, in the line of {self.start.line+1}

    """+lang.showerror(self.start.text, self.start, self.end)

        return result
    def loadcontext(self):
        result = ""
        context = self.context
        position = self.start
        i = 1

        while context:
            i += 1
            result = f"|-> Code file {position.name}, in the line of {str(position.line+1)}, in {context.name}" + result
            position = context.parent_position
            context = context.parent
        return result

class NameException(Exception):
    def __init__(self, start, end, description) -> None:
        super().__init__(start, end, "NameException", description)
class RuntimeException(Exception):
    def __init__(self, start, end, name, description, context) -> None:
        super().__init__(start, end, name, description)
        self.context = context

    def __repr__(self) -> str:
        
        result = f"""{self.name}: {self.description}
{self.loadcontext()}
        
    """+lang.showerror(self.start.text, self.start, self.end)
        return result
class InOperableException(Exception):
    def __init__(self, start, end, description) -> None:
        super().__init__(start, end, "InOperableException", description)
class DivisionByZeroException(RuntimeException):
    def __init__(self, start, end, description, context) -> None:
        super().__init__(start, end, "DivisionByZeroException", description, context)
        self.context = context

class IllegalException(Exception):
    def __init__(self, start, end, name, description) -> None:
        super().__init__(start, end, name, description)

class TypeInferenceException(Exception):
    def __init__(self, start, end, description) -> None:
        self.start = start
        self.end = end

        super().__init__(start, end, "TypeInferenceException", description)

class SyntaxIllegalException(IllegalException):
    def __init__(self, start, end, description) -> None:
        self.start = start
        self.end = end

        super().__init__(start, end, "SyntaxIllegalException", description)
class OperatorIllegalException(IllegalException):
    def __init__(self, start, end, description) -> None:
        super().__init__(start, end, "OperatorIllegalException", description)
class CharacterIllegalException(IllegalException):
    def __init__(self, start, end, description) -> None:
        self.start = start
        self.end = end

        super().__init__(start, end, "CharacterIllegalException", description)

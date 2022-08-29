# This code is written by turrnut
# open source under the AGPL-v3.0 license
# Copyright
#

import sys
import tokens
from types import NoneType
from error import *

version = "1.0"

KEYWORD_DICT = {
    "variable_definition":"val",
    "boolean_true":"true",
    "boolaen_false":"false",
    "logical_and":"&&",
    "logical_not":"!=",
    "logical_or":"||"
}

VAR_PTN = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_" + "$|&"
DIGITS = "0123456789"

FILENAME = ""
TEXT = ""
def run (filename, text):
    """
        RUN THE CODE <text> using the <filename> file name.
        arguments:
        filename - file name of the code file
    """
    tokens, error = Tokenizer(filename, text).tokens()

    if error :
        return None, error

    abstract_syntax_tree = Parser().parse(tokens)

    context = Enviroment("main")
    context.symbols = global_data
    
    if check(abstract_syntax_tree):
        return None, abstract_syntax_tree.error
    resp = Calculator().cal(abstract_syntax_tree.fork, context)
    if type(resp) == tuple:
        return RuntimeResponse().success(NoValue), None
    return resp, resp.error

def check(obj):
    """
    Check if the <obj> Object contains an error
    arguments:
    obj - the Object to check
    """
    if obj.error:
        return True
    return False
def showerror(text, poss, pose):
    """
        Show error arrows.
        arguments:
        text - the error text
        poss - start position
        pose - end position
    """
    result = ''
    start = 0
    end = 0
    if text.rfind('\n', 0, poss.index) > 0 :
        start = text.rfind('\n', 0, poss.index)
    
    elif text.rfind(';', 0, poss.index) > 0:
        start = text.rfind(';', 0, poss.index)
    end = text.find('\n', start + 1)
    if end < 0:
        end = text.find(';', start + 1)
        if end < 0 :
            end = len(text)
    count = pose.line - poss.line + 1
    for index in range(count):
        line = text[start:end]
        cols = 0
        if index == 0:
            cols = poss.col
        
        if index == count - 1:
            cole = pose.col
        else :
            cols = len(line) - 1
        result += line + '\n'
        result += ' ' * cols + '^' * (cole - cols)
        start = end
        end = text.find('\n', start + 1)
        if end < 0:
            end = len(text)
    result = result.replace("\t", "")

    return "\a" + result
class Object:
    pass
class Int(Object):
    def __init__(this, value) -> None:
        this.value = value
        this.setcontext()

    def clone(this):
        new = Int(this.value)
        if type(this) == NoValue:
            new.locate()
            new.setcontext()
            return new
        new.locate(this.start, this.end)
        new.setcontext(this.context)

        return new

    def locate(this, start=None, end=None):
        this.start = start
        this.end = end
        return this

    def setcontext(this, context=None):
        this.context = context
        return this

    def not_op(this):
        if this.value == 0:
            return Bool(1).setcontext(this.context), None
        return Bool(0).setcontext(this.context), None

    def ee(this, num):
        if this.check(num):
            if this.value == num.value:
                return Bool(0).setcontext(this.context), None
            return Bool(1).setcontext(this.context), None
    def ne(this, num):
        if this.check(num):
            if this.value != num.value:
                return Bool(0).setcontext(this.context), None
            return Bool(1).setcontext(this.context), None
    def gt(this, num):
        if this.check(num):
            if this.value > num.value:
                return Bool(0).setcontext(this.context), None
            return Bool(1).setcontext(this.context), None
    def lt(this, num):  
        if this.check(num):
            if this.value < num.value:
                return Bool(0).setcontext(this.context), None
            return Bool(1).setcontext(this.context), None
    def gte(this, num):
        if this.check(num):
            if this.value >= num.value:
                return Bool(0).setcontext(this.context), None
            return Bool(1).setcontext(this.context), None
    def lte(this, num):
        if this.check(num):
            if this.value <= num.value:
                return Bool(0).setcontext(this.context), None
            return Bool(1).setcontext(this.context), None
    
    def modulo(this, factor):
        if this.check(factor):
            return Int(this.value % factor.value).setcontext(), None

    def power(this, factor):
        if this.check(factor):
            return Int(pow(this.value, factor.value)).setcontext(), None
        
    def add(this, factor):
        if this.check(factor): 
            return Int(this.value + factor.value).setcontext(), None

    def minus(this, factor):
        if this.check(factor): 
            return Int(this.value - factor.value).setcontext(), None

    def multiply(this, factor):
        if this.check(factor): 
            return Int(this.value * factor.value).setcontext(), None
    
    def divide(this, factor):
        if this.check(factor):
            if factor.value == 0:
                if type(factor) == Int:
                    return None, DivisionByZeroException(factor.start, factor.end, "Cannot divide by zero", this.context)
                return None, DivisionByZeroException(this.start, this.end, "Cannot divide by zero", this.context)
            return Int(this.value / factor.value).setcontext(), None
    
    def __repr__(this) -> str:
        return f"{this.value}"
    def __str__(this) -> str:
        return f"{this.value}"

    def check(this, obj):
        return isinstance(obj, Int) and obj != None
class Bool(Object):
    def __init__(this, value) -> None:
        this.value = value
        this.setcontext()
        this.locate()

    def clone(this):
        new = Int(this.value)
        if type(this) == NoValue:
            new.locate()
            new.setcontext()
            return new
        new.locate(this.start, this.end)
        new.setcontext(this.context)
        return new
    def locate(this, start=None, end=None):
        this.start = start
        this.end = end
        return this

    def setcontext(this, context=None):
        this.context = context
        return this
  
    def equals(this, factor) -> bool:
        if this.check(factor):
            if this.value == 0 and factor.value == 0 :
                return Bool(0).setcontext(), None
            elif this.value != 0 and factor.value != 0 :
                return Bool(0).setcontext(), None
            elif this.value == 0 and factor.value != 0:
                return Bool(1).setcontext(), None
            elif this.value != 0 and factor.value == 0:
                return Bool(1).setcontext(), None
    def not_equals(this, factor) -> bool:
        if this.check(factor):
            if this.value == 0 and factor.value == 0 :
                return Bool(1).setcontext(), None
            elif this.value != 0 and factor.value != 0 :
                return Bool(1).setcontext(), None
            return Bool(0).setcontext(), None
    
    def and_op(this, factor) -> bool:
        if this.check(factor):
            if this.value == 0 and factor.value == 0 :
                return Bool(0).setcontext(), None
            return Bool(1).setcontext(), None

    def or_op(this,factor) -> bool:
        if this.check(factor):
            if this.value == 0 and factor.value == 0 :
                return Bool(0).setcontext(), None
            elif this.value != 0 and factor.value == 0 :
                return Bool(0).setcontext(), None
            elif this.value == 0 and factor.value != 0 :
                return Bool(0).setcontext(), None
            elif this.value != 0 and this.value != 0 :
                return Bool(1).setcontext(), None

    def check(this, obj):
        return isinstance(obj, Bool)
    
    def __repr__(this) -> str:
        if this.value == 0:
            return "true"
        return "false"
    def __str__(this) -> str:
        if this.value == 0:
            return "true"
        return "false"

class Calculator:
    def cal(this, fork, context):
        name = f"cal_{type(fork).__name__}"
        fun = getattr(this, name, this.no)
        result = fun(fork, context)
        return result
    
    def cal_VariableCreate(this, fork, context):
        response = RuntimeResponse()
        key = fork.name_token.value
        value = response.response(Calculator().cal(fork.value, context))

        if check(response):
            return response
        BANNED = ["nov"]
        for c in key:
            if c in "$|&":
                response.error = NameException(fork.start, fork.end, f"Invalid Naming of variable {key}")
                break    

        if check(response):
            return response

        if key in BANNED:
            response.error = SyntaxIllegalException(fork.start, fork.end, f"Invalid Syntax of usage of keyword '{key}' ")

        if check(response):
            return response
        context.symbols.set(key, value)
        
        return response.success(value)

    def cal_VariableAccess(this, fork, context):
        response = RuntimeResponse()
        key = fork.name_token.value

        value = context.symbols.get(key)

        if not value:
            return response.failure(RuntimeException(fork.start, fork.end, "RuntimeException", f"{key} is not defined", context))
        
        value = value.clone().locate(fork.start, fork.end)

        return response.success(value)
    def cal_Number(this, fork, context):
        return RuntimeResponse().success(Int(fork.token.value).setcontext(context).locate(fork.start, fork.end))

    def cal_Bool(this, fork, context):
        return RuntimeResponse().success((fork).setcontext(context).locate(fork.start, fork.end))

    def cal_Binary(this, fork, context):
        response = RuntimeResponse()
        l = response.response(this.cal(fork.l, context))
        if check(response):
            return response
        result, error = None, None

        r = response.response(this.cal(fork.r, context))
        if check(response):
            return response
        if isinstance(r, Int) and isinstance(l, Int):
            if fork.o.type == tokens.plus : result, error = l.add(r)
            elif fork.o.type == tokens.minus : result, error = l.minus(r)
            elif fork.o.type == tokens.mul : result, error = l.multiply(r)
            elif fork.o.type == tokens.div : result, error = l.divide(r)
            elif fork.o.type == tokens.POWER : result, error = l.power(r)
            elif fork.o.type == tokens.MOD : result, error = l.modulo(r)
            elif fork.o.type == tokens.EE : result, error = l.ee(r)
            elif fork.o.type == tokens.NE : result, error = l.ne(r)
            elif fork.o.type == tokens.GT : result, error = l.gt(r)
            elif fork.o.type == tokens.GTE : result, error = l.gte(r)
            elif fork.o.type == tokens.LT : result, error = l.lt(r)
            elif fork.o.type == tokens.LTE : result, error = l.lte(r)
        elif isinstance(l, Bool) and isinstance(r, Bool):
            if fork.o.equals(tokens.KW, KEYWORD_DICT["logical_and"]): 

                result, error = l.and_op(r)
            elif fork.o.equals(tokens.KW, KEYWORD_DICT["logical_or"]):result, error = l.or_op(r)
            elif fork.o.type == tokens.NE: result, error = l.not_equals(r)
            elif fork.o.type == tokens.EE: result, error = l.equals(r)

        elif type(result)== NoneType or type(response) == NoneType: 
            return NoValue(), None
        if error: return response.failure(error)
        return response.success(result.locate(fork.start, fork.end))
    def checktype(this):
        pass

    def cal_Unary(this, fork, context):
        response = RuntimeResponse()
        result = response.response(this.cal(fork.n, context))
        if check(response) :
            return response
        
        if fork.o.type == tokens.minus:
            result, error = result.multiply(Int(-1))
        elif fork.o.equal(tokens.KW, KEYWORD_DICT["logical_not"]):
            result, error = result.not_op()

        if error :
            response.failure(error)
        return response.success(result.locate(fork.start, fork.end))

    def no(this, a, b):
        pass
class Enviroment:
    def __init__(this, name, parent=None, parent_position=None) -> None:
        this.name = name
        this.parent = parent
        this.parent_position = parent_position
        this.symbols = DataTable()

class Token:
    def __init__(this, type, value=None, start=None, end=None) -> None:
        this.type = type
        this.value = value

        if start: 
            this.start = start.clone()
            this.end = start.clone()
            this.end.step()
        if end : this.end = end
    def __repr__(this) -> str:
        if this.value :
            return f"{this.type}:{this.value}"
        return f"{this.type}"
    def equals(this, type, value):
        return this.type == type and this.value == value
class Tokenizer :
    def __init__(this, name, text) -> None:
        this.text = text
        this.filename = name
        this.position = Position(-1, 0, -1, name, text)
        this.current = None
        this.step()
    
    def step(this):
        this.position.step(this.current)
        this.current = this.text[this.position.index] if this.position.index < len(this.text) else None

    def tokens(this):
        tokenlist = []
        while this.current != None:
            if this.current in " \t\r\n;":
                this.step()
            elif this.current in DIGITS:
                tokenlist.append(this.numbers())
            elif this.current in VAR_PTN :
                tokenlist.append(this.id())
                    
            elif this.current == "%":
                tokenlist.append(Token(tokens.MOD, start=this.position))
                this.step()
            elif this.current == "^":
                tokenlist.append(Token(tokens.POWER, start=this.position))
                this.step()
            elif this.current == "+":
                tokenlist.append(Token(tokens.plus, start=this.position))
                this.step()
            elif this.current == "-":
                tokenlist.append(Token(tokens.minus, start=this.position))
                this.step()
            elif this.current == "*":
                tokenlist.append(Token(tokens.mul, start=this.position))
                this.step()
            elif this.current == "/":
                tokenlist.append(Token(tokens.div, start=this.position))
                this.step()
            elif this.current == "(":
                tokenlist.append(Token(tokens.LPN, start=this.position))
                this.step()
            elif this.current == ")":
                tokenlist.append(Token(tokens.RPN, start=this.position))
                this.step()
            elif this.current == "!":
                token, error = this.not_equals()
                if error:
                    return [], error
                tokenlist.append(token)
            elif this.current == "=":
                tokenlist.append(this.equals())
            elif this.current == "<":
                tokenlist.append(this.lt())
            elif this.current == ">":
                tokenlist.append(this.gt())
            else:
                start = this.position.clone()
                char = this.current
                this.step()
                return [], CharacterIllegalException(start, this.position, "Illegal Characteracter '"+char+"'")
        tokenlist.append(Token(tokens.END, start = this.position))

        return tokenlist,None

    def lt(this):
        ttype = tokens.LT
        start = this.position.clone()
        this.step()
        
        if this.current == "=":
            this.step()
            ttype = tokens.LTE

        end = this.position
        return Token(type=ttype, start=start, end=end)

    def gt(this):
        ttype = tokens.GT
        start = this.position.clone()
        this.step()
        
        if this.current == "=":
            this.step()
            ttype = tokens.GTE

        end = this.position
        return Token(type=ttype, start=start, end=end)

    def equals(this):
        ttype = tokens.EQ
        start = this.position.clone()
        this.step()
        
        if this.current == "=":
            this.step()
            ttype = tokens.EE

        end = this.position
        return Token(type=ttype, start=start, end=end)

    def not_equals(this):
        start = this.position.clone()
        this.step()
        end = this.position
        if this.current == "=":
            this.step()
            end = this.position
            return Token(tokens.NE, start=start , end=end), None
        this.step()
        return None, CharacterIllegalException(start, end, "Invalid operater, are you referring '!=' ?")


    def id(this):
        id = ""
        start = this.position.clone()

        while this.current != None and this.current in VAR_PTN + DIGITS:
            id += this.current
            this.step()
        token_type = tokens.ID
        for k in KEYWORD_DICT:
            if id == KEYWORD_DICT[k]:
                token_type = tokens.KW
        end = this.position
        if id == "true":
            token_type = tokens.boolean
            return Token(token_type, Bool(0), start, end)
        elif id == "false":
            token_type = tokens.boolean
            return Token(token_type, Bool(1), start, end)
        
        return Token(token_type, id, start, end)
        
    def numbers(this):
        numstr = ""
        dots = 0
        start = this.position.clone()

        while this.current != None and this.current in DIGITS + ".":
            if this.current == ".":
                if dots == 1:
                    break
                dots += 1
                numstr += "."
            else:
                numstr += this.current
            this.step()
            
        if dots == 0:
            return Token(tokens.INT, int(numstr), start=start, end=this.position )
        else:
            return Token(tokens.DEC, float(numstr), start=this.position, end=this.position)
class Number:
    def __init__(this, token) -> None:
        this.token = token

        this.start = token.start
        this.end = token.end
    
    def __repr__(this):
        return f"{this.token}"
class Binary:
    def __init__(this, l, o, r) -> None:
        this.l = l
        this.o = o
        this.r = r
        if l.start and r.end:
            this.start = l.start
            this.end = r.end

    def __repr__(this) -> str:
        return f"({this.l},{this.o},{this.r})"
class Response:
    def __init__(this) -> None:
        this.fork = None
        this.error = None
    def response(this, result):
        if isinstance(result, Response):
            if check(result) :this.error = result.error
            return result.fork
        return result
    
    def success(this, fork):
        this.fork = fork
        return this
    
    def failure(this, error):
        this.error = error
        return this
class DataTable:
    def __init__(this) -> None:
        this.data = {}
        this.parent = None

    def get(this, key):
        value = this.data.get(key, None)
        if value == None and this.parent :
            return this.parent.get(key)
        return value

    def set(this, key, value):
        this.data[key] = value

    def pop(this, key):
        del this.data[key]

class RuntimeResponse(Response):
    def __init__(this) -> None:
        this.error = None
        this.value = None

    def response(this, response):

        if check(response):
            this.error = response.error
        return response.value

    def success(this, value):
        this.value = value
        return this
    
class Parser:
    def parse(this, tokens):
        this.current = None
        this.tokenlist = tokens
        this.index = -1
        this.step()
        result = this.expr()
        if check(result):
            err = type(result.error)
            return result.failure(err(this.current.start, this.current.end, "Invalid Syntax"))
        return result
    def step(this):
        this.index += 1
        if this.index < len(this.tokenlist):
            this.current = this.tokenlist[this.index]
        return this.current
    def expr(this):
        (this.tokenlist)
        response = Response()
        if this.current.equals(tokens.KW, KEYWORD_DICT["variable_definition"]):
            response.response(this.step())

            if this.current.type != tokens.ID:
                response.failure(SyntaxIllegalException(this.current.start, this.current.end, "Expected identifier"))
            
            name = this.current
            response.response(this.step())

            if this.current.type != tokens.EQ :
                return response.failure(SyntaxIllegalException(this.current.start, this.current.end, "Expected '='"))

            response.response(this.step())
            expression = response.response(this.expr())
            if response.error :
                return response
            
            return response.success(VariableCreate(name, expression))

        return this.operation(this.logical, ((tokens.KW, KEYWORD_DICT["logical_and"]),(tokens.KW, KEYWORD_DICT["logical_or"])))
    def logical(this):
        response = Response()

        if this.current.equals(tokens.KW, KEYWORD_DICT["logical_not"]):
            o = this.current
            this.step()
            response.response(this.step())

            n = response.response(this.logical())
            if check(response) :
                return response
            return response.success(Unary(o, n))
        n = response.response(this.operation(this.math, (tokens.EE, tokens.NE, tokens.LT, tokens.GT, tokens.LTE, tokens.GTE, tokens.true, tokens.false)))

        if check(response):
            err = type(response.error)
            return response.failure(err(this.current.start, this.current.end, "Invalid Syntax"))
        return response.success(n)
    def math(this):
        return this.operation(this.term, (tokens.plus, tokens.minus))

    def operation(this, get, operation_tokens):
        response = Response()
        l = response.response(get())
        if check(response):
            return response

        while this.current.type in operation_tokens or (this.current.type,this.current.value) in operation_tokens:
            o = this.current
            response.response(this.step())
            r = response.response(get())
            if check(response):
                return response
            l = Binary(l, o, r)
        return response.success(l)
        
    def term(this):
        return this.operation(this.cell, (tokens.mul, tokens.div))

    def cell(this):
        return this.operation(this.element, (tokens.POWER,))
    
    def element(this):
        response = Response()
        token = this.current
        if token.type in (tokens.plus, tokens.minus):
            response.response(this.step())
            element = response.response(this.element())
            if check(response):
                return response
            return response.success(Unary(token, element))
        elif token.type == tokens.ID:
            response.response(this.step())
            return response.success(VariableAccess(token))
        
        elif token.type in (tokens.INT, tokens.DEC):
            response.response(this.step())
            return response.success(Number(token))

        elif token.type in (tokens.boolean,):
            response.response(this.step())
            return response.success(token.value)
        elif token.type in (tokens.EE, tokens.NE, tokens.LT, tokens.GT, tokens.LTE, tokens.GTE): 
            response.response(this.step())
            return response.success(token.type)
        elif token.type == tokens.LPN :
            response.response(this.step())
            expression = response.response(this.expr())
            if check(response):
                return response
            if this.current.type == tokens.RPN:
                response.response(this.step())
                return response.success(expression)
            else:
                return response.failure(SyntaxIllegalException(this.current.start, this.current.end, "Expected a ')'"))

        return response.failure(SyntaxIllegalException(token.start, token.end, "Expected code "))
class Position:
    def __init__(this, index, line, col, name, text) -> None:
        this.name = name
        this.text = text
        this.index = index
        this.line = line
        this.col = col

    def step(this, current = None):
        this.index += 1
        this.col += 1
        
        if current == "\n":
            this.line += 1
            this.col = 0
        return this
    
    def clone(this):
        return Position(this.index, this.line, this.col, this.name, this.text)
class VariableCreate:
    def __init__(this, name_token, value) -> None:
        this.name_token = name_token
        this.value = value

        this.start = this.name_token.start
        this.end = this.value.end

class VariableAccess:
    def __init__(this, name_token) -> None:
        this.name_token = name_token

        this.start = this.name_token.start
        this.end = this.name_token.end
class Unary:
    def __init__(this, o, n) -> None:
        this.o = o
        this.n = n

        this.start = this.o.start
        this.end = n.end

    def __repr__(this) -> str:
        return f"({this.o},{this.n})"

        
class NoValue(Int):
    def __init__(this) -> None:
        this.value = 0

    def __repr__(this) -> str:
        return "nov"
    def __str__(self) -> str:
        return "nov"


global_data = DataTable()
global_data.set("nov", NoValue())
global_data.set("true", Bool(0))
global_data.set("false", Bool(1))
global_data.set("$EXPORTS", [])

args = sys.argv

def gettype(obj):
    if type(obj) == Int:
        return "Int"
    if type(obj) == Bool:
        return "Bool"
    if type(obj) == NoValue:
        return "nov"
    return "Object"

if len(args) == 2:
    with open(args[1], "r") as fobj:
        code = fobj.read()
    run(args[1], code)




import sys
import interpreter as lang

try:

    print(f"""
ExInt - Expression Interpreter
Version: {lang.version}
""")
    while True:
        enter = input("exint $ ")
        if enter == "quit":
            sys.exit(0)

        result, error = lang.run("shell.exint", enter)
        if error :
            print(error)
        else:
            print(result.value)

            
except Exception as e:
    print("Intrepretation Failed", e)
    sys.exit(0)
except KeyboardInterrupt as k :
    print("Enter \"quit\" to exit.")
except:
    print("Intrepretation failed due to unknown error")
    sys.exit(0)

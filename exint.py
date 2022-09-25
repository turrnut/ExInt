import sys
import interpreter as lang
try:
    args = sys.argv
    if len(args) == 2 :
        with open(args[1], "r") as f:
            data = f.read()
        dataarr = data.split("\n")
        reslist = []
        for d in dataarr:
            result, e = lang.run(args[1], d)
            if e:
                print(e)
                sys.exit(1)
            reslist.append(result.value)
        for val in reslist:
            print(val)
except Exception as e:
    print("Interpretation Failed", e)
    sys.exit(0)
except KeyboardInterrupt as k :
    print("Interpretation interrupted")
except:
    print("Interpretation failed due to unknown error")
    sys.exit(0)

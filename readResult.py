import json
resultObjName = "2017_08_22_17_2"
with open("results/"+resultObjName,'r') as o:
    str = o.read()

resultObj = json.loads(str)
pass


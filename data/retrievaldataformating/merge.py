import json

with open('account.json') as f:
    data1 = json.load(f)

with open('goodafternoon.json') as f:
    data2 = json.load(f)

with open('branchoffice.json') as f:
    data3 = json.load(f)

with open('develop.json') as f:
    data4 = json.load(f)

with open('evening.json') as f:
    data5 = json.load(f)

with open('extendconnection.json') as f:
    data6 = json.load(f)

with open('fair.json') as f:
    data7 = json.load(f)

with open('gamewinprice.json') as f:
    data8 = json.load(f)

with open('gaming.json') as f:
    data9 = json.load(f)

with open('hello.json') as f:
    data10 = json.load(f)

with open('hi.json') as f:
    data11 = json.load(f)

with open('hidessid.json') as f:
    data12 = json.load(f)

with open('hosting.json') as f:
    data13 = json.load(f)

with open('meaningofonline.json') as f:
    data14 = json.load(f)

with open('goodmorning.json') as f:
    data15 = json.load(f)

with open('namastae.json') as f:
    data16 = json.load(f)

with open('name.json') as f:
    data17 = json.load(f)

with open('nettv.json') as f:
    data18 = json.load(f)

with open('watchmobile.json') as f:
    data19 = json.load(f)

with open('newconnection.json') as f:
    data20 = json.load(f)

with open('offensive.json') as f:
    data21 = json.load(f)

with open('okay.json') as f:
    data22 = json.load(f)

with open('onlineservicespassword.json') as f:
    data23 = json.load(f)

with open('pairremote.json') as f:
    data24 = json.load(f)

with open('phonebusy.json') as f:
    data25 = json.load(f)

with open('photoupload.json') as f:
    data26 = json.load(f)

with open('portforward.json') as f:
    data27 = json.load(f)

with open('renewconnection.json') as f:
    data28 = json.load(f)

with open('safenet.json') as f:
    data29 = json.load(f)

with open('services.json') as f:
    data30 = json.load(f)

with open('slowconnection.json') as f:
    data31 = json.load(f)

with open('thankyou.json') as f:
    data32 = json.load(f)

with open('timeback.json') as f:
    data33 = json.load(f)

with open('torrent.json') as f:
    data34 = json.load(f)

with open('wiredamaged.json') as f:
    data35 = json.load(f)


intents1 = data1["intents"]
intents2 = data2["intents"]
intents3 = data3["intents"]
intents4 = data4["intents"]
intents5 = data5["intents"]
intents6 = data6["intents"]
intents7 = data7["intents"]
intents8 = data8["intents"]
intents9 = data9["intents"]
intents10 = data10["intents"]
intents11 = data11["intents"]
intents12 = data12["intents"]
intents13 = data13["intents"]
intents14 = data14["intents"]
intents15 = data15["intents"]
intents16 = data16["intents"]
intents17 = data17["intents"]
intents18 = data18["intents"]
intents19 = data19["intents"]
intents20 = data20["intents"]
intents21 = data21["intents"]
intents22 = data22["intents"]
intents23 = data23["intents"]
intents24 = data24["intents"]
intents25 = data15["intents"]
intents26 = data26["intents"]
intents27 = data27["intents"]
intents28 = data28["intents"]
intents29 = data29["intents"]
intents30 = data30["intents"]
intents31 = data31["intents"]
intents32 = data32["intents"]
intents33 = data33["intents"]
intents34 = data34["intents"]
intents35 = data35["intents"]






listitem = [intents1, intents2, intents3, intents4, intents5, intents6, intents7, intents8, intents9, intents10, intents11, intents12, intents13, intents14, intents15, intents16, intents17, intents18, intents19, intents20, intents21, intents22, intents23, intents24, intents25, intents26, intents27, intents28, intents29, intents30, intents31, intents32, intents33, intents34, intents35]
data = {"intents" : [] }

data["intents"].extend(intents1)
data["intents"].extend(intents2)
data["intents"].extend(intents3)
data["intents"].extend(intents4)
data["intents"].extend(intents5)
data["intents"].extend(intents6)
data["intents"].extend(intents7)
data["intents"].extend(intents8)
data["intents"].extend(intents9)
data["intents"].extend(intents10)
data["intents"].extend(intents11)
data["intents"].extend(intents12)
data["intents"].extend(intents13)
data["intents"].extend(intents14)
data["intents"].extend(intents15)
data["intents"].extend(intents16)
data["intents"].extend(intents17)
data["intents"].extend(intents18)
data["intents"].extend(intents19)
data["intents"].extend(intents20)
data["intents"].extend(intents21)
data["intents"].extend(intents22)
data["intents"].extend(intents23)
data["intents"].extend(intents24)
data["intents"].extend(intents25)
data["intents"].extend(intents26)
data["intents"].extend(intents27)
data["intents"].extend(intents28)
data["intents"].extend(intents29)
data["intents"].extend(intents30)
data["intents"].extend(intents31)
data["intents"].extend(intents32)
data["intents"].extend(intents33)
data["intents"].extend(intents34)
data["intents"].extend(intents35)


print(json.dumps(data, indent=4))

with open('merged_json.json', "w") as f:
    f.write(json.dumps(data, indent=4))

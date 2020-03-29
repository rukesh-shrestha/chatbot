import json

import re

with open('nointernet', 'r') as file:
    lines = file.readlines()
    patterns = []
    responses = []
    who = ""
    what = ""
    firstTime = True
    d = {
        'tag': 'nointernetconnection',
        'patterns': [],
        'responses': [],
        'context_set': ''
    }

    for index, line in enumerate(lines):
        try:
            line = line.strip()
            if line == '' or line == '?':
                continue
            if re.match('human', line.lower() ) or re.match('reply', line.lower()):
                if not firstTime:
                    if who == 'human':
                        d['patterns'].append(what)
                    else:
                        d['responses'].append(what)

                person, message = line.strip().split(':')
                #This is the new line of convo
                who = person.strip().lower()
                what = message.strip().lower()
                firstTime = False
            else:
                what += line.strip().trim() + '\n'
        except:
            print("Error around line " + str(index + 1))
        
    
    j = {
        'intents': [
            d
        ]
    }
    with open('nointernet.json', 'w') as write_file:
        json.dump(j, write_file)
    

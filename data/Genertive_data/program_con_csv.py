import csv
import re

with open('file.txt', 'r') as file:
    lines = file.readlines()
    with open('file.csv', 'w') as csvfile:
        fieldnames = ['human', 'reply']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        towrite =  {}
        messagebuffer = ''
        personbuffer = ''
        for index, line in enumerate(lines):
            try:
                line = line.strip()
                if line == '' or line == '?':
                    continue
                if re.match(fieldnames[0], line.lower() ) or re.match(fieldnames[1], line.lower()):
                    person, message = line.strip().split(':')
                    if personbuffer == '' or messagebuffer == '':
                        personbuffer = person.strip().lower()
                        messagebuffer += message.strip()
                    else:
                        towrite[personbuffer] = messagebuffer.strip()
                        personbuffer = person.strip().lower()
                        messagebuffer = message.strip()
                else:
                    messagebuffer += line.strip() + " "
                if len(towrite) == 2:
                    writer.writerow(towrite)
                    towrite.clear()
            except:
                print("Error around line " + str(index + 1))
        if personbuffer != '' and messagebuffer != '':
            towrite[personbuffer] = messagebuffer
            if len(towrite) == 2:
                writer.writerow(towrite)

        
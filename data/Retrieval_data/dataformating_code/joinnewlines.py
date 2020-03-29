
#below code will make the text in single line
with open ("aa.txt",'r') as file:
	lines = file.readlines()
	single_lines = '\t'.join([line.strip() for line in lines])

with open("a.txt",'w') as ff:
	ff.write(single_lines)

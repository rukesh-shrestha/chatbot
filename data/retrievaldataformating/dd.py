
#below code will make the text in single line
with open ("a.txt",'r') as file:
	lines = file.readlines()
	single_lines = '\t'.join([line.strip() for line in lines])

with open("aa.txt",'w') as ff:
	ff.write(single_lines) 

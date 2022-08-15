import csv


file = open('smarthome_plaintext.csv')
reader = csv.reader(file)
header_row = next(reader)



header = ['Age','Student Status', 'Highest education level completed', 
'Technology use at work', 'Experience', 'Gender identity',
'Datatype','Recipient','Condition','Class']

final_file = []
final_file.append(header)
import random


for k,v in enumerate(reader):
    random_value = random.randint(0,295400)
    import pdb;pdb.set_trace()

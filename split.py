import sys
from sklearn.model_selection import train_test_split

en = []
ar = []
with open(sys.argv[1]) as f:
	for line in f:
		en.append(line)
with open(sys.argv[2]) as f:
	for line in f:
		ar.append(line)

en_train, en_test, ar_train, ar_test = train_test_split(en, ar, test_size=0.4, random_state=42)
en_test, en_valid, ar_test, ar_valid = train_test_split(en_test, ar_test, test_size=0.25, random_state=42)

with open('fr-ar_ar_train.txt','w') as f0:
	for sent in ar_train:
		f0.write(sent)
with open('fr-ar_ar_test.txt','w') as f1:
	for sent in ar_test:
		f1.write(sent)
with open('fr-ar_ar_valid.txt','w') as f2:
	for sent in ar_valid:
		f2.write(sent)
with open('en-ar_ar_train.txt','w') as f3:
	for sent in en_train:
		f3.write(sent)
with open('en-ar_ar_test.txt','w') as f4:
	for sent in en_test:
		f4.write(sent)
with open('en-ar_ar_valid.txt','w') as f5:
	for sent in en_valid:
		f5.write(sent)

f0.close()
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
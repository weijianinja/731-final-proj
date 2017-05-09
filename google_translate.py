import sys
from googletrans import Translator

en = []
fr = []
with open(sys.argv[1]) as f0:
	for line in f0:
		en.append(line)
with open(sys.argv[2]) as f1:
	for line in f1:
		fr.append(line)


start = int(sys.argv[8])
end = int(sys.argv[9])

en = en[start:end]
fr = fr[start:end]
translator = Translator()

count = 0
with open(sys.argv[3],'w') as f2:
	with open(sys.argv[4],'w') as f3:
		with open(sys.argv[5],'w') as f4:
			with open(sys.argv[6],'w') as f5:
				with open(sys.argv[7],'w') as f6:
					for sent_fr,sent_en in zip(fr,en):
						try:
							#print sent_fr,sent_en
							fr_ar = translator.translate(sent_fr,src='fr',dest='ar').text.encode('utf8')
							en_ar = translator.translate(sent_en,src='en',dest='ar').text.encode('utf8')
							f2.write(en_ar+'\n')
							f2.flush()
							f3.write(fr_ar+'\n')
							f3.flush()
							f4.write(sent_en)
							f4.flush()
							f5.write(sent_fr)
							f5.flush()
							count += 1
							f6.write(str(count)+'\n')
							f6.flush()
						except ValueError:			   
							f6.write('error here!\n')
							f6.flush()
							

f2.close()
f3.close()
f4.close()
f5.close()
f6.close()

import os,loadaud,pickle
import numpy as np
k = os.listdir("19/198")
j = os.listdir("26/495")
l=os.listdir("4")
m=0
for i in l:
    if (m==0):
        m=1
        #print (i)
        print ("inif")
        ko=loadaud.get_audio("4/"+i)
        lo=loadaud.get_audio("19/198/"+(i.split("_"))[1]+".wav")
        
    else:
        print ("inelse")
        ko=np.concatenate((ko,loadaud.get_audio("4/"+i)),axis=0)
        lo=np.concatenate((lo,(loadaud.get_audio("19/198/"+(i.split("_"))[1]+".wav"))),axis=0)
        print (ko.shape)
print(ko.shape,lo.shape)        
filename = "truedata"
outfile = open(filename,'wb')    
pickle.dump(lo,outfile)
outfile.close()
flename = "labeldata"
otfile = open(flename,'wb')
pickle.dump(ko,otfile)
otfile.close()
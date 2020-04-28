

import webbrowser as wb


import csv
import numpy as np
from sklearn import tree

def main():
    
    count = 0
    cnt = 0
    l =0
    
    while True:
        
        x = np.genfromtxt('demox.csv', delimiter=',',)# skip_header=1) 
        y = np.genfromtxt('demoy.csv', delimiter=',',) 

        clf = tree.DecisionTreeClassifier()
        clf.fit(x,y)

        temp = np.genfromtxt('temp.csv', delimiter=',',)
        tmp = temp.reshape(1, -1)


        res = clf.predict(tmp)
        res = np.round(res,0)
        res = int(res)
        #print (res)


        if (count == 0):
            l = res

            count = count + 1

        else:

            if (cnt == 3):
                print(res)
                ress=str(res)
                with open('result.txt', 'a') as the_file:
                    the_file.write(ress)
                    
                wb.open_new_tab('results.html')
                break

            elif (l == res):
                cnt = cnt+1
               
                
            else:
                
                cnt=0
                

if __name__ == '__main__':    
    main()

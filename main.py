import numpy as np
import random as rd   
import os
import math
import greedi
import modulo
                        
#manipolazione dati                                                     #Se vuole fare il run dalla sua macchina potrebbe essere
os.chdir("C:\\Users\\39348\\Desktop\\resource optimization\\INSTANCES")  #necessario modificare questa riga
instance="PROBLEM_NGCUT01.ins2D_00.inp"
A=np.fromfile(instance,dtype=int, count=-1, sep=" ")

solution=0
sol_best=0
width_sol=0
width_best=0
w=A[2]
cost_shelfs=0
cost_best_shelfs=0
costs=0  
mas=0
mas_high=0
boole=0
value=0
num_greedy=90    

i=3
while(i<A.size):
    if A[i]>mas:
        mas=A[i]
    
    if A[i+1]>mas_high:
        mas_high=A[i+1]
    i=i+3
  
tempi=np.zeros((A[1],mas))
i=4
while(i<A.size):
    tempi[A[i+1]-1,A[i-1]-1]=A[i]
    i=i+3

for i in range (tempi[:,0].size):
    for j in range (tempi[0,:].size):
        if tempi[i,j]==0:
            tempi[i,j]=mas_high+1
#preparazione UB
UB=mas_high*tempi[:,0].size
#preparazione LB
LB=np.zeros((tempi[:,0].size,tempi[0,:].size))
for j in range (tempi[0,:].size):
    for i in range (tempi[:,0].size):
        LB[i,j]=tempi[i,j]*(j+1)

LB=math.ceil(sum(np.amin(LB,1))/w)

if LB< max( np.min(tempi,1)):
    LB=max( np.min(tempi,1))


width=min(w,tempi[:,0].size)
width_matrix=np.zeros(width)      
col=np.zeros(width)
num_iter=round(num_greedy/((tempi[0,:].size)*3))

#preparazione e realizzazione greedy
for j in range (tempi[0,:].size):
    prep=np.zeros((tempi[:,0].size,3))
    for i in range (tempi[:,0].size):
        boole=0
        prep[i,0]=i+1
        if tempi[i,j]== mas_high+1:
            if j!=tempi[0,:].size-1 and tempi[i,j+1]<mas_high+1:
                prep[i,1]=tempi[i,j+1]
                prep[i,2]=j+2
                boole=1
            if j!=0 and tempi[i,j-1]<mas_high+1 and boole==0:
                prep[i,1]=tempi[i,j-1]
                prep[i,2]=j
                
            else:
                h=0                                                         
                boole=0
                while boole==0:      
                    if tempi[i,h]<mas_high+1:
                        boole=1
                        prep[i,1]=tempi[i,h]
                        prep[i,2]=h+1
                    else:
                        h=h+1
        else:
            prep[i,1]=tempi[i,j]
            prep[i,2]=j+1
    for g in range (num_iter):
        if g==0:
            greedy=np.flipud(prep[prep[:,1].argsort()])
        else:
            s=0
            t=0
            n=0
            boole=0
            job_yep=np.zeros(tempi[:,0].size)
            job_nope=np.zeros(tempi[:,0].size)
            
            while s <(tempi[:,0].size):
                rando=rd.random()
                if rando>0.5 and boole==0:
                    greedy[t,:]=prep[s,:]
                    t=t+1
                    job_yep[s]=1
                    
                elif rando>0.5 and boole==1:
                    if job_yep[s]==0:
                        greedy[t,:]=prep[s,:]
                        t=t+1
                        
                elif rando<0.5 and boole==1:
                    if job_yep[s]==0:
                        job_nope[n]=s+1
                        n=n+1
                s=s+1
                if s==tempi[:,0].size-1 and boole==0: 
                    s=0
                    boole=1
                       
                    
            n=0
            while job_nope[n]!=0:
                greedy[t,:]=prep[job_nope.astype(int)[n]-1,:]
                n=n+1
                t=t+1

        solution,width_sol,value,cost_shelfs=greedi.next_fit(greedy,w,tempi)
        if value<UB:
            UB=value
            sol_best=solution
            width_best=width_sol
            cost_best_shelfs=cost_shelfs
        costs=np.append(costs,cost_shelfs)
        width_matrix=np.vstack((width_matrix,width_sol))
        col=np.vstack((col,solution))
        solution,width_sol,value,cost_shelfs=greedi.first_fit(greedy,w,tempi)
        if value<UB:
            UB=value
            sol_best=solution
            width_best=width_sol
            cost_best_shelfs=cost_shelfs
        costs=np.append(costs,cost_shelfs)
        width_matrix=np.vstack((width_matrix,width_sol))
        col=np.vstack((col,solution))
        solution,width_sol,value,cost_shelfs=greedi.best_fit(greedy,w,tempi)
        if value<UB:
            UB=value
            sol_best=solution
            width_best=width_sol
            cost_best_shelfs=cost_shelfs
        costs=np.append(costs,cost_shelfs)
        width_matrix=np.vstack((width_matrix,width_sol))
        col=np.vstack((col,solution))

print("UB dopo fase di realizzazione greedy e LS vale: ",UB)
#preparazione e realizzazione set covering euristico
costs=np.delete(costs,0)
width_matrix=np.delete(width_matrix,0,0)
col=np.delete(col,0,0).T
col_set=np.zeros((tempi[:,0].size,col[0,:].size))
for j in range(col_set[0,:].size):
    for i in range(col[:,0].size):
        if col[i,j]==0:
            break
        else:
            col_set[col.astype(int)[i,j]-1,j]=1
boole=0
num_copie=0
col_return=0
LB_set=0
value,LB_set,boole,num_copie,col_return=modulo.set_covering(col_set,costs,UB)

#manipolazione soluzione set covering

solution=np.zeros(width)
width_sol=np.zeros(width)
for i in range (col_return.size):
    solution=np.vstack((solution,col[:,col_return.astype(int)[i]]))
    width_sol=np.vstack((width_sol,width_matrix[col_return.astype(int)[i],:]))
solution=np.delete(solution,0,0)
width_sol=np.delete(width_sol,0,0)
cost_shelfs_set=np.zeros(solution[:,0].size)


for i in range (solution[:,0].size):
    for j in range(solution[0,:].size):
        if num_copie[solution.astype(int)[i,j]-1]==0 and solution[i,j]>0:
            if tempi[solution.astype(int)[i,j]-1,width_sol.astype(int)[i,j]-1]>cost_shelfs_set[i]:
                cost_shelfs_set[i]=tempi[(solution.astype(int)[i,j])-1,(width_sol.astype(int)[i,j])-1]
        elif num_copie[solution.astype(int)[i,j]-1]!=0 and solution[i,j]>0:
            num_copie[solution.astype(int)[i,j]-1]= num_copie[solution.astype(int)[i,j]-1]-1
            solution[i,j]=-1
            width_sol[i,j]=0

width_sol,cost_shelfs_set=greedi.get_better(tempi,solution,width_sol,cost_shelfs_set,w)

UB_set=sum(cost_shelfs_set)
print("")
print("UB dopo fase di realizzazione set covering vale: ",UB_set)

if UB_set<UB:
    UB=UB_set
    width_best=width_sol
    sol_best=solution
    cost_best_shelfs=cost_shelfs_set

if LB==UB:
    print(" ")
    print("Ho trovato la soluzione ottima")
    

#preparazione e stampa risultato
stampa_mat=np.zeros((tempi[:,0].size,4))                          
counter_width=0
counter_high=0
instance_out=instance.removesuffix("inp")+"sol"


for i in range (sol_best[:,0].size):
    counter_width=0
    if i==0:
        counter_high=0
    else:
        counter_high=counter_high+cost_best_shelfs[i-1]
        
    for j in range (sol_best[0,:].size):
        if sol_best [i,j]>0:
            stampa_mat[sol_best.astype(int) [i,j]-1,0]=width_best[i,j]
            stampa_mat[sol_best.astype(int) [i,j]-1,1]=tempi[(sol_best.astype(int)[i,j])-1,(width_best.astype(int)[i,j])-1]
            stampa_mat[sol_best.astype(int) [i,j]-1,2]=counter_width
            counter_width=counter_width+width_best[i,j]
            stampa_mat[sol_best.astype(int) [i,j]-1,3]=counter_high
            
            
with open(instance_out ,'w') as outfile:
    UB.tofile(outfile,sep="  ",format="%d")
    outfile.write("\n")
    np.savetxt(outfile,stampa_mat,fmt="%d",delimiter="  ")












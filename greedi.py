import numpy as np
def next_fit(greedy,w,tempi) :
    width=min(w,greedy[:,0].size)
    UB=0
    t=0
    shelf_level=0
    cost_shelfs=np.zeros(2)     
    solution=np.zeros((2,width))
    width_sol=np.zeros((2,width))
    counter_width=w
    for i in range (greedy[:,0].size):
        if greedy[i,2]<=counter_width:
            counter_width=counter_width-greedy[i,2]
            solution[shelf_level,t]=greedy[i,0]
            width_sol[shelf_level,t]=greedy[i,2]
            t=t+1
            if greedy[i,1]>cost_shelfs[shelf_level]:
               cost_shelfs[shelf_level]=greedy[i,1]
        else:
            shelf_level=shelf_level+1
            cost_shelfs=np.append(cost_shelfs,0)
            cost_shelfs[shelf_level]=greedy[i,1]
            t=0
            counter_width=w-greedy[i,2]
            solution=np.vstack((solution,np.zeros(width)))
            width_sol=np.vstack((width_sol,np.zeros(width)))
            solution[shelf_level,t]=greedy[i,0]
            width_sol[shelf_level,t]=greedy[i,2]
            t=t+1
    cost_shelfs=np.delete(cost_shelfs,shelf_level+1)
    solution=np.delete(solution,shelf_level+1,0)
    width_sol=np.delete(width_sol,shelf_level+1,0)
    width_sol,cost_shelfs=get_better(tempi,solution,width_sol,cost_shelfs,w)
    UB=sum(cost_shelfs)
    return solution,width_sol, UB,cost_shelfs
            
def first_fit (greedy,w,tempi):
    width=min(w,greedy[:,0].size)
    UB=0
    c=0
    boole=0
    shelf_level=0 #posizione dello shelf più alto che ho creato
    cost_shelfs=np.zeros(2)     
    solution=np.zeros((2,width))
    width_sol=np.zeros((2,width))
    counter_width=np.zeros(greedy[:,0].size)
    counter_width[0]=w   #vettore delle capcità residue dinamico
    t=np.zeros(2)        #vettore degli indici di ogni dinamico
    for i in range (greedy[:,0].size):
        c=0
        boole=0
        for c in range (counter_width.size):
            if counter_width[c]-greedy[i,2]>=0:
                boole=1
                counter_width[c]=counter_width[c]-greedy[i,2]
                solution[c,t.astype(int)[c]]=greedy[i,0]   
                width_sol[c,t.astype(int)[c]]=greedy[i,2]
                t[c]=t[c]+1
                if greedy[i,1]>cost_shelfs[c]:
                   cost_shelfs[c]=greedy[i,1]
                break

        if boole==0:      #necessario aprire uno shelf
            shelf_level=shelf_level+1
            solution=np.vstack((solution,np.zeros(width)))
            width_sol=np.vstack((width_sol,np.zeros(width)))
            cost_shelfs=np.append(cost_shelfs,0)
            t=np.append(t,0)
            cost_shelfs[shelf_level]=greedy[i,1]
            counter_width[shelf_level]=w-greedy[i,2]
            solution[shelf_level,0]=greedy[i,0]
            width_sol[shelf_level,0]=greedy[i,2]
            t[shelf_level]=t[shelf_level]+1

    cost_shelfs=np.delete(cost_shelfs,shelf_level+1)
    solution=np.delete(solution,shelf_level+1,0)
    width_sol=np.delete(width_sol,shelf_level+1,0)
    width_sol,cost_shelfs=get_better(tempi,solution,width_sol,cost_shelfs,w)
    UB=sum(cost_shelfs)
    return solution, width_sol, UB,cost_shelfs

def best_fit (greedy,w,tempi):
    width=min(w,greedy[:,0].size)
    UB=0
    c=0
    boole=0
    shelf_level=0 #posizione dello shelf più alto che ho creato
    cost_shelfs=np.zeros(2)     
    solution=np.zeros((2,width))
    width_sol=np.zeros((2,width))
    counter_width=np.zeros(greedy[:,0].size)
    counter_width[0]=w   #vettore delle capcità residue dinamico
    t=np.zeros(2)        #vettore degli indici di ogni dinamico
    for i in range (greedy[:,0].size):
        c=0
        c_best=0
        boole=0
        best=w+1
        for c in range (counter_width.size):
            if counter_width[c]-greedy[i,2]>=0:
                boole=1
                if counter_width[c]<best:
                    best=counter_width[c]
                    c_best=c

        if boole==1:
            counter_width[c_best]=counter_width[c_best]-greedy[i,2]
            solution[c_best,t.astype(int)[c_best]]=greedy[i,0]   
            width_sol[c_best,t.astype(int)[c_best]]=greedy[i,2]
            t[c_best]=t[c_best]+1
            if greedy[i,1]>cost_shelfs[c_best]:
                cost_shelfs[c_best]=greedy[i,1]
                

        if boole==0:      #necessario aprire uno shelf
            shelf_level=shelf_level+1
            solution=np.vstack((solution,np.zeros(width)))
            width_sol=np.vstack((width_sol,np.zeros(width)))
            cost_shelfs=np.append(cost_shelfs,0)
            t=np.append(t,0)
            cost_shelfs[shelf_level]=greedy[i,1]
            counter_width[shelf_level]=w-greedy[i,2]
            solution[shelf_level,0]=greedy[i,0]
            width_sol[shelf_level,0]=greedy[i,2]
            t[shelf_level]=t[shelf_level]+1

    cost_shelfs=np.delete(cost_shelfs,shelf_level+1)
    solution=np.delete(solution,shelf_level+1,0)
    width_sol=np.delete(width_sol,shelf_level+1,0)
    width_sol,cost_shelfs=get_better(tempi,solution,width_sol,cost_shelfs,w)
    UB=sum(cost_shelfs)
    return  solution,width_sol,UB,cost_shelfs

def get_better (tempi,solution,width_sol,cost_shelfs,w):
    boole=0
    mas_high=0
    k=0
    job=0
    width=0
    

    for i in range (solution[:,0].size):
        boole=0
        mas_high=0
        if sum(width_sol[i,:])!=w and sum(width_sol[i,:])!=0:
            while boole==0:
                boole=1
                mas_high=0
                for j in range (solution[0,:].size):
                    if solution[i,j]>0 and tempi[solution.astype(int)[i,j]-1,width_sol.astype(int)[i,j]-1]>mas_high:
                        mas_high=tempi[solution.astype(int)[i,j]-1,width_sol.astype(int)[i,j]-1]
                        k=j
                        job=solution[i,j]-1
                        width=width_sol[i,j]-1
                cost_shelfs[i]=mas_high
                for t in range (tempi[0,:].size):
                    if sum(width_sol[i,:])-(width+1)+(t+1)<=w and tempi[int(job),t]<tempi[int(job),int(width)]:
                        width_sol[i,k]=t+1
                        boole=0
                        break

    return width_sol, cost_shelfs
                        
                        
                    
                
            
            
            
            
                                       
    
        

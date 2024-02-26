def com(l1,l2):
    l3=[]
    for x in l1:
        for y in l2:
            if x==y:
                l3.append(x)
            else :
                break
    return l3
com([1,2,3,4],[3,4,5,6])
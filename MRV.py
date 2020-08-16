import sys
import time
from itertools import permutations


def processFile():
    filename = sys.argv[1] + '.txt'
    with open(filename) as f:
        h=0
        w = 0
        arr=[]

        for line in f:
            data=line.split()
            for i in range(0,len(data)):
                    arr.append(int(data[i]))

            """
            the first line of input txt file should treat carefully
            h and w should be calculated one more time
            because we need a new matrix to contain all the symbols
            """
            h=arr[0]
            h=h+h+1
            w=arr[1]
            w=w+w+1
            break

        Matrix = [[0 for x in range(w)] for y in range(h)]
        for m in range(0,h):
            for n in range(0,w):
                Matrix[m][n]=" "

        row=0
        for line in f:
            column = 0
            for j in range(0,len(line)):
                if ord(line[j])==10:
                    break

                """
                reset the two dimensional matrix
                """
                if ord(line[j])>=48 and ord(line[j])<=57:
                    Matrix[row][column]="."
                else:
                    Matrix[row][column]=line[j]
                column+=1
            row+=1
            if row==h:
                break
    return Matrix,h,w


"""
using BFS to add all tuples from specific region in an array 
"""
def processRegion(graph):
    g=graph
    array=[]
    stack=[]
    counter=0

    for i in range(1, h , 2):
        for j in range(1, w , 2):
            if (i,j) not in array:
                bypass=BFS(g,(i,j))
                for index in range(0,len(bypass)):
                    array.append(bypass[index])
                stack.append(bypass)
                counter+=1
    return stack

"""
BFS algorithm to search all the elements in one specific region
"""
def BFS(graph,s):
    queue = []
    queue.append(s)
    seen = set()
    seen.add(s)
    parent = {s: None}
    bypass=[]

    while len(queue) > 0:
        vertex = queue.pop(0)
        bypass.append(vertex)
        nodes = graph[vertex]
        for w in nodes:
            if w not in seen:
                queue.append(w)
                seen.add(w)
                parent[w] = vertex
    return bypass


"""
the boundary condition in the matrix
"""
def produceGraph(Matrix,h,w):
    graph = dict()
    for i in range(1, h-1 , 2):
        for j in range(1, w-1 , 2):
            graph[(i, j)] = []
            if j-2>=0:
                if Matrix[i][j-2] == "." and Matrix[i][j-1]!="|":
                    graph[(i, j)].append((i, j-2))
            if j+2<=w-1:
                if Matrix[i][j+2] == "." and Matrix[i][j+1]!="|":
                    graph[(i, j)].append((i , j+2))
            if i-2>=0:
                if Matrix[i-2][j ] == "." and Matrix[i-1][j]!="-":
                    graph[(i, j)].append((i-2, j ))
            if i+2<=h-1:
                if Matrix[i+2][j ] == "." and Matrix[i+1][j]!="-":
                    graph[(i, j)].append((i+2, j ))
    return graph

"""
convert the matrix to a user friendly smaller matrix
"""
def processStack(stack):
    x=1
    y=1
    for i in range(0,len(stack)):
        for j in range(0,len(stack[i])):
            if stack[i][j][0]!=1:
                temp=(stack[i][j][0]-1)/2
                temp=int(temp)
                x=stack[i][j][0]
                x=x-2*temp+temp
            if stack[i][j][1]!=1:
                temp=(stack[i][j][1]-1)/2
                temp=int(temp)
                y=stack[i][j][1]
                y=y-2*temp+temp
            stack[i][j]=(x,y)
            x=1
            y=1
    return stack


"""
quickSort to sort the region
from region which contains the least blocks to 
the region which contains the most blocks
"""
def partition(arr, low, high,brr):
    i = (low - 1)  # index of smaller element
    pivot = arr[high]  # pivot

    for j in range(low, high):

        # If current element is smaller than or
        # equal to pivot
        if arr[j] <= pivot:
            # increment index of smaller element
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]
            brr[i], brr[j] = brr[j], brr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    brr[i + 1], brr[high] = brr[high], brr[i + 1]
    return (i + 1)


"""
Function to do Quick sort
"""
def quickSort(arr, low, high,brr):
    if low < high:
        # pi is partitioning index, arr[p] is now
        # at right place
        pi = partition(arr, low, high,brr)
        # Separately sort elements before
        # partition and after partition
        quickSort(arr, low, pi - 1,brr)
        quickSort(arr, pi + 1, high,brr)


"""
this function calls quickSort function to sort the entire 
matrix from region to region
"""
def sortStack(stack):
    arr=[]
    brr=[]
    newStack=[]
    for i in range(0,len(stack)):
        arr.append(len(stack[i]))
        brr.append(i)
    quickSort(arr,0,len(arr)-1,brr)
    for j in range(0,len(stack)):
        newStack.append(stack[brr[j]])
    return newStack


"""
finally output the result which is user friendly
"""
def postProcess(matrix):
    for i in range(1,len(matrix)):
        matrix[i]=matrix[i][1:]
        print(matrix[i])

"""
class for the DFS algorithm
"""
from collections import defaultdict
class Graph:

    # Constructor
    def __init__(self):

        # default dictionary to store graph
        self.graph = defaultdict(list)
        self.flag=0
        self.mark=False
        self.forwardchecking=defaultdict(list)
        self.dic=defaultdict(list)
        self.count=0

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)
        # A function used by DFS

    """
    check the boundary to decide whether DFS should continue
    """
    def boundary(self,matrix,number,x,y,h,w):
        self.flag=0

        for i in range(1,number+1):
            if y+i<=w-1:
                if matrix[x][y+i]==number:
                    self.flag=1
                    break

        for j in range(1,number+1):
            if y-j>=1:
                if matrix[x][y-j]==number:
                    self.flag = 1
                    break
        for m in range(1,number+1):
            if x+m<=h-1:
                if matrix[x+m][y]==number:
                    self.flag = 1
                    break
        for n in range(1,number+1):
            if x-n>=1:
                if matrix[x-n][y]==number:
                    self.flag = 1
                    break
        if self.flag==1:
            return False
        else:
            return True

    """
    augment my solver for intelligent solutions
    implement the minimum-remaining-values heuristic function
    """
    def MRV(self,x,y,number,h,w):
        for i in range(1,number+1):
            if y+i<=w-1:
                self.forwardchecking[(x,y+i)].append(number)

        for j in range(1,number+1):
            if y-j>=1:
                self.forwardchecking[(x, y-j)].append(number)

        for m in range(1,number+1):
            if x+m<=h-1:
                self.forwardchecking[(x + m, y)].append(number)

        for n in range(1,number+1):
            if x-n>=1:
                self.forwardchecking[(x - n, y)].append(number)


    """
    DFS brute-force solver
    """
    def DFSUtil(self, current,end, visited,stack,matrix,h,w):

        self.count+=1

        # Mark the current node as visited and print it
        if current==end:
            return True

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[current]:

            """
            creat a container which contains all possibilities
            """
            arr = stack[i]
            array=[]
            for tem in range(1,len(arr)+1):
                array.append(tem)

            """
            according to the minimum-remaining values heuristic function
            immediately eliminate all possible values in other cells that 
            would be conflict
            """
            for var in range(0,len(arr)):
                if arr[var] in self.forwardchecking.keys():
                    self.dic[var].append(self.forwardchecking[arr[var]])


            brrused = list(permutations(array, len(arr)))
            brr=[]
            for n in range(0,len(brrused)):
                for p in range(0,len(brrused[n])):
                    if brrused[n][p]  in self.dic[n]:
                        break
                brr.append(brrused[n])

            """
            every time after doing MRV reset the dictionary
            """
            self.forwardchecking.clear()

            for index in range(0,len(brr)):

                for s in range(0, len(arr)):
                    x = arr[s][0]
                    y = arr[s][1]
                    number=brr[index][s]
                    matrix[x][y] = number

                    """
                    call minimum remaining values heuristic function
                    """
                    self.MRV(x, y, number, h, w)


                    # to judge whether the boundary condition is fulfilled
                    status=self.boundary(matrix,number,x,y,h,w)
                    if not status:
                        break
                if status:
                    mark=self.DFSUtil(i, end, visited, stack, matrix, h, w)
                    if mark==True:
                        return True

                """
                every time when one solution is not correct
                we need to erase all the numbers which are already filled 
                in the matrix
                """
                for index in range(0, len(arr)):
                    x = arr[index][0]
                    y = arr[index][1]
                    matrix[x][y] = 0
        return False

    # recursive DFSUtil()
    def DFS(self, start,end,stack,matrix,h,w):

        # Mark all the vertices as not visited
        visited =set()

        arr = stack[0]
        array = []
        for tem in range(1, len(arr) + 1):
            array.append(tem)

        brr = list(permutations(array, len(arr)))
        for index in range(0, len(brr)):
            if self.mark == True:
                break
            for s in range(0, len(arr)):
                """
                first array element in stack
                x is the x coordinate of tuple in the array
                """
                x = arr[s][0]
                y = arr[s][1]

                """
                get all possible values and add these values to the 
                region of the matrix
                """
                number = brr[index][s]
                matrix[x][y] = number

                # call minimum remaining values heuristic function
                self.MRV(x,y,number,h,w)

                """
                check whether the boundary condition is valid
                if not ,return to the last recursive layer
                """
                status=self.boundary(matrix, number, x, y, h, w)


                if not status:
                    break
            if status:
                """
                the boundary condition is valid
                enter the next layer
                """
                self.mark=self.DFSUtil(start, end, visited, stack, matrix, h, w)
                if self.mark==True:
                    break

            for index in range(0, len(arr)):
                x = arr[index][0]
                y = arr[index][1]
                matrix[x][y] = 0
        return matrix,self.count

        # Call the recursive helper function to print
        # DFS traversal


def searchDFS(stack,h,w):
    h=h/2
    h=int(h)
    h+=1
    w=w/2
    w=int(w)
    w+=1
    g=Graph()
    matrix = [[0 for x in range(w)] for y in range(h)]
    for i in range(0,len(stack)):
            g.addEdge(i,i+1)
    """
    determine the DFS recursive depth
    """
    lengthValue=len(g.graph.keys())
    lengthValue-=1
    return g.DFS(0,lengthValue,stack,matrix,h,w)


if __name__ == '__main__':
    start=time.time()
    Matrix,h,w=processFile()
    graph=produceGraph(Matrix,h,w)
    stack=processRegion(graph)
    stack=processStack(stack)
    stack=sortStack(stack)
    matrix,counter=searchDFS(stack,h,w)
    postProcess(matrix)
    end=time.time()
    print("number of calls to my solver function: " + str(counter))
    print("optimizing time is "+str(end-start))









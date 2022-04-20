import pyfma

class Matrix:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, numberOfRows, numberOfColumns, data):
        self.numberOfRows = numberOfRows
        self.numberOfColumns = numberOfColumns
        self.data = data

    def scl(self, scalar):
        result = []
        for i in range(self.numberOfRows):
            newRow = []
            for j in range(self.numberOfColumns):
                newRow.append(self.data[i][j] * scalar)
            result.append(newRow)
        return result
    
    def validityCheck(self):
        if (not self.data or not self.numberOfColumns or not self.numberOfRows):
            print("Invalid matrix")
            exit()
        if (self.numberOfRows != len(self.data)):
            print("Invalid number of lines in matrix")
            exit()
        for i in range(self.numberOfRows):
            if (self.numberOfColumns != len(self.data[i])):
                print("Invalid number of Columns in matrix")
                exit()
    
    def mulMatrix(self, m: "Matrix"):
        if (self.numberOfColumns != m.numberOfColumns
            or self.numberOfRows != m.numberOfRows
            or self.numberOfRows != self.numberOfColumns):
            print("Both matrices should be square matrices and should have the same dimensions")
            return
        m_map = []
        for i in range(self.numberOfRows):
            newRow = []
            for k in range(m.numberOfColumns):
                tmp = 0
                for j in range(self.numberOfColumns):
                    tmp = pyfma.fma(self.data[i][j], m.data[j][k], tmp)
                newRow.append(tmp)
            m_map.append(newRow)
        return m_map

    # def mulVector(self, v: vector.Vector):
    #     if (self.numberOfColumns != self.numberOfRows 
    #         or self.numberOfColumns != v.numberOfColumns):
    #         print("Matrix should be a square matrix and have the same amount of columns as the vector")
    #         return
    #     result = []
    #     for i in range(self.numberOfRows):
    #         tmp = 0
    #         for j in range(self.numberOfColumns):
    #             tmp = pyfma.fma(self.data[i][j], v.data[j], tmp)
    #         result.append(tmp)
    #     return result

    def transpose(self):
        m_transp = [[0 for x in range(self.numberOfColumns)] for y in range(self.numberOfRows)]
        for i in range(self.numberOfRows):
            for j in range(self.numberOfColumns):
                m_transp[j][i] = self.data[i][j]
        return m_transp



            
def createMatrix():
    r = int(input("enter rows: "))
    c = int(input("enter columns: "))

    m = []
    for i in range(r):
        l = []
        for j in range(c):
            print("enter value for row", i, "column", j)
            v = float(input())
            l.append(v)
        m.append(l)
    matrix = Matrix(r, c, m)
    return matrix


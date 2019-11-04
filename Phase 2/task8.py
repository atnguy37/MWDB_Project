from dimReduction import dimReduction
from PostgresDB import PostgresDB
arg = input("Please enter the home directory for the images "
            "(Default: C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Dataset\\) : ")
if arg == '':
    arg = 'C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Dataset\\'
dim = dimReduction(arg, '*.jpg')
db = PostgresDB(password='mynhandepg', host='localhost',
                        database='mwdb', user='postgres', port=5432)
conn = db.connect()
# Create for M and PCA
_, _, _, _ = dim.saveDim('m', 'pca', 'imagedata_m', 10, password ="mynhandepg", database="mwdb")
bin_matrix, feature_matrix = dim.binMat(conn, 'imagedata_m_pca')
# code to print
# print(bin_matrix[0][0])
dim.imgViz(bin_matrix)
for idx, sub in enumerate(feature_matrix):
    print('\nLatent Semantic {x}'.format(x=idx+1))
    for s in sub:
        print('Subject: ', s[0])
        print('Weight: ', s[1])

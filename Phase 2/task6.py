from dimReduction import dimReduction
from PostgresDB import PostgresDB
arg = input("Please enter the home directory for the images "
            "(Default: C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Dataset\\) : ")
if arg == '':
    arg = 'C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Dataset\\'
dim = dimReduction(arg, '*.jpg')
arg1 = input("Please enter the Subject ID you would want to compare: ")
db = PostgresDB(password='mynhandepg', host='localhost',
                        database='mwdb', user='postgres', port=5432)
conn = db.connect()
_, _, _, _ = dim.saveDim('l', 'svd', 'imagedata_l', 10, password ="mynhandepg", database="mwdb")
# Change the database name in case you want to test with a different combination of features and dim
sub_matrix = dim.subMatrix(conn, 'imagedata_l_svd', arg1, mat=False)
for sub in sub_matrix:
    print('Subject: ', sub[0])
    print('Weight:', sub[1])
    print('\n')

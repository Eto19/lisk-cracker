import csv
import argparse
import os.path

"""
Create list_01.txt file from blockchain.db lisk snapshot

usage :
python create_list_from_snapshot.py

download snapshot here : https://snapshots.lisk.io/main/


extract the blockchain.db file in same directory
"""

parser = argparse.ArgumentParser()
parser.add_argument('--input-db', dest="db_filename", default="blockchain.db", help="Filename of blockchain.db file, default : blockchain.db")
parser.add_argument('--min-balance', dest="min_bal", type=float, default=1, help="Minimum balance of account, default : 1")

args = parser.parse_args()

if not os.path.isfile(str(args.db_filename)):
    print('Error : db file not found')
    print('download snapshot here : https://snapshots.lisk.io/main/')
    print('extract the blockchain.db file in same directory')
    exit()

db_filename = str(args.db_filename)
minimum_balance = float(args.min_bal)

start_mem_account = 'COPY public.mem_accounts (username, "isDelegate", "secondSignature", address, "publicKey", "secondPublicKey", balance, vote, rank, delegates, multisignatures, multimin, multilifetime, nameexist, "producedBlocks", "missedBlocks", fees, rewards, asset) FROM stdin;'
end_mem_account = '\\.'

account_mem = ''

print('Reading ' + str(args.db_filename) + ' file...')

with open(db_filename) as infile, open('account_python.txt', 'w+') as outfile:
    copy = False
    for line in infile:
        if line.strip() == start_mem_account:
            copy = True
            continue
        elif line.strip() == end_mem_account:
            copy = False
            continue
        elif copy:
            outfile.write(line)


account_list_01 = []

print('Parsing accounts...')

with open('account_python.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter="\x09")
    line_count = 0
    for row in csv_reader:
        if (row[4] == '\\N') and (int(row[3].replace('L', '')) < 2**64) and ((int(row[3][0]) != 0) or row[3] == '0L') and (float(row[6]) <  (100000000*(10**8))) and (float(row[6]) > (minimum_balance*(10**8))):
            address = row[3].replace('L', '')
            account_list_01.append([address, float(row[6])/(10**8)])

account_list_02 = sorted(account_list_01,key=lambda x: x[1], reverse=True)

f = open('list_01.txt', 'w+')

i = 0
for account in account_list_02:
    f.write(str(i)+','+account[0]+','+str(account[1])+'\n')
    i += 1

f.close()
print('done. Parsed '+ str(i) + ' accounts')
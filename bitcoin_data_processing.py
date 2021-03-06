# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 19:49:53 2021

@author: mlapa
"""

import json
import csv
from urllib.request import urlopen

# This function computes the total input amount in a transaction.
def input_amount(input_list):
    answer = 0
    for item in input_list:
        answer += item["prev_out"]["value"]
    
    return answer

# This function computes the total input amount in a transaction.
def output_amount(output_list):
    answer = 0
    for item in output_list:
        answer += item["value"]
    
    return answer
    
# This function counts the number of unique input addresses
def num_unique_input_addresses(input_list):
    d = {}
    for item in input_list:
        if item["prev_out"]["addr"] not in d:
            d[item["prev_out"]["addr"]] = 1
        else:
            d[item["prev_out"]["addr"]] += 1
        
    return len(d)

# Next, we load the data of a particular block in the Bitcoin blockchain
# using the Blockchain Data API from 
# https://www.blockchain.com/api/blockchain_api

# The particular blocks that we are downloaded all contain famous transactions
# from the history of Bitcoin. They are all described in this article:
# https://news.bitcoin.com/eight-historic-bitcoin-transactions/

# Load data of block 170. Satoshi Nakamoto sends 50 BTC to Hal Finney.
# with urlopen("https://blockchain.info/rawblock/00000000d1145790a8694403d4063f323d499e655c83426834d4ce2f8dd4a2ee") as response:
#     source = response.read()
    
# Load data of block 132749. Huge transaction by Mt. Gox CEO.
# with urlopen("https://blockchain.info/rawblock/00000000000004bea72d0f390194b08162665a4fc99469c576338cd37164a15a") as response:
#     source = response.read()

# Load data of block 228940. Large payment linked to Bitcoin fake murder story. 
# with urlopen("https://blockchain.info/rawblock/0000000000000156fad2c13be218e4c1f2c5101177717deab97859e08c0f8644") as response:
#     source = response.read()

# Load data of block 236502. Accidental large transaction fee. 
with urlopen("https://blockchain.info/rawblock/000000000000015fdabaddbdf1bc139849594152dd451059fba863d434561552") as response:
    source = response.read()
    
data = json.loads(source)

# We remove the first transaction (the coinbase transaction that mints new
# Bitcoin and rewards the miner) from the list of transactions.
transactions = data["tx"][1:]

# Process the data using our functions from above, then write to a csv file.
with open("block_236502_data.csv", mode = "w") as csv_file:
    fieldnames = ["transaction_hash", "num_unique_input_addresses", "total_input", "transaction_fee"]
    writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
    
    writer.writeheader()
    for t in transactions:
        unique_inputs = num_unique_input_addresses(t["inputs"])
        total_input = input_amount(t["inputs"])
        total_output = output_amount(t["out"])
        transaction_fee = total_input - total_output
                                   
        writer.writerow({"transaction_hash": t["hash"], "num_unique_input_addresses": unique_inputs, "total_input": total_input, "transaction_fee": transaction_fee})






    

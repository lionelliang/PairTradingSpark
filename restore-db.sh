#!/usr/bin/env bash
mongoimport stock_pairs_list_header45.csv --type csv --headerline -d stock -c pairs45
mongoimport stock_pairing_list_header100.csv --type csv --headerline -d stock -c pairs4k
mongoimport stock_pairing_list_header12k.csv --type csv --headerline -d stock -c pairs12k
mongoimport stock_pairing_list_all_header.csv --type csv --headerline -d stock -c pairsall
mongoimport stock_pairs_mongoexport.csv --type csv --headerline -d stock -c pairs45optm
mongoimport stock_pairs_mongoexport4k.csv --type csv --headerline -d stock -c pairs4koptm

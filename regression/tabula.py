import tabula

tables = tabula.read_pdf("zip_county.pdf", pages='all', multiple_tables=True)
print(tables[0].head())
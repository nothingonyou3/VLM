import csv

def filter_csv(input_file_path, output_file_path, keyword):
    try:
        with open(input_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)  # Legge la prima riga come intestazione
            
            filtered_rows = [headers]  # Inizia con le intestazioni
            
            for row in reader:
                if len(row) > 5:
                    pathology_list = row[5].strip('[]').split(', ')
                    # Rimuove eventuali virgolette intorno agli elementi della lista
                    pathology_list = [item.strip().strip('\'"') for item in pathology_list]
                    if keyword in pathology_list:
                        print("Keyword trovata")
                        filtered_rows.append(row)
            
        with open(output_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(filtered_rows)
            
        print(f"File filtrato salvato come {output_file_path}")
        
    except FileNotFoundError:
        print(f"Il file {input_file_path} non è stato trovato.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

# Esempio di utilizzo
input_file_path = '/home/giuliavanzato/Desktop/Quilt_complete/quilt_1M_lookup.csv'
output_file_path = '/home/giuliavanzato/Desktop/Quilt_complete/FILTRATO_OGGI.csv'
keyword = 'Breast'

filter_csv(input_file_path, output_file_path, keyword)

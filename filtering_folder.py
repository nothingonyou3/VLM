import csv
import os
import shutil


def copy_filtered_images(csv_file_path, source_dir, destination_dir):
    try:
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        
        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            image_paths = [row['image_path'] for row in reader]
            
            for image_name in image_paths:
                source_path = os.path.join(source_dir, image_name)
                destination_path = os.path.join(destination_dir, image_name)
                
                if os.path.exists(source_path):
                    shutil.copy2(source_path, destination_path)
                    print(f"Copiato: {image_name}")
                    print("FILE MATCHATO E SPOSTATO!")
                else:
                    print(f"File non trovato: {image_name}")
                    
        print("Copia completata.")
    
    except FileNotFoundError:
        print(f"Il file CSV {csv_file_path} non è stato trovato.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

# Esempio di utilizzo



output_file_path = '/home/giuliavanzato/Desktop/Quilt_complete/FILTRATO_OGGI.csv'
keyword = 'Breast'



source_dir = '/home/giuliavanzato/Desktop/Quilt_complete/images_part_2/quilt_1m'
destination_dir = '/home/giuliavanzato/Desktop/Quilt_complete/Destinazione_1'
copy_filtered_images(output_file_path, source_dir, destination_dir)

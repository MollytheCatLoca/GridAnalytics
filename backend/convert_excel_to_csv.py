import pandas as pd
import os

def convert_excel_to_csv(excel_file_path, output_path=None):
    """
    Convierte un archivo Excel a CSV
    
    Args:
        excel_file_path (str): Ruta del archivo Excel
        output_path (str): Ruta de salida para el CSV (opcional)
    """
    try:
        # Leer el archivo Excel
        print(f"Leyendo archivo Excel: {excel_file_path}")
        
        # Leer todas las hojas del Excel
        excel_file = pd.ExcelFile(excel_file_path)
        print(f"Hojas encontradas: {excel_file.sheet_names}")
        
        # Si no se especifica ruta de salida, usar el mismo nombre pero con extensión .csv
        if output_path is None:
            base_name = os.path.splitext(excel_file_path)[0]
            output_path = f"{base_name}.csv"
        
        # Si hay múltiples hojas, procesar cada una
        if len(excel_file.sheet_names) > 1:
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
                sheet_output_path = f"{os.path.splitext(output_path)[0]}_{sheet_name}.csv"
                df.to_csv(sheet_output_path, index=False, encoding='utf-8')
                print(f"Hoja '{sheet_name}' convertida a: {sheet_output_path}")
                print(f"Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")
                print(f"Columnas: {list(df.columns)}")
                print("---")
        else:
            # Solo una hoja
            df = pd.read_excel(excel_file_path)
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Archivo convertido a: {output_path}")
            print(f"Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")
            print(f"Columnas: {list(df.columns)}")
            
            # Mostrar primeras filas para vista previa
            print("\nPrimeras 5 filas:")
            print(df.head())
        
    except Exception as e:
        print(f"Error al convertir el archivo: {str(e)}")

if __name__ == "__main__":
    # Ruta del archivo Excel
    excel_path = "public/Mediciones Originales CEB .xlsx"
    
    # Convertir a CSV
    convert_excel_to_csv(excel_path)